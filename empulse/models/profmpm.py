from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from .._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ..metrics import BaseMetric, MaxProfit
from .csclassifier import CostSensitiveClassifier, MetricStrategyFactory


class ProfMPMClassifier(CostSensitiveClassifier):
    """
    Profit-driven minimax probability machine classifier with a shared worst-case accuracy bound.

    Learns a linear decision boundary that maximizes the worst-case (distribution-free)
    expected profit, using only the empirical means and covariances of each class
    through the multivariate Chebyshev-Cantelli inequality.

    Unlike :class:`~empulse.models.ProfMEMPMClassifier`, which allows the worst-case class
    accuracies to differ between classes, this model constrains both classes to share the
    same worst-case accuracy bound, determined by the tighter (minimum) of the two class bounds.

    Setting ``lambda_reg=0`` (default) reproduces the original Profit Maximizing Minimax
    Probability Machine (ProfMPM): the weight vector is unconstrained and no additional
    regularization is applied.

    Setting ``lambda_reg>0`` switches to the Lp-regularized variant (Lp-ProfMPM):
    an L1 or L2 penalty (controlled by ``penalty``) on the weight vector is added
    to the objective, controlling the scale of ``w``.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    Parameters
    ----------
    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            Since this model only supports class-dependent costs, array-like costs
            are aggregated to their mean value before fitting.

    loss : :class:`empulse.metrics.BaseMetric` or None, default=None
        Only :class:`~empulse.metrics.BaseMetric` instances built with the
        :class:`~empulse.metrics.MaxProfit` strategy are supported, since this model requires
        the costs and benefits to be reducible to four scalar values.

        .. note::
            If the costs or benefits contain stochastic variables, they are replaced by
            their mean/expectation before fitting.

        If :class:`~empulse.metrics.BaseMetric`, metric parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.ProfMPMClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    penalty : 'l1' or 'l2', default='l2'
        Norm used in the regularization term. Only used when ``lambda_reg > 0``.

    lambda_reg : float, default=0.0
        Regularization strength of the ``penalty`` term. Must be non-negative.

        If ``0.0``, no regularization is applied, reproducing the original
        (non-regularized) ProfMPM formulation.

        If greater than ``0.0``, ``w`` is regularized, reproducing the Lp-ProfMPM formulation.

    ridge_penalty : float, default=1e-6
        Small positive value added to the diagonal of the empirical covariance matrices
        to keep them positive definite (Tikhonov/ridge stabilization).
        This is independent of ``lambda_reg`` and is always applied.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    coef_ : numpy.ndarray, shape=(n_features,)
        Coefficients of the linear decision boundary.

    intercept_ : float
        Intercept of the linear decision boundary.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfMPMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfMPMClassifier()
        model.fit(X, y, tp_cost=-200, fp_cost=10)

    References
    ----------
    .. [1] Bravo, C., & Vanderschueren, T. (2023, September). Profit maximizing
        distribution-free classifiers: a study on the minimax probability machine.
        In Joint European Conference on Machine Learning and Knowledge Discovery in Databases.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'penalty': [StrOptions({'l1', 'l2'})],
        'lambda_reg': [Interval(Real, 0, None, closed='left')],
        'ridge_penalty': [Interval(Real, 0, None, closed='left')],
    }
    _default_metric_strategy: ClassVar[MetricStrategyFactory] = MaxProfit

    def __init__(
        self,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: BaseMetric | None = None,
        penalty: Literal['l1', 'l2'] = 'l2',
        lambda_reg: float = 0.0,
        ridge_penalty: float = 1e-6,
    ) -> None:
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.ridge_penalty = ridge_penalty
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        tp_benefit, tn_benefit, fp_cost, fn_cost = self._prepare_class_costs(loss_params)

        pos_mask = y == 1
        neg_mask = y == 0

        mu_1 = np.mean(X[pos_mask], axis=0)
        mu_0 = np.mean(X[neg_mask], axis=0)

        n_features = X.shape[1]
        ridge = np.eye(n_features) * self.ridge_penalty
        sigma_1 = np.cov(X[pos_mask], rowvar=False).reshape(n_features, n_features) + ridge
        sigma_0 = np.cov(X[neg_mask], rowvar=False).reshape(n_features, n_features) + ridge

        pi_1 = float(np.mean(pos_mask))
        pi_0 = float(np.mean(neg_mask))

        regularized = self.lambda_reg > 0

        def objective(params: FloatNDArray) -> float:
            w = params[:-1]
            b = params[-1]

            denom_1 = np.sqrt(w.T @ sigma_1 @ w)
            denom_0 = np.sqrt(w.T @ sigma_0 @ w)

            if denom_1 == 0 or denom_0 == 0:
                return np.inf

            # Apply the Chebyshev-Cantelli inequality transformation to evaluate bounding values.
            k_1 = (w.T @ mu_1 + b) / denom_1
            k_0 = (-(w.T @ mu_0 + b)) / denom_0

            # ProfMPM forces alpha_1 = alpha_0 = alpha.
            # The maximum valid kappa for a shared alpha is constrained by the tighter of the two class bounds.
            k_min = min(k_1, k_0)

            # Compute the shared worst-case accuracy bound alpha.
            alpha = (k_min**2 / (1 + k_min**2)) if k_min > 0 else 0.0

            expected_profit = pi_1 * (alpha * tp_benefit - (1 - alpha) * fn_cost) + pi_0 * (
                alpha * tn_benefit - (1 - alpha) * fp_cost
            )

            if regularized:
                if self.penalty == 'l1':
                    reg_term = self.lambda_reg * np.sum(np.abs(w))
                else:
                    reg_term = self.lambda_reg * 0.5 * np.sum(w**2)
            else:
                reg_term = 0.0

            # Minimize negative profit + regularization penalty
            return -expected_profit + reg_term

        w0 = np.ones(n_features) / np.sqrt(n_features)
        b0 = 0.0
        initial_params = np.append(w0, b0)

        self.result_ = minimize(
            objective,
            initial_params,
            method='SLSQP',
        )

        self.coef_ = self.result_.x[:-1]
        self.intercept_ = self.result_.x[-1]

        return self

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, 2)
            Predicted probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        scores = X @ self.coef_ + self.intercept_
        y_score = expit(scores)
        return np.vstack((1 - y_score, y_score)).T
