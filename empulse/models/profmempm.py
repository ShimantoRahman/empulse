from typing import Any

import numpy as np

from ._base import BaseMinimaxProbabilityMachine


class ProfMEMPMClassifier(BaseMinimaxProbabilityMachine):
    """
    Profit-driven minimax probability machine classifier.

    Learns a linear decision boundary that maximizes the worst-case (distribution-free)
    expected profit, using only the empirical means and covariances of each class
    through the multivariate Chebyshev-Cantelli inequality.

    Setting ``lambda_reg=0`` (default) reproduces the original Profit Maximizing Minimax
    Probability Machine (MEMPM): the weight vector is constrained to unit norm (``||w||=1``)
    and no additional regularization is applied.

    Setting ``lambda_reg>0`` switches to the Lp-regularized variant (Lp-ProfMEMPM):
    the unit-norm constraint is dropped and an L1 or L2 penalty (controlled by ``penalty``)
    on the weight vector is added to the objective instead, controlling the scale of ``w``.

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
        to the :meth:`~empulse.models.ProfMEMPMClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    penalty : 'l1' or 'l2', default='l2'
        Norm used in the regularization term. Only used when ``lambda_reg > 0``.

    lambda_reg : float, default=0.0
        Regularization strength of the ``penalty`` term. Must be non-negative.

        If ``0.0``, no regularization is applied and ``w`` is instead constrained
        to unit norm, reproducing the original (non-regularized) MEMPM formulation.

        If greater than ``0.0``, the unit-norm constraint is dropped and ``w`` is
        regularized instead, reproducing the Lp-ProfMEMPM formulation.

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

        from empulse.models import ProfMEMPMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfMEMPMClassifier()
        model.fit(X, y, tp_cost=-200, fp_cost=10)

    References
    ----------
    .. [1] Bravo, C., & Vanderschueren, T. (2023, September). Profit maximizing
        distribution-free classifiers: a study on the minimax probability machine.
        In Joint European Conference on Machine Learning and Knowledge Discovery in Databases.
    """

    def _worst_case_accuracies(self, k_1: float, k_0: float) -> tuple[float, float]:
        alpha_1 = (k_1**2 / (1 + k_1**2)) if k_1 > 0 else 0.0
        alpha_0 = (k_0**2 / (1 + k_0**2)) if k_0 > 0 else 0.0
        return alpha_1, alpha_0

    def _build_constraints(self, *, regularized: bool) -> dict[str, Any] | tuple[()]:
        if regularized:
            return ()
        # Fix the scale invariance by constraining the L2 norm of the weight vector to 1.
        return {'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:-1]) - 1.0}
