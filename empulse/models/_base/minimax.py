from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, MaxProfit
from ..csclassifier import CostSensitiveClassifier, MetricStrategyFactory


class BaseMinimaxProbabilityMachine(CostSensitiveClassifier, ABC):
    """Shared fit/predict_proba logic for the profit-driven minimax probability machine models.

    :class:`~empulse.models.ProfMPMClassifier` and :class:`~empulse.models.ProfMEMPMClassifier`
    differ only in how the worst-case class accuracy bound(s) are derived from the
    Chebyshev-Cantelli kappa values, and in whether a unit-norm constraint applies when
    unregularized. Subclasses implement :meth:`_worst_case_accuracies` and may override
    :meth:`_build_constraints` to plug in those differences.
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

    @abstractmethod
    def _worst_case_accuracies(self, k_1: float, k_0: float) -> tuple[float, float]:
        """Return ``(alpha_1, alpha_0)``, the worst-case accuracy bound(s) for each class."""

    def _build_constraints(self, *, regularized: bool) -> dict[str, Any] | tuple[()]:
        """Return the ``scipy.optimize.minimize`` constraints to use for this fit.

        The default is no constraints. Overridden by :class:`~empulse.models.ProfMEMPMClassifier`
        to add a unit-norm constraint when unregularized.
        """
        return ()

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

            alpha_1, alpha_0 = self._worst_case_accuracies(k_1, k_0)

            expected_profit = pi_1 * (alpha_1 * tp_benefit - (1 - alpha_1) * fn_cost) + pi_0 * (
                alpha_0 * tn_benefit - (1 - alpha_0) * fp_cost
            )

            if regularized:
                if self.penalty == 'l1':
                    reg_term = self.lambda_reg * np.sum(np.abs(w))
                else:
                    reg_term = self.lambda_reg * 0.5 * np.sum(w**2)
            else:
                reg_term = 0.0

            # Minimize negative profit + regularization penalty
            return float(-expected_profit + reg_term)

        w0 = np.ones(n_features) / np.sqrt(n_features)
        b0 = 0.0
        initial_params = np.append(w0, b0)

        constraints = self._build_constraints(regularized=regularized)

        self.result_: OptimizeResult = minimize(  # type: ignore[call-overload]
            objective,
            initial_params,
            method='SLSQP',
            constraints=constraints,
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
