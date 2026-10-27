from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.special import expit
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..._common._sklearn_compat import validate_data
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, MaxProfit
from .cost_sensitive import CostSensitiveClassifier, MetricStrategyFactory

if TYPE_CHECKING:
    from scipy.optimize._typing import Constraint


class BaseMinimaxProbabilityMachine(CostSensitiveClassifier, ABC):
    """Shared fit/predict_proba logic for the profit-driven minimax probability machine models.

    :class:`~empulse.models.ProfMPMClassifier` and :class:`~empulse.models.ProfMEMPMClassifier`
    differ only in how the worst-case class accuracy bound(s) are derived from the
    Chebyshev-Cantelli kappa values: :class:`~empulse.models.ProfMPMClassifier` forces both
    classes to share the tighter of the two bounds, while
    :class:`~empulse.models.ProfMEMPMClassifier` lets them differ. Subclasses implement
    :meth:`_worst_case_accuracies` to plug in that difference; every ``_solve_*`` method below
    derives its ``kappa``/``beta`` values from the optimization and defers to it for the
    final ``(alpha_1, alpha_0)``.
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

    @abstractmethod
    def _fit_minimax(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        """Solve the minimax problem and return (coef, intercept, alpha_1, alpha_0, result)."""

    @staticmethod
    def _margin_constraints(
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        unpack: Callable[[FloatNDArray], tuple[FloatNDArray, float]],
    ) -> list['Constraint']:
        """Scale-fixing constraints of the regularized MEMPM (Eq. 9 in Maldonado et al. 2020).

        w^T mu_1 + b >= 1 and -(w^T mu_0 + b) >= 1.

        The Chebyshev constraints are positively homogeneous in (w, b): scaling both by any t > 0
        keeps them satisfied at the same kappa. Without a constraint that fixes the scale, the
        penalty lambda * rho(w) is driven towards zero by shrinking w alone, so regularization only
        ever changed the size of w -- never its direction, the worst-case accuracies or a single
        prediction. Requiring each class mean to lie at least one unit on its own side of the
        hyperplane, as the regularized MEMPM of Eq. (9) that Sections 4.2 and 4.3 extend does, is
        what makes the penalty trade off against the worst-case accuracies. ``unpack`` maps the
        solver's parameter vector to ``(w, b)``.
        """

        def positive_margin(params: FloatNDArray) -> float:
            w, b = unpack(params)
            return float(w @ mu_1 + b) - 1.0

        def negative_margin(params: FloatNDArray) -> float:
            w, b = unpack(params)
            return -float(w @ mu_0 + b) - 1.0

        return [{'type': 'ineq', 'fun': positive_margin}, {'type': 'ineq', 'fun': negative_margin}]

    def _margin_feasible_mpm(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
    ) -> tuple[FloatNDArray, float]:
        """Solve the unregularized MPM and rescale it to satisfy :meth:`_margin_constraints`.

        Its intercept puts each class mean kappa * d_i > 0 inside its own half-space, so dividing
        (w, b) by the smaller of the two margins meets both margin constraints, and by homogeneity
        leaves the Chebyshev constraints satisfied at the same kappa.
        """
        w, b, _, _, _ = self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)
        margin = min(float(w @ mu_1 + b), -float(w @ mu_0 + b))
        if margin > 0:
            w, b = np.asarray(w, dtype=np.float64) / margin, b / margin
        return w, b

    def _solve_unregularized_mpm(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        """Solve unregularized MPM (Eq. 7 in Maldonado et al. 2020 / Lanckriet et al. 2003).

        min_w sqrt(w^T Sigma_1 w) + sqrt(w^T Sigma_0 w) s.t. w^T(mu_1 - mu_0) = 1.
        """
        delta_mu = np.asarray(mu_1 - mu_0, dtype=np.float64)
        delta_norm_sq = float(delta_mu @ delta_mu)
        w0 = np.asarray(delta_mu / max(delta_norm_sq, 1e-12), dtype=np.float64)

        def obj(w: FloatNDArray) -> float:
            return float(np.sqrt(w @ sigma_1 @ w) + np.sqrt(w @ sigma_0 @ w))

        constraints = {'type': 'eq', 'fun': lambda w: float(w @ delta_mu - 1.0)}
        res: OptimizeResult = minimize(obj, w0, method='SLSQP', constraints=constraints)  # type: ignore[call-overload]

        w_star = res.x
        d1 = float(np.sqrt(w_star @ sigma_1 @ w_star))
        d0 = float(np.sqrt(w_star @ sigma_0 @ w_star))
        denom = max(d1 + d0, 1e-12)
        kappa = 1.0 / denom
        alpha_1, alpha_0 = self._worst_case_accuracies(kappa, kappa)

        # Intercept from Eq. (29): b* = -w*^T mu_1 + kappa* d_1 = -w*^T mu_0 - kappa* d_0
        b_1 = -float(w_star @ mu_1) + kappa * d1
        b_0 = -float(w_star @ mu_0) - kappa * d0
        b_star = 0.5 * (b_1 + b_0)

        return w_star, b_star, alpha_1, alpha_0, res

    def _solve_unregularized_mempm(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        """Solve unregularized ProfMEMPM (Eq. 12-16, 24 in Maldonado et al. 2020).

        Algorithm 1: search over beta in (0, 1) maximizing f_profit(w*(beta), beta),
        where w*(beta) solves inner fractional program Eq. (24) under w^T(mu_1 - mu_0) = 1.
        """
        w_mpm, _, _, _, _ = self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)
        delta_mu = mu_1 - mu_0
        constraints = {'type': 'eq', 'fun': lambda w: float(w @ delta_mu - 1.0)}

        # Upper bound for kappa(beta): the supremum of (w^T delta_mu) / sqrt(w^T Sigma_0 w) over
        # w^T delta_mu = 1 is sqrt(delta_mu^T Sigma_0^-1 delta_mu), attained at w proportional to
        # Sigma_0^-1 delta_mu (Cauchy-Schwarz). Bounding it instead by d0 at the unrelated MPM
        # solution w_mpm can leave most of the feasible (0, 1) range for beta unexplored whenever
        # Sigma_0 and Sigma_1 differ in shape.
        kappa_0_sup_sq = max(float(delta_mu @ np.linalg.solve(sigma_0, delta_mu)), 0.0)
        kappa_max = 0.999 * float(np.sqrt(kappa_0_sup_sq))
        beta_max = min(float(kappa_max**2 / (kappa_max**2 + 1.0)), 0.999)

        def _solve_inner(beta_val: float) -> tuple[FloatNDArray, float, float]:
            k0 = float(np.sqrt(beta_val / (1.0 - beta_val)))

            def inner_obj(w: FloatNDArray) -> float:
                d1 = np.sqrt(w @ sigma_1 @ w)
                d0 = np.sqrt(w @ sigma_0 @ w)
                gamma = (1.0 - k0 * d0) / max(float(d1), 1e-12)
                return float(-gamma)

            res_inner = minimize(inner_obj, w_mpm, method='SLSQP', constraints=constraints)  # type: ignore[call-overload]
            w_sol = res_inner.x
            d1_sol = float(np.sqrt(w_sol @ sigma_1 @ w_sol))
            d0_sol = float(np.sqrt(w_sol @ sigma_0 @ w_sol))
            gamma_sol = (1.0 - k0 * d0_sol) / max(d1_sol, 1e-12)
            return w_sol, gamma_sol, d1_sol

        def neg_profit(beta_val: float) -> float:
            _, gamma_val, _ = _solve_inner(beta_val)
            if gamma_val <= 0:
                return 1e9
            alpha_1 = float(gamma_val**2 / (gamma_val**2 + 1.0))
            profit = c1 * alpha_1 + c0 * beta_val
            return -profit

        if beta_max <= 0:  # identical class means: no direction separates them
            return self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)
        scalar_res = minimize_scalar(neg_profit, bounds=(min(1e-4, 0.5 * beta_max), beta_max), method='bounded')
        best_beta = float(scalar_res.x)

        w_star, gamma_star, d1_star = _solve_inner(best_beta)
        if gamma_star <= 0:
            return self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)

        k0_star = float(np.sqrt(best_beta / (1.0 - best_beta)))
        d0_star = float(np.sqrt(w_star @ sigma_0 @ w_star))
        alpha_1, alpha_0 = self._worst_case_accuracies(gamma_star, k0_star)

        b_1 = -float(w_star @ mu_1) + gamma_star * d1_star
        b_0 = -float(w_star @ mu_0) - k0_star * d0_star
        b_star = 0.5 * (b_1 + b_0)

        res = OptimizeResult(  # type: ignore[call-arg]
            x=np.append(w_star, b_star),
            success=True,
            status=0,
            message='Optimization terminated successfully',
        )
        return w_star, b_star, alpha_1, alpha_0, res

    def _solve_regularized(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
        *,
        shared_beta: bool,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        """
        Solve regularized Lp-ProfMPM or Lp-ProfMEMPM (Algorithm 2).

        With *shared_beta*, one beta bounds both classes (Lp-ProfMPM, Section 4.2); otherwise each class
        has its own (Lp-ProfMEMPM, Section 4.3). The parameters are the weights, the intercept and the
        betas, ``beta_0`` before ``beta_1``. Under the L1 penalty the weights are ``u - v`` with ``u`` and
        ``v`` non-negative, so the penalty is smooth.
        """
        w, b = self._margin_feasible_mpm(mu_1, mu_0, sigma_1, sigma_0)
        n_features = len(mu_1)
        l1 = self.penalty != 'l2'
        n_weights = 2 * n_features if l1 else n_features
        betas = [1.0] if shared_beta else [1.0, 1.0]
        t1, t2 = 1.0, 1.0

        def unpack(params: FloatNDArray) -> tuple[FloatNDArray, float, float, float]:
            """Return the weights, intercept, beta_0 and beta_1; a shared beta is both."""
            w_curr = params[:n_features] - params[n_features:n_weights] if l1 else params[:n_features]
            return w_curr, params[n_weights], params[n_weights + 1], params[n_weights + len(betas)]

        def objective(params: FloatNDArray) -> float:
            if l1:
                reg = self.lambda_reg * float(np.sum(params[:n_features] + params[n_features:n_weights]))
            else:
                reg = 0.5 * self.lambda_reg * float(np.sum(params[:n_features] ** 2))
            _, _, beta_0, beta_1 = unpack(params)
            if shared_beta:
                return float(reg + (c0 + c1) / (beta_0 + 1.0))
            return float(reg + c0 / (beta_0 + 1.0) + c1 / (beta_1 + 1.0))

        def margin_1(params: FloatNDArray, t1: float) -> float:
            w_curr, b_curr, _, beta_1 = unpack(params)
            lhs = 0.5 * (beta_1 * t1 + float(w_curr @ sigma_1 @ w_curr) / t1)
            return float(float(w_curr @ mu_1 + b_curr) - lhs)

        def margin_0(params: FloatNDArray, t2: float) -> float:
            w_curr, b_curr, beta_0, _ = unpack(params)
            lhs = 0.5 * (beta_0 * t2 + float(w_curr @ sigma_0 @ w_curr) / t2)
            return float(float(-(w_curr @ mu_0 + b_curr)) - lhs)

        weight_bounds: list[tuple[float | None, float | None]] = [(0.0 if l1 else None, None)] * n_weights
        bounds = [*weight_bounds, (None, None), *[(1e-6, None)] * len(betas)]
        margins = self._margin_constraints(mu_1, mu_0, lambda p: unpack(p)[:2])

        for _ in range(25):
            if l1:
                weights = np.concatenate([np.maximum(w, 0), np.maximum(-np.asarray(w, dtype=np.float64), 0)])
            else:
                weights = np.asarray(w, dtype=np.float64)
            res = minimize(
                objective,
                np.concatenate([weights, [b], betas]),
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': margin_1, 'args': (t1,)},
                    {'type': 'ineq', 'fun': margin_0, 'args': (t2,)},
                    *margins,
                ],
            )
            w, b_value, beta_0, beta_1 = unpack(res.x)
            b = float(b_value)
            betas = [float(beta) for beta in res.x[n_weights + 1 :]]

            d1 = float(np.sqrt(w @ sigma_1 @ w))
            d0 = float(np.sqrt(w @ sigma_0 @ w))
            t1_new = max(d1 / np.sqrt(beta_1), 1e-6)
            t2_new = max(d0 / np.sqrt(beta_0), 1e-6)

            if abs(t1_new - t1) < 1e-4 and abs(t2_new - t2) < 1e-4:
                break
            t1, t2 = t1_new, t2_new

        # Each beta is kappa^2 of its class's worst-case accuracy (Eq. 21; a shared beta is the AM-GM
        # relaxation Lp-ProfMPM shares with Section 4.3), so kappa = sqrt(beta) recovers the worst-case
        # accuracies via _worst_case_accuracies.
        kappa_1 = float(np.sqrt(max(float(beta_1), 0.0)))
        kappa_0 = float(np.sqrt(max(float(beta_0), 0.0)))
        alpha_1, alpha_0 = self._worst_case_accuracies(kappa_1, kappa_0)
        return w, b, alpha_1, alpha_0, res

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        tp_benefit, tn_benefit, fp_cost, fn_cost = self._prepare_class_costs(loss_params)

        pos_mask = y == 1
        neg_mask = y == 0

        mu_1 = np.asarray(np.mean(X[pos_mask], axis=0, dtype=np.float64), dtype=np.float64)
        mu_0 = np.asarray(np.mean(X[neg_mask], axis=0, dtype=np.float64), dtype=np.float64)

        n_features = X.shape[1]
        ridge = np.eye(n_features, dtype=np.float64) * self.ridge_penalty
        sigma_1 = np.cov(X[pos_mask].astype(np.float64), rowvar=False).reshape(n_features, n_features) + ridge
        sigma_0 = np.cov(X[neg_mask].astype(np.float64), rowvar=False).reshape(n_features, n_features) + ridge

        pi_1 = float(np.mean(pos_mask))
        pi_0 = float(np.mean(neg_mask))

        # The penalty is measured against the average cost of a wrong decision, so that lambda_reg
        # does not depend on the units of the costs.
        scale = loss.model.decision_cost_scale(y, **loss_params)
        c1 = max(pi_1 * (tp_benefit + fn_cost) / scale, 1e-12)
        c0 = max(pi_0 * (tn_benefit + fp_cost) / scale, 1e-12)

        coef, intercept, alpha_1, alpha_0, result = self._fit_minimax(
            mu_1=mu_1, mu_0=mu_0, sigma_1=sigma_1, sigma_0=sigma_0, c1=c1, c0=c0
        )

        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)
        self.alpha_1_ = float(alpha_1)
        self.alpha_0_ = float(alpha_0)
        self.result_ = result

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
