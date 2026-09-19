from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.special import expit
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, MaxProfit
from .cost_sensitive import CostSensitiveClassifier, MetricStrategyFactory


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

        scalar_res = minimize_scalar(neg_profit, bounds=(1e-4, beta_max), method='bounded')
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

    def _solve_regularized_mpm(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        """Solve regularized Lp-ProfMPM (Section 4.2 / Algorithm 2 with shared beta)."""
        w_init, b_init, _, _, _ = self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)
        n_features = len(mu_1)
        t1, t2 = 1.0, 1.0
        w = w_init
        b = b_init
        beta = 1.0
        last_res: OptimizeResult | None = None

        for _ in range(25):
            if self.penalty == 'l2':

                def obj_l2(params: FloatNDArray) -> float:
                    w_curr = params[:n_features]
                    beta_curr = params[n_features + 1]
                    reg = 0.5 * self.lambda_reg * float(np.sum(w_curr**2))
                    return float(reg + (c0 + c1) / (beta_curr + 1.0))

                def con1_l2(params: FloatNDArray, t1: float = t1) -> float:
                    w_curr = params[:n_features]
                    b_curr = params[n_features]
                    beta_curr = params[n_features + 1]
                    lhs = 0.5 * (beta_curr * t1 + float(w_curr @ sigma_1 @ w_curr) / t1)
                    rhs = float(w_curr @ mu_1 + b_curr)
                    return float(rhs - lhs)

                def con2_l2(params: FloatNDArray, t2: float = t2) -> float:
                    w_curr = params[:n_features]
                    b_curr = params[n_features]
                    beta_curr = params[n_features + 1]
                    lhs = 0.5 * (beta_curr * t2 + float(w_curr @ sigma_0 @ w_curr) / t2)
                    rhs = float(-(w_curr @ mu_0 + b_curr))
                    return float(rhs - lhs)

                bounds = [(None, None)] * n_features + [(None, None)] + [(1e-6, None)]
                init = np.append(np.append(w, b), beta)
                res = minimize(
                    obj_l2,
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': con1_l2}, {'type': 'ineq', 'fun': con2_l2}],
                )
                w = res.x[:n_features]
                b = float(res.x[n_features])
                beta = float(res.x[n_features + 1])
                last_res = res
            else:

                def obj_l1(params: FloatNDArray) -> float:
                    u = params[:n_features]
                    v = params[n_features : 2 * n_features]
                    beta_curr = params[2 * n_features + 1]
                    reg = self.lambda_reg * float(np.sum(u + v))
                    return float(reg + (c0 + c1) / (beta_curr + 1.0))

                def con1_l1(params: FloatNDArray, t1: float = t1) -> float:
                    w_curr = params[:n_features] - params[n_features : 2 * n_features]
                    b_curr = params[2 * n_features]
                    beta_curr = params[2 * n_features + 1]
                    lhs = 0.5 * (beta_curr * t1 + float(w_curr @ sigma_1 @ w_curr) / t1)
                    rhs = float(w_curr @ mu_1 + b_curr)
                    return float(rhs - lhs)

                def con2_l1(params: FloatNDArray, t2: float = t2) -> float:
                    w_curr = params[:n_features] - params[n_features : 2 * n_features]
                    b_curr = params[2 * n_features]
                    beta_curr = params[2 * n_features + 1]
                    lhs = 0.5 * (beta_curr * t2 + float(w_curr @ sigma_0 @ w_curr) / t2)
                    rhs = float(-(w_curr @ mu_0 + b_curr))
                    return float(rhs - lhs)

                bounds = [(0, None)] * n_features + [(0, None)] * n_features + [(None, None)] + [(1e-6, None)]
                init = np.zeros(2 * n_features + 2)
                init[:n_features] = np.maximum(w, 0)
                init[n_features : 2 * n_features] = np.maximum(-np.asarray(w, dtype=np.float64), 0)
                init[2 * n_features] = b
                init[2 * n_features + 1] = beta
                res = minimize(
                    obj_l1,
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': con1_l1}, {'type': 'ineq', 'fun': con2_l1}],
                )
                w = res.x[:n_features] - res.x[n_features : 2 * n_features]
                b = float(res.x[2 * n_features])
                beta = float(res.x[2 * n_features + 1])
                last_res = res

            d1 = float(np.sqrt(w @ sigma_1 @ w))
            d0 = float(np.sqrt(w @ sigma_0 @ w))
            t1_new = max(d1 / np.sqrt(beta), 1e-6)
            t2_new = max(d0 / np.sqrt(beta), 1e-6)

            if abs(t1_new - t1) < 1e-4 and abs(t2_new - t2) < 1e-4:
                break
            t1, t2 = t1_new, t2_new

        # beta here is kappa^2(alpha) (the AM-GM relaxation shared with Section 4.3), so
        # kappa = sqrt(beta) and alpha_1 = alpha_0 = beta / (beta + 1) via _worst_case_accuracies.
        kappa = float(np.sqrt(max(beta, 0.0)))
        alpha_1, alpha_0 = self._worst_case_accuracies(kappa, kappa)
        return w, b, alpha_1, alpha_0, last_res or res

    def _solve_regularized_mempm(
        self,
        mu_1: FloatNDArray,
        mu_0: FloatNDArray,
        sigma_1: FloatNDArray,
        sigma_0: FloatNDArray,
        c1: float,
        c0: float,
    ) -> tuple[FloatNDArray, float, float, float, OptimizeResult]:
        """Solve regularized Lp-ProfMEMPM (Section 4.3 / Algorithm 2 with per-class beta)."""
        w_init, b_init, _, _, _ = self._solve_unregularized_mpm(mu_1, mu_0, sigma_1, sigma_0)
        n_features = len(mu_1)
        t1, t2 = 1.0, 1.0
        w = w_init
        b = b_init
        beta_0 = 1.0
        beta_1 = 1.0
        last_res: OptimizeResult | None = None

        for _ in range(25):
            if self.penalty == 'l2':

                def obj_l2(params: FloatNDArray) -> float:
                    w_curr = params[:n_features]
                    b0_curr = params[n_features + 1]
                    b1_curr = params[n_features + 2]
                    reg = 0.5 * self.lambda_reg * float(np.sum(w_curr**2))
                    return float(reg + c0 / (b0_curr + 1.0) + c1 / (b1_curr + 1.0))

                def con1_l2(params: FloatNDArray, t1: float = t1) -> float:
                    w_curr = params[:n_features]
                    b_curr = params[n_features]
                    b1_curr = params[n_features + 2]
                    lhs = 0.5 * (b1_curr * t1 + float(w_curr @ sigma_1 @ w_curr) / t1)
                    rhs = float(w_curr @ mu_1 + b_curr)
                    return float(rhs - lhs)

                def con2_l2(params: FloatNDArray, t2: float = t2) -> float:
                    w_curr = params[:n_features]
                    b_curr = params[n_features]
                    b0_curr = params[n_features + 1]
                    lhs = 0.5 * (b0_curr * t2 + float(w_curr @ sigma_0 @ w_curr) / t2)
                    rhs = float(-(w_curr @ mu_0 + b_curr))
                    return float(rhs - lhs)

                bounds = [(None, None)] * n_features + [(None, None)] + [(1e-6, None), (1e-6, None)]
                init = np.append(np.append(w, b), [beta_0, beta_1])
                res = minimize(
                    obj_l2,
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': con1_l2}, {'type': 'ineq', 'fun': con2_l2}],
                )
                w = res.x[:n_features]
                b = float(res.x[n_features])
                beta_0 = float(res.x[n_features + 1])
                beta_1 = float(res.x[n_features + 2])
                last_res = res
            else:

                def obj_l1(params: FloatNDArray) -> float:
                    u = params[:n_features]
                    v = params[n_features : 2 * n_features]
                    b0_curr = params[2 * n_features + 1]
                    b1_curr = params[2 * n_features + 2]
                    reg = self.lambda_reg * float(np.sum(u + v))
                    return float(reg + c0 / (b0_curr + 1.0) + c1 / (b1_curr + 1.0))

                def con1_l1(params: FloatNDArray, t1: float = t1) -> float:
                    w_curr = params[:n_features] - params[n_features : 2 * n_features]
                    b_curr = params[2 * n_features]
                    b1_curr = params[2 * n_features + 2]
                    lhs = 0.5 * (b1_curr * t1 + float(w_curr @ sigma_1 @ w_curr) / t1)
                    rhs = float(w_curr @ mu_1 + b_curr)
                    return float(rhs - lhs)

                def con2_l1(params: FloatNDArray, t2: float = t2) -> float:
                    w_curr = params[:n_features] - params[n_features : 2 * n_features]
                    b_curr = params[2 * n_features]
                    b0_curr = params[2 * n_features + 1]
                    lhs = 0.5 * (b0_curr * t2 + float(w_curr @ sigma_0 @ w_curr) / t2)
                    rhs = float(-(w_curr @ mu_0 + b_curr))
                    return float(rhs - lhs)

                bounds = (
                    [(0, None)] * n_features + [(0, None)] * n_features + [(None, None)] + [(1e-6, None), (1e-6, None)]
                )
                init = np.zeros(2 * n_features + 3)
                init[:n_features] = np.maximum(w, 0)
                init[n_features : 2 * n_features] = np.maximum(-np.asarray(w, dtype=np.float64), 0)
                init[2 * n_features] = b
                init[2 * n_features + 1] = beta_0
                init[2 * n_features + 2] = beta_1
                res = minimize(
                    obj_l1,
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': con1_l1}, {'type': 'ineq', 'fun': con2_l1}],
                )
                w = res.x[:n_features] - res.x[n_features : 2 * n_features]
                b = float(res.x[2 * n_features])
                beta_0 = float(res.x[2 * n_features + 1])
                beta_1 = float(res.x[2 * n_features + 2])
                last_res = res

            d1 = float(np.sqrt(w @ sigma_1 @ w))
            d0 = float(np.sqrt(w @ sigma_0 @ w))
            t1_new = max(d1 / np.sqrt(beta_1), 1e-6)
            t2_new = max(d0 / np.sqrt(beta_0), 1e-6)

            if abs(t1_new - t1) < 1e-4 and abs(t2_new - t2) < 1e-4:
                break
            t1, t2 = t1_new, t2_new

        # beta_0, beta_1 are kappa_0^2(alpha_0), kappa_1^2(alpha_1) (Eq. 21), so
        # kappa_i = sqrt(beta_i) recovers the per-class bounds via _worst_case_accuracies.
        kappa_1 = float(np.sqrt(max(beta_1, 0.0)))
        kappa_0 = float(np.sqrt(max(beta_0, 0.0)))
        alpha_1, alpha_0 = self._worst_case_accuracies(kappa_1, kappa_0)
        return w, b, alpha_1, alpha_0, last_res or res

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

        c1 = max(pi_1 * (tp_benefit + fn_cost), 1e-12)
        c0 = max(pi_0 * (tn_benefit + fp_cost), 1e-12)

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
