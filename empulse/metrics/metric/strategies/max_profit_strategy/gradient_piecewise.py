from collections.abc import Generator
from typing import Any

import numpy as np
import sympy
from scipy.special import expit

from ....._types import Float64Array, FloatNDArray
from ...common import _safe_lambdify
from .common import (
    _BaseMaxProfitLogitObjective,
    _convex_hull,
    _smooth_step_derivatives,
    extract_distribution_parameters,
)
from .envelope import Partition, PolynomialEnvelope
from .piecewise import BasePositiveDistribution, compute_piecewise_bounds

# Type alias for the hull state tuple cached between gradient steps.
_HullCache = tuple[FloatNDArray, FloatNDArray, FloatNDArray, int]


def _parse_lambdify_reqs(eq: sympy.Expr, kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool, bool, bool, bool]:
    """Split kwargs into the static subset an expression needs and its dynamic requirements.

    Returns the subset of *kwargs* the expression's free symbols actually need, plus which
    dynamic ROC-axis inputs (``pi_0``, ``pi_1``, ``F_0``, ``F_1``) it requires.
    """
    reqs = {str(s) for s in eq.free_symbols}
    static_kws = {key: val for key, val in kwargs.items() if key in reqs}
    return static_kws, 'pi_0' in reqs, 'pi_1' in reqs, 'F_0' in reqs, 'F_1' in reqs


def _build_lambdify_args(
    reqs: tuple[dict[str, Any], bool, bool, bool, bool],
    pi0: float,
    pi1: float,
    seg_tprs: FloatNDArray,
    seg_fprs: FloatNDArray,
) -> dict[str, Any]:
    """Assemble the kwargs a lambdified expression needs, given its requirements from `_parse_lambdify_reqs`."""
    static_kw, n_p0, n_p1, n_F0, n_F1 = reqs  # ruff: ignore[non-lowercase-variable-in-function]
    args = static_kw.copy()
    if n_p0:
        args['pi_0'] = pi0
    if n_p1:
        args['pi_1'] = pi1
    if n_F0:
        args['F_0'] = seg_tprs
    if n_F1:
        args['F_1'] = seg_fprs
    return args


class _PiecewiseDerivativeState:
    """Shared setup for MaxProfit's single-stochastic-variable piecewise objectives.

    Precomputes the distribution parameters, the exact float bounds of the stochastic
    variable's support, and the profit function's F_0/F_1 derivatives (plus the static/dynamic
    argument requirements each one needs) - all data-independent, so it only needs computing
    once from *score_function* and *parameters*, shared unchanged by both the logit and
    boosting piecewise objectives.
    """

    # Set by the concrete objectives before `_init_piecewise_state` runs.
    pi0: float
    pi1: float

    def _init_piecewise_state(
        self, score_function: BasePositiveDistribution, parameters: dict[str, FloatNDArray | float]
    ) -> None:
        self.score_function = score_function

        self.dist_params, self.kwargs = extract_distribution_parameters(
            parameters, self.score_function.distribution_args
        )

        # Precalculate the exact float bounds of the distribution to avoid sympy overhead in loop
        lower_b = self.score_function.random_var_bounds[0]
        if isinstance(lower_b, sympy.Expr):
            lower_b = lower_b.subs(self.dist_params)
            self.lower_bound = -np.inf if lower_b == -sympy.oo else float(lower_b)
        else:
            self.lower_bound = float(lower_b)

        upper_b = self.score_function.random_var_bounds[1]
        if isinstance(upper_b, sympy.Expr):
            upper_b = upper_b.subs(self.dist_params)
            self.upper_bound = np.inf if upper_b == sympy.oo else float(upper_b)
        else:
            self.upper_bound = float(upper_b)

        F_0, F_1 = sympy.symbols('F_0 F_1')  # ruff: ignore[non-lowercase-variable-in-function]
        self.da_dF0_eqs = []
        self.da_dF1_eqs = []
        self.da_dF0_fns = []
        self.da_dF1_fns = []

        for eq in self.score_function.coefficient_eqs:
            da_dF0 = sympy.diff(eq, F_0)  # ruff: ignore[non-lowercase-variable-in-function]
            da_dF1 = sympy.diff(eq, F_1)  # ruff: ignore[non-lowercase-variable-in-function]
            self.da_dF0_eqs.append(da_dF0)
            self.da_dF1_eqs.append(da_dF1)
            self.da_dF0_fns.append(_safe_lambdify(da_dF0))
            self.da_dF1_fns.append(_safe_lambdify(da_dF1))

        self.da0_reqs = []
        self.da1_reqs = []
        for k in range(len(self.score_function.coefficient_eqs)):
            self.da0_reqs.append(_parse_lambdify_reqs(self.da_dF0_eqs[k], self.kwargs))
            self.da1_reqs.append(_parse_lambdify_reqs(self.da_dF1_eqs[k], self.kwargs))

        self.a_reqs = [_parse_lambdify_reqs(eq, self.kwargs) for eq in self.score_function.coefficient_eqs]

    def _partition(self, tprs: FloatNDArray, fprs: FloatNDArray) -> Partition:
        """
        Split the support into the regions where each hull vertex maximises profit.

        The coefficients are evaluated at every hull vertex rather than at a precomputed set of
        segments, because which vertex owns which region is what the partition decides.
        """
        coefficients = np.stack(
            [
                np.broadcast_to(
                    np.asarray(
                        self.score_function.coefficient_fns[k](
                            **_build_lambdify_args(self.a_reqs[k], self.pi0, self.pi1, tprs, fprs)
                        ),
                        dtype=np.float64,
                    ),
                    tprs.shape,
                )
                for k in range(len(self.a_reqs))
            ],
            axis=-1,
        )
        return compute_piecewise_bounds(
            PolynomialEnvelope(coefficients),
            tprs,
            fprs,
            self.score_function.random_var_bounds,
            self.dist_params,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )


class MaxProfitLogitGradientPiecewise(_BaseMaxProfitLogitObjective, _PiecewiseDerivativeState):
    """
    Picklable objective for Piecewise Stochastic MaxProfit optimized with logistic models.

    Methods
    -------
    * ``__call__(weights)`` – returns ``(value, gradient)``; delegates to :meth:`logit_loss_gradient`.
    * ``logit_loss_gradient(weights)`` – returns ``(value, gradient)`` using a fresh hull.
    * ``logit_loss(weights)`` – returns only the scalar loss (fresh hull).
      Cheap when only the objective value is needed (e.g. final fitness evaluation).
    * ``logit_gradient(weights)`` – returns only the gradient vector (fresh hull).
      Saves the value-accumulation work when the scalar is not needed.
    * ``logit_gradient_steps()`` – a *generator* that yields gradients while reusing
      a cached convex hull across multiple calls.  Ideal for the Lamarckian memetic pattern
      where a few gradient steps are applied before the true fitness is evaluated.
    """

    def __init__(
        self,
        *,
        score_function: BasePositiveDistribution,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        alpha: float,
        objective_scale: float = 1.0,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        super().__init__(
            features=features,
            y_true=y_true,
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            alpha=alpha,
            objective_scale=objective_scale,
        )
        self._init_piecewise_state(score_function, parameters)

    def _compute_hull_state(self, y_score: FloatNDArray) -> _HullCache:
        """Build the ROC convex hull and derive piecewise segment arrays.

        Returns
        -------
        bounds : ndarray, shape (M+1,)
        seg_tprs : ndarray, shape (M,)
        seg_fprs : ndarray, shape (M,)
        M : int
            Number of piecewise segments.
        """
        tprs, fprs = _convex_hull(self.y_true, y_score)
        partition = self._partition(tprs, fprs)
        bounds = np.asarray(partition.bounds, dtype=np.float64)
        seg_tprs = np.asarray(partition.tprs, dtype=np.float64)
        seg_fprs = np.asarray(partition.fprs, dtype=np.float64)
        M = len(seg_tprs)  # ruff: ignore[non-lowercase-variable-in-function]
        return bounds, seg_tprs, seg_fprs, M

    def _compute_thresholds(
        self,
        y_score: FloatNDArray,
        seg_tprs: FloatNDArray,
        seg_fprs: FloatNDArray,
    ) -> FloatNDArray:
        """Compute per-segment classification thresholds from current scores."""
        rates = np.clip(seg_tprs * self.pi0 + seg_fprs * self.pi1, 0.0, 1.0)
        return np.quantile(y_score, 1.0 - rates)  # type: ignore[return-value]

    def _accumulate_value(
        self,
        bounds: FloatNDArray,
        seg_tprs: FloatNDArray,
        seg_fprs: FloatNDArray,
        M: int,  # ruff: ignore[invalid-argument-name]
    ) -> float:
        """Sum the piecewise EMP objective value across all segments and polynomial terms."""
        total_value = 0.0
        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # ruff: ignore[non-lowercase-variable-in-function]
            if not np.any(R_kM):
                continue

            args_a = _build_lambdify_args(self.a_reqs[k], self.pi0, self.pi1, seg_tprs, seg_fprs)
            a_k_raw = self.score_function.coefficient_fns[k](**args_a)
            a_k_M = np.broadcast_to(np.asarray(a_k_raw, dtype=np.float64), (M,))  # ruff: ignore[non-lowercase-variable-in-function]
            total_value += float(np.sum(a_k_M * R_kM))
        return total_value

    def _accumulate_gradient(
        self,
        w: FloatNDArray,
        y_score: FloatNDArray,
        T_M: FloatNDArray,  # ruff: ignore[invalid-argument-name]
        bounds: FloatNDArray,
        seg_tprs: FloatNDArray,
        seg_fprs: FloatNDArray,
        M: int,  # ruff: ignore[invalid-argument-name]
        alpha: float,
    ) -> Float64Array:
        """Compute the raw (un-negated, un-regularized) gradient vector."""
        s_pos = y_score[self.pos_mask]
        s_neg = y_score[self.neg_mask]
        sd_pos = s_pos * (1.0 - s_pos)
        sd_neg = s_neg * (1.0 - s_neg)

        _, dsig_pos = _smooth_step_derivatives(np.subtract.outer(s_pos, T_M), alpha, order=1)
        _, dsig_neg = _smooth_step_derivatives(np.subtract.outer(s_neg, T_M), alpha, order=1)

        # (M, F) feature-gradient matrices for the two ROC axes
        grad_F0_M = (alpha / self.n_pos) * ((dsig_pos * sd_pos[:, None]).T @ self.X_pos)  # ruff: ignore[non-lowercase-variable-in-function]
        grad_F1_M = (alpha / self.n_neg) * ((dsig_neg * sd_neg[:, None]).T @ self.X_neg)  # ruff: ignore[non-lowercase-variable-in-function]

        total_gradient: Float64Array = np.zeros(w.shape, dtype=np.float64)
        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # ruff: ignore[non-lowercase-variable-in-function]
            if not np.any(R_kM):
                continue

            # da/dF_0
            args_da0 = _build_lambdify_args(self.da0_reqs[k], self.pi0, self.pi1, seg_tprs, seg_fprs)
            da_dF0_M = np.broadcast_to(np.asarray(self.da_dF0_fns[k](**args_da0), dtype=np.float64), (M,))  # ruff: ignore[non-lowercase-variable-in-function]

            # da/dF_1
            args_da1 = _build_lambdify_args(self.da1_reqs[k], self.pi0, self.pi1, seg_tprs, seg_fprs)
            da_dF1_M = np.broadcast_to(np.asarray(self.da_dF1_fns[k](**args_da1), dtype=np.float64), (M,))  # ruff: ignore[non-lowercase-variable-in-function]

            weight_F0 = R_kM * da_dF0_M  # ruff: ignore[non-lowercase-variable-in-function]
            weight_F1 = R_kM * da_dF1_M  # ruff: ignore[non-lowercase-variable-in-function]
            total_gradient += (weight_F0 @ grad_F0_M) + (weight_F1 @ grad_F1_M)

        return total_gradient

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Compute the negated EMP objective value (no gradient).

        Always uses a freshly computed convex hull.  Intended for fitness evaluation
        rather than gradient-based updates.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Negated EMP loss (suitable for minimization).
        """
        w = np.asarray(weights, dtype=np.float64)
        y_score = self._compute_y_score(w)
        bounds, seg_tprs, seg_fprs, m = self._compute_hull_state(y_score)

        value = float(-self._accumulate_value(bounds, seg_tprs, seg_fprs, m))
        start_coef = self._start_coef
        value += self._regularization_value(w[start_coef:])
        return value

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Compute the gradient of the negated EMP objective (no scalar value).

        Always uses a freshly computed convex hull.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient vector matched in shape to *weights*.
        """
        w = np.asarray(weights, dtype=np.float64)
        alpha = self.alpha

        y_score = self._compute_y_score(w)
        bounds, seg_tprs, seg_fprs, m = self._compute_hull_state(y_score)
        T_M = self._compute_thresholds(y_score, seg_tprs, seg_fprs)  # ruff: ignore[non-lowercase-variable-in-function]

        grad = -self._accumulate_gradient(w, y_score, T_M, bounds, seg_tprs, seg_fprs, m, alpha)
        start_coef = self._start_coef
        grad[start_coef:] += self._regularization_gradient(w[start_coef:])
        return grad

    def _logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients while reusing a cached convex hull.

        The convex hull (and derived piecewise segment arrays) are computed once
        from *initial_weights* on the first call, then reused for subsequent
        gradient steps.  This avoids the hull-reconstruction cost inside tight
        local-search loops (e.g. a few Adam steps applied Lamarckian-style before
        a fitness evaluation) where the hull is unlikely to change substantially.

        When computing the actual loss (``logit_loss()``) always use a fresh hull.

        Send in a ``weights`` vector to reuse the cached hull (only scores and thresholds are
        recomputed), or a ``(weights, refresh)`` tuple with ``refresh=True`` to rebuild the hull
        from the new weights first.

        Yields
        ------
        gradient : ndarray
            Negated, regularized gradient at the weights last sent in.

        Examples
        --------
        Driving the generator by hand (``objective`` is a built objective,
        ``theta`` a coefficient vector)::

            gen = objective.logit_gradient_steps()
            grad = gen.send(theta)  # first time hull is built
            grad = gen.send(theta)  # hull reused
            grad = gen.send((theta, True))  # hull refreshed
            gen.close()
        """
        weights: FloatNDArray
        cached: _HullCache | None = None  # (bounds, seg_tprs, seg_fprs, M)

        # Priming yield: its value is discarded by the caller's next(generator) advance below,
        # so it is never actually observed as a FloatNDArray - mypy doesn't model that.
        sent = yield  # type: ignore[misc]

        while True:
            # Sending None is an alternative to close() for terminating the generator - matches
            # the sibling _logit_gradient_steps() implementations (CostLogitObjective,
            # MaxProfitLogitGradientDeterministic, LogitObjective's own default).
            if sent is None:
                return
            if isinstance(sent, tuple):
                weights, refresh = sent
                if refresh:
                    cached = None
            else:
                weights = sent

            w = np.asarray(weights, dtype=np.float64)
            alpha = self.alpha

            y_score = self._compute_y_score(w)

            if cached is None:
                cached = self._compute_hull_state(y_score)

            bounds, seg_tprs, seg_fprs, m = cached
            thresholds = self._compute_thresholds(y_score, seg_tprs, seg_fprs)

            grad = -self._accumulate_gradient(w, y_score, thresholds, bounds, seg_tprs, seg_fprs, m, alpha)

            start_coef = self._start_coef
            grad[start_coef:] += self._regularization_gradient(w[start_coef:])

            sent = yield grad

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated stochastic EMP objective and its gradient for minimization.

        Computes both value and gradient from a freshly built convex hull.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        loss : float
            Negated EMP value plus regularization.
        gradient : ndarray
            Gradient vector matched in shape to *weights*.
        """
        w = np.asarray(weights, dtype=np.float64)
        alpha = self.alpha

        y_score = self._compute_y_score(w)
        bounds, seg_tprs, seg_fprs, m = self._compute_hull_state(y_score)
        T_M = self._compute_thresholds(y_score, seg_tprs, seg_fprs)  # ruff: ignore[non-lowercase-variable-in-function]

        total_value = self._accumulate_value(bounds, seg_tprs, seg_fprs, m)
        total_gradient = self._accumulate_gradient(w, y_score, T_M, bounds, seg_tprs, seg_fprs, m, alpha)

        value = float(-total_value)
        gradient = -total_gradient

        start_coef = self._start_coef
        coef = w[start_coef:]
        value += self._regularization_value(coef)
        gradient[start_coef:] += self._regularization_gradient(coef)

        return value, gradient


class MaxProfitBoostGradientPiecewise(_PiecewiseDerivativeState):
    """Prepared piecewise objective for MaxProfit stochastic gradient boosting."""

    def __init__(
        self,
        *,
        score_function: BasePositiveDistribution,
        y_true: FloatNDArray,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        self.y_true = np.asarray(y_true).reshape(-1).astype(np.int32)
        self.parameters = parameters

        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = max(int(np.sum(self.pos_mask)), 1)
        self.n_neg = max(int(np.sum(self.neg_mask)), 1)

        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

        self._init_piecewise_state(score_function, parameters)

    def __call__(self, y_score: FloatNDArray, alpha: float) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the stochastic objective."""
        y_score_arr = expit(np.asarray(y_score, dtype=np.float64).reshape(-1))

        tprs, fprs = _convex_hull(self.y_true, y_score_arr)
        partition = self._partition(tprs, fprs)

        bounds = np.asarray(partition.bounds, dtype=np.float64)
        segment_tprs_arr = np.asarray(partition.tprs, dtype=np.float64)
        segment_fprs_arr = np.asarray(partition.fprs, dtype=np.float64)
        M = len(segment_tprs_arr)  # ruff: ignore[non-lowercase-variable-in-function]

        # Vectorized Thresholds
        rates = np.clip(segment_tprs_arr * self.pi0 + segment_fprs_arr * self.pi1, 0.0, 1.0)
        T_M = np.quantile(y_score_arr, 1.0 - rates)  # ruff: ignore[non-lowercase-variable-in-function]

        # Precompute logistic derivatives for instances
        s_pos = y_score_arr[self.pos_mask]
        s_neg = y_score_arr[self.neg_mask]

        _, factor1_pos, factor2_pos = _smooth_step_derivatives(np.subtract.outer(s_pos, T_M), alpha)
        _, factor1_neg, factor2_neg = _smooth_step_derivatives(np.subtract.outer(s_neg, T_M), alpha)

        sig_prime_pos = alpha * factor1_pos
        sig_prime_neg = alpha * factor1_neg

        sig_sec_pos = alpha**2 * factor2_pos
        sig_sec_neg = alpha**2 * factor2_neg

        weight_F0_M = np.zeros(M)  # ruff: ignore[non-lowercase-variable-in-function]
        weight_F1_M = np.zeros(M)  # ruff: ignore[non-lowercase-variable-in-function]

        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # ruff: ignore[non-lowercase-variable-in-function]

            if not np.any(R_kM):
                continue

            args_da0 = _build_lambdify_args(self.da0_reqs[k], self.pi0, self.pi1, segment_tprs_arr, segment_fprs_arr)
            da_dF0_raw = self.da_dF0_fns[k](**args_da0)  # ruff: ignore[non-lowercase-variable-in-function]
            da_dF0_M = np.broadcast_to(np.asarray(da_dF0_raw, dtype=np.float64), (M,))  # ruff: ignore[non-lowercase-variable-in-function]

            args_da1 = _build_lambdify_args(self.da1_reqs[k], self.pi0, self.pi1, segment_tprs_arr, segment_fprs_arr)
            da_dF1_raw = self.da_dF1_fns[k](**args_da1)  # ruff: ignore[non-lowercase-variable-in-function]
            da_dF1_M = np.broadcast_to(np.asarray(da_dF1_raw, dtype=np.float64), (M,))  # ruff: ignore[non-lowercase-variable-in-function]

            weight_F0_M += R_kM * da_dF0_M  # ruff: ignore[non-lowercase-variable-in-function]
            weight_F1_M += R_kM * da_dF1_M  # ruff: ignore[non-lowercase-variable-in-function]

        # Convert to minimization constants per segment
        c_pos_M = -weight_F0_M / self.n_pos  # ruff: ignore[non-lowercase-variable-in-function]
        c_neg_M = -weight_F1_M / self.n_neg  # ruff: ignore[non-lowercase-variable-in-function]

        # Matrix Multiply to compute global gradients/hessians per instance
        grad_pos = sig_prime_pos @ c_pos_M
        grad_neg = sig_prime_neg @ c_neg_M

        hess_pos = np.abs(sig_sec_pos @ c_pos_M)
        hess_neg = np.abs(sig_sec_neg @ c_neg_M)

        gradient = np.zeros_like(y_score_arr)
        gradient[self.pos_mask] = grad_pos
        gradient[self.neg_mask] = grad_neg

        hessian = np.zeros_like(y_score_arr)
        hessian[self.pos_mask] = hess_pos
        hessian[self.neg_mask] = hess_neg

        # At epoch 0, all scores are identical. sigma = 0.5, so the exact hessian is 0.
        # XGBoost refuses to split nodes if sum(hessian) < min_child_weight.
        # We fall back to a strict numerical floor (or gradient magnitude) ONLY when it vanishes.
        hessian = np.abs(hessian)

        # Use a floor of 0.1 to guarantee it passes the default min_child_weight of 1.0,
        # or use the absolute gradient if it's larger.
        hessian_floor = np.maximum(np.abs(gradient), 0.1)
        hessian = np.where(hessian < 1e-7, hessian_floor, hessian)

        return gradient, hessian
