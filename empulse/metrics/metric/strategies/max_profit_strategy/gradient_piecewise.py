import numpy as np
import sympy
from scipy.special import expit

from ....._types import FloatNDArray
from ...common import _safe_lambdify
from .common import _convex_hull, extract_distribution_parameters
from .piecewise import BasePositiveDistribution, compute_piecewise_bounds


class MaxProfitLogitGradientPiecewise:
    """Picklable objective for Piecewise Stochastic MaxProfit optimized with logistic models."""

    def __init__(
        self,
        *,
        score_function: BasePositiveDistribution,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        alpha_0: float,
        alpha_growth: float,
        alpha_max: float,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        self.score_function = score_function
        self.features = features
        self.y_true = y_true.ravel().astype(np.int32)
        self.C = C
        self.l1_ratio = l1_ratio
        self.soft_threshold = soft_threshold
        self.fit_intercept = fit_intercept
        self.alpha_0 = alpha_0
        self.alpha_growth = alpha_growth
        self.alpha_max = alpha_max
        self._epoch = 0

        # --- 1. PRECOMPUTE DATASET CONSTANTS ---
        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = int(self.pos_mask.sum())
        self.n_neg = int(self.neg_mask.sum())
        self.X_pos = self.features[self.pos_mask]
        self.X_neg = self.features[self.neg_mask]

        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

        # --- 2. PRECOMPUTE DISTRIBUTION & BUSINESS CONSTANTS ---
        self.dist_params, self.kwargs = extract_distribution_parameters(
            parameters, self.score_function.distribution_args
        )
        self.fix_inf = not self.score_function.derivative.subs(self.kwargs).is_negative

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

        # --- 3. SYMBOLIC DERIVATIONS ---
        F_0, F_1 = sympy.symbols('F_0 F_1')  # noqa: N806
        self.da_dF0_eqs = []
        self.da_dF1_eqs = []
        self.da_dF0_fns = []
        self.da_dF1_fns = []

        for eq in self.score_function.coefficient_eqs:
            da_dF0 = sympy.diff(eq, F_0)  # noqa: N806
            da_dF1 = sympy.diff(eq, F_1)  # noqa: N806

            self.da_dF0_eqs.append(da_dF0)
            self.da_dF1_eqs.append(da_dF1)
            self.da_dF0_fns.append(_safe_lambdify(da_dF0))
            self.da_dF1_fns.append(_safe_lambdify(da_dF1))

        # --- 4. PRECOMPUTE LAMBDA ARGUMENT DISPATCHERS ---
        # We pre-calculate exactly which kwargs, F_0, F_1, pi_0, and pi_1 are needed
        # by each specific sympy function so we can bypass slow filtering in the train loop.
        self.a_reqs = []
        self.da0_reqs = []
        self.da1_reqs = []

        for k in range(len(self.score_function.coefficient_eqs)):
            # Helper to parse required arguments once
            def _get_reqs(eq):
                reqs = {str(s) for s in eq.free_symbols}
                static_kws = {key: val for key, val in self.kwargs.items() if key in reqs}
                return static_kws, 'pi_0' in reqs, 'pi_1' in reqs, 'F_0' in reqs, 'F_1' in reqs

            self.a_reqs.append(_get_reqs(self.score_function.coefficient_eqs[k]))
            self.da0_reqs.append(_get_reqs(self.da_dF0_eqs[k]))
            self.da1_reqs.append(_get_reqs(self.da_dF1_eqs[k]))

    def _current_alpha(self) -> float:
        """Compute annealed temperature for the current objective evaluation."""
        try:
            alpha = self.alpha_0 * (self.alpha_growth**self._epoch)
        except OverflowError:
            alpha = self.alpha_max
        return float(min(self.alpha_max, alpha))

    def reset(self):
        """Reset the objective evaluation."""
        self._epoch = 0

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated stochastic EMP objective and gradient for minimization."""
        start_coef = 1 if self.fit_intercept else 0

        w = np.asarray(weights, dtype=np.float64).copy()
        if self.soft_threshold:
            abs_w = np.abs(w[start_coef:])
            diff = abs_w - self.C
            w[start_coef:] = np.where(
                diff > 0,
                np.sign(w[start_coef:]) * diff,
                np.where(diff < 0, 0.0, w[start_coef:]),
            )

        alpha = self._current_alpha()
        self._epoch += 1

        # 1. Forward Pass: Dynamic Scores & Hull
        y_score = expit(self.features @ w)
        tprs, fprs = _convex_hull(self.y_true, y_score)

        # 2. Extract Piecewise Segments
        bounds, _, _, segment_tprs, segment_fprs = compute_piecewise_bounds(
            self.score_function.compute_bounds_fns,
            tprs,
            fprs,
            self.pi0,
            self.pi1,
            self.score_function.random_var_bounds,
            self.dist_params,
            fix_inf=self.fix_inf,
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound,
            **self.kwargs,
        )

        # --- THE FIX: Force cast SymPy bounds to native NumPy floats ---
        bounds = np.asarray(bounds, dtype=np.float64)

        # Convert segments to NumPy arrays for vectorization
        segment_tprs_arr = np.asarray(segment_tprs)  # Shape: (M,)
        segment_fprs_arr = np.asarray(segment_fprs)  # Shape: (M,)

        M = len(segment_tprs_arr)  # noqa: N806

        # Compute thresholds for all M segments
        # rates = segment_tprs_arr * self.pi0 + segment_fprs_arr * self.pi1
        # T_M = np.array([classification_threshold(self.y_true, y_score, float(r)) for r in rates])  # Shape: (M,)
        rates = np.clip(segment_tprs_arr * self.pi0 + segment_fprs_arr * self.pi1, 0.0, 1.0)
        T_M = np.quantile(y_score, 1.0 - rates)  # noqa: N806

        # 3. Precompute logistic derivatives
        s_pos = y_score[self.pos_mask]  # Shape: (N_pos,)
        s_neg = y_score[self.neg_mask]  # Shape: (N_neg,)
        sd_pos = s_pos * (1.0 - s_pos)
        sd_neg = s_neg * (1.0 - s_neg)

        # --- THE VECTORIZATION MAGIC ---
        # Broadcast scores (N, 1) against thresholds (1, M) to create a (N, M) matrix of sigmoids
        # sig_pos = expit(alpha * (s_pos[:, None] - T_M[None, :]))  # Shape: (N_pos, M)
        # sig_neg = expit(alpha * (s_neg[:, None] - T_M[None, :]))  # Shape: (N_neg, M)
        sig_pos = expit(alpha * np.subtract.outer(s_pos, T_M))
        sig_neg = expit(alpha * np.subtract.outer(s_neg, T_M))

        dsig_pos = sig_pos * (1.0 - sig_pos)
        dsig_neg = sig_neg * (1.0 - sig_neg)

        # Matrix multiplication computes the feature gradients for ALL M segments instantly!
        # Transpose (N_pos, M) to (M, N_pos) @ (N_pos, F_features) -> Result is (M, F_features)
        grad_F0_M = (alpha / self.n_pos) * ((dsig_pos * sd_pos[:, None]).T @ self.X_pos)  # noqa: N806
        grad_F1_M = (alpha / self.n_neg) * ((dsig_neg * sd_neg[:, None]).T @ self.X_neg)  # noqa: N806

        total_value = 0.0
        total_gradient = np.zeros_like(w)

        # 4. Loop over polynomial terms (k is usually just 0 and 1, so this loop is tiny)
        for k in range(len(self.score_function.coefficient_eqs)):
            # Fetch analytical integrals for all M segments
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # noqa: N806

            if not np.any(R_kM):
                continue

            # Evaluate sympy lambdas for all M segments simultaneously using array inputs.
            static_kw, n_p0, n_p1, n_F0, n_F1 = self.a_reqs[k]  # noqa: N806
            args_a = static_kw.copy()
            if n_p0:
                args_a['pi_0'] = self.pi0
            if n_p1:
                args_a['pi_1'] = self.pi1
            if n_F0:
                args_a['F_0'] = segment_tprs_arr
            if n_F1:
                args_a['F_1'] = segment_fprs_arr

            a_k_raw = self.score_function.coefficient_fns[k](**args_a)
            a_k_M = np.broadcast_to(np.asarray(a_k_raw, dtype=np.float64), (M,))  # noqa: N806

            # --- EVALUATE da_dF0 ---
            static_kw0, n_p0, n_p1, n_F0, n_F1 = self.da0_reqs[k]  # noqa: N806
            args_da0 = static_kw0.copy()
            if n_p0:
                args_da0['pi_0'] = self.pi0
            if n_p1:
                args_da0['pi_1'] = self.pi1
            if n_F0:
                args_da0['F_0'] = segment_tprs_arr
            if n_F1:
                args_da0['F_1'] = segment_fprs_arr

            da_dF0_raw = self.da_dF0_fns[k](**args_da0)  # noqa: N806
            da_dF0_M = np.broadcast_to(np.asarray(da_dF0_raw, dtype=np.float64), (M,))  # noqa: N806

            # --- EVALUATE da_dF1 ---
            static_kw1, n_p0, n_p1, n_F0, n_F1 = self.da1_reqs[k]  # noqa: N806
            args_da1 = static_kw1.copy()
            if n_p0:
                args_da1['pi_0'] = self.pi0
            if n_p1:
                args_da1['pi_1'] = self.pi1
            if n_F0:
                args_da1['F_0'] = segment_tprs_arr
            if n_F1:
                args_da1['F_1'] = segment_fprs_arr

            da_dF1_raw = self.da_dF1_fns[k](**args_da1)  # noqa: N806
            da_dF1_M = np.broadcast_to(np.asarray(da_dF1_raw, dtype=np.float64), (M,))  # noqa: N806

            # Accumulate value and gradient arrays
            total_value += np.sum(a_k_M * R_kM)

            # Multiply the scalars first (Shape: (M,))
            weight_F0 = R_kM * da_dF0_M  # noqa: N806
            weight_F1 = R_kM * da_dF1_M  # noqa: N806

            # Matrix multiplication (M,) @ (M, F) -> instantly reduces to (F,)
            total_gradient += (weight_F0 @ grad_F0_M) + (weight_F1 @ grad_F1_M)

        # Convert to minimization problem
        value = float(-total_value)
        gradient = -total_gradient

        # 5. Regularization
        coef = w[start_coef:]
        if self.l1_ratio == 0.0:
            gradient[start_coef:] += coef / self.C
            value += 0.5 * float(np.dot(coef, coef)) / self.C
        elif self.l1_ratio == 1.0:
            gradient[start_coef:] += np.sign(coef) / self.C
            value += float(np.sum(np.abs(coef))) / self.C
        else:
            gradient[start_coef:] += ((1.0 - self.l1_ratio) * coef + self.l1_ratio * np.sign(coef)) / self.C
            value += (
                (1.0 - self.l1_ratio) * 0.5 * float(np.dot(coef, coef)) + self.l1_ratio * float(np.sum(np.abs(coef)))
            ) / self.C

        return value, gradient


class MaxProfitBoostGradientPiecewise:
    """Prepared piecewise objective for MaxProfit stochastic gradient boosting."""

    def __init__(
        self,
        *,
        score_function: BasePositiveDistribution,
        y_true: FloatNDArray,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        self.score_function = score_function
        self.y_true = np.asarray(y_true).reshape(-1).astype(np.int32)
        self.parameters = parameters

        # --- 1. PRECOMPUTE DATASET CONSTANTS ---
        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = max(int(np.sum(self.pos_mask)), 1)
        self.n_neg = max(int(np.sum(self.neg_mask)), 1)

        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

        # --- 2. PRECOMPUTE DISTRIBUTION & BUSINESS CONSTANTS ---
        self.dist_params, self.kwargs = extract_distribution_parameters(
            parameters, self.score_function.distribution_args
        )
        self.fix_inf = not self.score_function.derivative.subs(self.kwargs).is_negative

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

        # --- 3. SYMBOLIC DERIVATIONS ---
        F_0, F_1 = sympy.symbols('F_0 F_1')  # noqa: N806
        self.da_dF0_eqs = []
        self.da_dF1_eqs = []
        self.da_dF0_fns = []
        self.da_dF1_fns = []

        for eq in self.score_function.coefficient_eqs:
            da_dF0 = sympy.diff(eq, F_0)  # noqa: N806
            da_dF1 = sympy.diff(eq, F_1)  # noqa: N806
            self.da_dF0_eqs.append(da_dF0)
            self.da_dF1_eqs.append(da_dF1)
            self.da_dF0_fns.append(_safe_lambdify(da_dF0))
            self.da_dF1_fns.append(_safe_lambdify(da_dF1))

        # --- 4. PRECOMPUTE LAMBDA ARGUMENT DISPATCHERS ---
        self.da0_reqs = []
        self.da1_reqs = []
        for k in range(len(self.score_function.coefficient_eqs)):

            def _get_reqs(eq):
                reqs = {str(s) for s in eq.free_symbols}
                static_kws = {key: val for key, val in self.kwargs.items() if key in reqs}
                return static_kws, 'pi_0' in reqs, 'pi_1' in reqs, 'F_0' in reqs, 'F_1' in reqs

            self.da0_reqs.append(_get_reqs(self.da_dF0_eqs[k]))
            self.da1_reqs.append(_get_reqs(self.da_dF1_eqs[k]))

    def __call__(self, y_score: FloatNDArray, alpha: float) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the stochastic objective."""
        y_score_arr = expit(np.asarray(y_score, dtype=np.float64).reshape(-1))

        # 1. Forward Pass: Dynamic Hull & Bounds
        tprs, fprs = _convex_hull(self.y_true, y_score_arr)
        bounds, _, _, segment_tprs, segment_fprs = compute_piecewise_bounds(
            self.score_function.compute_bounds_fns,
            tprs,
            fprs,
            self.pi0,
            self.pi1,
            self.score_function.random_var_bounds,
            self.dist_params,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            fix_inf=self.fix_inf,
            **self.kwargs,
        )

        bounds = np.asarray(bounds, dtype=np.float64)
        segment_tprs_arr = np.asarray(segment_tprs, dtype=np.float64)
        segment_fprs_arr = np.asarray(segment_fprs, dtype=np.float64)
        M = len(segment_tprs_arr)  # noqa: N806

        # 2. Vectorized Thresholds
        rates = np.clip(segment_tprs_arr * self.pi0 + segment_fprs_arr * self.pi1, 0.0, 1.0)
        T_M = np.quantile(y_score_arr, 1.0 - rates)  # noqa: N806

        # 3. Precompute logistic derivatives for instances
        s_pos = y_score_arr[self.pos_mask]
        s_neg = y_score_arr[self.neg_mask]

        sig_pos = expit(alpha * np.subtract.outer(s_pos, T_M))
        sig_neg = expit(alpha * np.subtract.outer(s_neg, T_M))

        sig_prime_pos = alpha * sig_pos * (1.0 - sig_pos)
        sig_prime_neg = alpha * sig_neg * (1.0 - sig_neg)

        sig_sec_pos = alpha**2 * sig_pos * (1.0 - sig_pos) * (1.0 - 2.0 * sig_pos)
        sig_sec_neg = alpha**2 * sig_neg * (1.0 - sig_neg) * (1.0 - 2.0 * sig_neg)

        # 4. Leibniz Summation
        weight_F0_M = np.zeros(M)  # noqa: N806
        weight_F1_M = np.zeros(M)  # noqa: N806

        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # noqa: N806

            if not np.any(R_kM):
                continue

            # Fast Dispatch for da_dF0
            static_kw0, n_p0, n_p1, n_F0, n_F1 = self.da0_reqs[k]  # noqa: N806
            args_da0 = static_kw0.copy()
            if n_p0:
                args_da0['pi_0'] = self.pi0
            if n_p1:
                args_da0['pi_1'] = self.pi1
            if n_F0:
                args_da0['F_0'] = segment_tprs_arr
            if n_F1:
                args_da0['F_1'] = segment_fprs_arr

            da_dF0_raw = self.da_dF0_fns[k](**args_da0)  # noqa: N806
            da_dF0_M = np.broadcast_to(np.asarray(da_dF0_raw, dtype=np.float64), (M,))  # noqa: N806

            # Fast Dispatch for da_dF1
            static_kw1, n_p0, n_p1, n_F0, n_F1 = self.da1_reqs[k]  # noqa: N806
            args_da1 = static_kw1.copy()
            if n_p0:
                args_da1['pi_0'] = self.pi0
            if n_p1:
                args_da1['pi_1'] = self.pi1
            if n_F0:
                args_da1['F_0'] = segment_tprs_arr
            if n_F1:
                args_da1['F_1'] = segment_fprs_arr

            da_dF1_raw = self.da_dF1_fns[k](**args_da1)  # noqa: N806
            da_dF1_M = np.broadcast_to(np.asarray(da_dF1_raw, dtype=np.float64), (M,))  # noqa: N806

            weight_F0_M += R_kM * da_dF0_M  # noqa: N806
            weight_F1_M += R_kM * da_dF1_M  # noqa: N806

        # 5. Convert to minimization constants per segment
        c_pos_M = -weight_F0_M / self.n_pos  # noqa: N806
        c_neg_M = -weight_F1_M / self.n_neg  # noqa: N806

        # 6. Matrix Multiply to compute global gradients/hessians per instance
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

        # --- THE INITIALIZATION FIX ---
        # At epoch 0, all scores are identical. sigma = 0.5, so the exact hessian is 0.
        # XGBoost refuses to split nodes if sum(hessian) < min_child_weight.
        # We fall back to a strict numerical floor (or gradient magnitude) ONLY when it vanishes.
        hessian = np.abs(hessian)

        # Use a floor of 0.1 to guarantee it passes the default min_child_weight of 1.0,
        # or use the absolute gradient if it's larger.
        hessian_floor = np.maximum(np.abs(gradient), 0.1)
        hessian = np.where(hessian < 1e-7, hessian_floor, hessian)

        return gradient, hessian
