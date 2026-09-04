import copy
from collections.abc import Generator
from typing import Any

import numpy as np
import sympy
from scipy.special import expit

from ....._types import Float64Array, FloatNDArray
from ...common import _safe_lambdify
from ..metric_strategy import LogitObjective
from .common import _convex_hull, extract_distribution_parameters
from .piecewise import BasePositiveDistribution, compute_piecewise_bounds

# Type alias for the hull state tuple cached between gradient steps.
_HullCache = tuple[FloatNDArray, FloatNDArray, FloatNDArray, int]


class MaxProfitLogitGradientPiecewise(LogitObjective):
    """
    Picklable objective for Piecewise Stochastic MaxProfit optimized with logistic models.

    Methods
    -------
    * ``__call__(weights)`` – returns ``(value, gradient)``; delegates to :meth:`logit_loss_gradient`.
    * ``logit_loss_gradient(weights)`` – returns ``(value, gradient)`` using a fresh hull.
    * ``logit_loss(weights)`` – returns only the scalar loss (fresh hull, no epoch increment).
      Cheap when only the objective value is needed (e.g. final fitness evaluation).
    * ``logit_gradient(weights)`` – returns only the gradient vector (fresh hull, increments epoch).
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
        self._alpha_override: float | None = None

        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = int(self.pos_mask.sum())
        self.n_neg = int(self.neg_mask.sum())
        self.X_pos: Float64Array = np.asarray(self.features[self.pos_mask], dtype=np.float64)
        self.X_neg: Float64Array = np.asarray(self.features[self.neg_mask], dtype=np.float64)

        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

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

        self.a_reqs = []
        self.da0_reqs = []
        self.da1_reqs = []

        for k in range(len(self.score_function.coefficient_eqs)):
            # Helper to parse required arguments once
            def _get_reqs(eq: sympy.Expr) -> tuple[dict[str, Any], bool, bool, bool, bool]:
                reqs = {str(s) for s in eq.free_symbols}
                static_kws = {key: val for key, val in self.kwargs.items() if key in reqs}
                return static_kws, 'pi_0' in reqs, 'pi_1' in reqs, 'F_0' in reqs, 'F_1' in reqs

            self.a_reqs.append(_get_reqs(self.score_function.coefficient_eqs[k]))
            self.da0_reqs.append(_get_reqs(self.da_dF0_eqs[k]))
            self.da1_reqs.append(_get_reqs(self.da_dF1_eqs[k]))

    def _apply_soft_threshold(self, weights: FloatNDArray) -> FloatNDArray:
        """Return a copy of *weights* with soft-thresholding applied (if enabled)."""
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
        return w

    def _compute_y_score(self, w: FloatNDArray) -> FloatNDArray:
        """Compute logistic scores for every sample."""
        return expit(self.features @ w)  # type: ignore[return-value]

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
        bounds = np.asarray(bounds, dtype=np.float64)
        seg_tprs = np.asarray(segment_tprs, dtype=np.float64)
        seg_fprs = np.asarray(segment_fprs, dtype=np.float64)
        M = len(seg_tprs)  # noqa: N806
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
        M: int,  # noqa: N803
    ) -> float:
        """Sum the piecewise EMP objective value across all segments and polynomial terms."""
        total_value = 0.0
        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # noqa: N806
            if not np.any(R_kM):
                continue

            static_kw, n_p0, n_p1, n_F0, n_F1 = self.a_reqs[k]  # noqa: N806
            args_a = static_kw.copy()
            if n_p0:
                args_a['pi_0'] = self.pi0
            if n_p1:
                args_a['pi_1'] = self.pi1
            if n_F0:
                args_a['F_0'] = seg_tprs
            if n_F1:
                args_a['F_1'] = seg_fprs

            a_k_raw = self.score_function.coefficient_fns[k](**args_a)
            a_k_M = np.broadcast_to(np.asarray(a_k_raw, dtype=np.float64), (M,))  # noqa: N806
            total_value += float(np.sum(a_k_M * R_kM))
        return total_value

    def _accumulate_gradient(
        self,
        w: FloatNDArray,
        y_score: FloatNDArray,
        T_M: FloatNDArray,  # noqa: N803
        bounds: FloatNDArray,
        seg_tprs: FloatNDArray,
        seg_fprs: FloatNDArray,
        M: int,  # noqa: N803
        alpha: float,
    ) -> Float64Array:
        """Compute the raw (un-negated, un-regularized) gradient vector."""
        s_pos = y_score[self.pos_mask]
        s_neg = y_score[self.neg_mask]
        sd_pos = s_pos * (1.0 - s_pos)
        sd_neg = s_neg * (1.0 - s_neg)

        sig_pos = expit(alpha * np.subtract.outer(s_pos, T_M))
        sig_neg = expit(alpha * np.subtract.outer(s_neg, T_M))
        dsig_pos = sig_pos * (1.0 - sig_pos)
        dsig_neg = sig_neg * (1.0 - sig_neg)

        # (M, F) feature-gradient matrices for the two ROC axes
        grad_F0_M = (alpha / self.n_pos) * ((dsig_pos * sd_pos[:, None]).T @ self.X_pos)  # noqa: N806
        grad_F1_M = (alpha / self.n_neg) * ((dsig_neg * sd_neg[:, None]).T @ self.X_neg)  # noqa: N806

        total_gradient: Float64Array = np.zeros(w.shape, dtype=np.float64)
        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # noqa: N806
            if not np.any(R_kM):
                continue

            # da/dF_0
            static_kw0, n_p0, n_p1, n_F0, n_F1 = self.da0_reqs[k]  # noqa: N806
            args_da0 = static_kw0.copy()
            if n_p0:
                args_da0['pi_0'] = self.pi0
            if n_p1:
                args_da0['pi_1'] = self.pi1
            if n_F0:
                args_da0['F_0'] = seg_tprs
            if n_F1:
                args_da0['F_1'] = seg_fprs
            da_dF0_M = np.broadcast_to(np.asarray(self.da_dF0_fns[k](**args_da0), dtype=np.float64), (M,))  # noqa: N806

            # da/dF_1
            static_kw1, n_p0, n_p1, n_F0, n_F1 = self.da1_reqs[k]  # noqa: N806
            args_da1 = static_kw1.copy()
            if n_p0:
                args_da1['pi_0'] = self.pi0
            if n_p1:
                args_da1['pi_1'] = self.pi1
            if n_F0:
                args_da1['F_0'] = seg_tprs
            if n_F1:
                args_da1['F_1'] = seg_fprs
            da_dF1_M = np.broadcast_to(np.asarray(self.da_dF1_fns[k](**args_da1), dtype=np.float64), (M,))  # noqa: N806

            weight_F0 = R_kM * da_dF0_M  # noqa: N806
            weight_F1 = R_kM * da_dF1_M  # noqa: N806
            total_gradient += (weight_F0 @ grad_F0_M) + (weight_F1 @ grad_F1_M)

        return total_gradient

    def _regularization_value(self, coef: FloatNDArray) -> float:
        """Regularization contribution to the scalar objective."""
        if self.l1_ratio == 0.0:
            return 0.5 * float(np.dot(coef, coef)) / self.C
        if self.l1_ratio == 1.0:
            return float(np.sum(np.abs(coef))) / self.C
        return (
            (1.0 - self.l1_ratio) * 0.5 * float(np.dot(coef, coef)) + self.l1_ratio * float(np.sum(np.abs(coef)))
        ) / self.C

    def _regularization_gradient(self, coef: FloatNDArray) -> Float64Array:
        """Regularization contribution to the gradient."""
        coef_f = np.asarray(coef, dtype=np.float64)
        if self.l1_ratio == 0.0:
            return coef_f / self.C
        if self.l1_ratio == 1.0:
            return np.sign(coef_f) / self.C
        return ((1.0 - self.l1_ratio) * coef_f + self.l1_ratio * np.sign(coef_f)) / self.C

    def _current_alpha(self) -> float:
        """Compute annealed temperature for the current objective evaluation."""
        # External override takes precedence (set by an alpha_schedule on the optimizer)
        if self._alpha_override is not None:
            return float(self._alpha_override)
        try:
            alpha = self.alpha_0 * (self.alpha_growth**self._epoch)
        except OverflowError:
            alpha = self.alpha_max
        return float(min(self.alpha_max, alpha))

    def set_alpha(self, alpha: float) -> None:
        """Override the smoothing parameter for the next gradient computation.

        When called by a gradient optimizer with an ``alpha_schedule``, the
        internal epoch-based annealing is bypassed and *alpha* is used directly.

        Parameters
        ----------
        alpha : float
            Smoothing parameter value.
        """
        self._alpha_override = float(alpha)

    def reset(self) -> None:
        """Reset the epoch counter (annealing schedule) and clear any alpha override."""
        self._epoch = 0
        self._alpha_override = None

    def with_indices(self, indices: FloatNDArray) -> 'MaxProfitLogitGradientPiecewise':
        """Return a shallow copy of this objective restricted to *indices*.

        The expensive pre-computed attributes (lambdified sympy expressions,
        distribution bounds, etc.) are shared with the original object.  Only
        the data-dependent attributes (``features``, ``y_true``, masks, class
        counts, and class rates) are re-derived for the batch.

        Parameters
        ----------
        indices : array-like of int
            Row indices into the full training set.

        Returns
        -------
        MaxProfitLogitGradientPiecewise
            A new objective for the selected samples.
        """
        obj = copy.copy(self)
        obj.features = self.features[indices]
        obj.y_true = self.y_true[indices]  # already int32
        obj.pos_mask = obj.y_true == 1
        obj.neg_mask = ~obj.pos_mask
        obj.n_pos = int(obj.pos_mask.sum())
        obj.n_neg = int(obj.neg_mask.sum())
        obj.X_pos = np.asarray(obj.features[obj.pos_mask], dtype=np.float64)
        obj.X_neg = np.asarray(obj.features[obj.neg_mask], dtype=np.float64)
        obj.pi0 = float(obj.n_pos / len(obj.y_true))
        obj.pi1 = 1.0 - obj.pi0
        obj._epoch = 0
        obj._alpha_override = self._alpha_override
        return obj

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Compute the negated EMP objective value (no gradient).

        Always uses a freshly computed convex hull.  The epoch counter is *not*
        incremented, because this method is intended for fitness evaluation rather
        than gradient-based updates.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Negated EMP loss (suitable for minimization).
        """
        w = self._apply_soft_threshold(weights)
        y_score = self._compute_y_score(w)
        bounds, seg_tprs, seg_fprs, m = self._compute_hull_state(y_score)

        value = float(-self._accumulate_value(bounds, seg_tprs, seg_fprs, m))
        start_coef = 1 if self.fit_intercept else 0
        value += self._regularization_value(w[start_coef:])
        return value

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Compute the gradient of the negated EMP objective (no scalar value).

        Always uses a freshly computed convex hull.  Increments the epoch counter
        so the annealing schedule advances exactly once per gradient step.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient vector matched in shape to *weights*.
        """
        w = self._apply_soft_threshold(weights)
        alpha = self._current_alpha()
        self._epoch += 1

        y_score = self._compute_y_score(w)
        bounds, seg_tprs, seg_fprs, m = self._compute_hull_state(y_score)
        T_M = self._compute_thresholds(y_score, seg_tprs, seg_fprs)  # noqa: N806

        grad = -self._accumulate_gradient(w, y_score, T_M, bounds, seg_tprs, seg_fprs, m, alpha)
        start_coef = 1 if self.fit_intercept else 0
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

        Yields
        ------
        gradient : ndarray
            Negated, regularized gradient at the current weights.

        Receives (via ``send``)
        -----------------------
        weights : ndarray
            New coefficient vector for the next gradient step.  The cached hull
            is reused; only scores and thresholds are recomputed.
        (weights, refresh) : (ndarray, bool)
            Pass ``refresh=True`` to force the convex hull to be rebuilt from the
            new *weights* before computing the gradient.

        Examples
        --------
        >>> gen = objective.logit_gradient_steps(theta)
        >>> grad = next(gen)  # hull built from theta
        >>> grad = gen.send(new_theta)  # hull reused
        >>> grad = gen.send((new_theta, True))  # hull refreshed
        >>> gen.close()
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

            w = self._apply_soft_threshold(weights)
            alpha = self._current_alpha()
            self._epoch += 1

            y_score = self._compute_y_score(w)

            if cached is None:
                cached = self._compute_hull_state(y_score)

            bounds, seg_tprs, seg_fprs, m = cached
            thresholds = self._compute_thresholds(y_score, seg_tprs, seg_fprs)

            grad = -self._accumulate_gradient(w, y_score, thresholds, bounds, seg_tprs, seg_fprs, m, alpha)

            start_coef = 1 if self.fit_intercept else 0
            grad[start_coef:] += self._regularization_gradient(w[start_coef:])

            sent = yield grad

    def logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients while reusing a cached convex hull.

        The convex hull (and derived piecewise segment arrays) are computed once
        from *initial_weights* on the first call, then reused for subsequent
        gradient steps.  This avoids the hull-reconstruction cost inside tight
        local-search loops (e.g. a few Adam steps applied Lamarckian-style before
        a fitness evaluation) where the hull is unlikely to change substantially.

        When computing the actual loss (``logit_loss()``) always use a fresh hull.

        Parameters
        ----------
        initial_weights : ndarray
            Starting coefficient vector.  The convex hull is built from these
            scores on the first iteration.

        Yields
        ------
        gradient : ndarray
            Negated, regularized gradient at the current weights.

        Receives (via ``send``)
        -----------------------
        weights : ndarray
            New coefficient vector for the next gradient step.  The cached hull
            is reused; only scores and thresholds are recomputed.
        (weights, refresh) : (ndarray, bool)
            Pass ``refresh=True`` to force the convex hull to be rebuilt from the
            new *weights* before computing the gradient.

        Examples
        --------
        >>> gen = objective.logit_gradient_steps()
        >>> grad = gen.send(theta)  # first time hull is built
        >>> grad = gen.send(theta)  # hull reused
        >>> grad = gen.send((new_theta, True))  # hull refreshed
        >>> gen.close()
        """
        generator = self._logit_gradient_steps()
        next(generator)
        return generator

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated stochastic EMP objective and its gradient for minimization.

        Computes both value and gradient from a freshly built convex hull and increments
        the epoch counter once.

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
        w = self._apply_soft_threshold(weights)
        alpha = self._current_alpha()
        self._epoch += 1

        y_score = self._compute_y_score(w)
        bounds, seg_tprs, seg_fprs, m = self._compute_hull_state(y_score)
        T_M = self._compute_thresholds(y_score, seg_tprs, seg_fprs)  # noqa: N806

        total_value = self._accumulate_value(bounds, seg_tprs, seg_fprs, m)
        total_gradient = self._accumulate_gradient(w, y_score, T_M, bounds, seg_tprs, seg_fprs, m, alpha)

        value = float(-total_value)
        gradient = -total_gradient

        start_coef = 1 if self.fit_intercept else 0
        coef = w[start_coef:]
        value += self._regularization_value(coef)
        gradient[start_coef:] += self._regularization_gradient(coef)

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

        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = max(int(np.sum(self.pos_mask)), 1)
        self.n_neg = max(int(np.sum(self.neg_mask)), 1)

        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

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

        self.da0_reqs = []
        self.da1_reqs = []
        for k in range(len(self.score_function.coefficient_eqs)):

            def _get_reqs(eq: sympy.Expr) -> tuple[dict[str, Any], bool, bool, bool, bool]:
                reqs = {str(s) for s in eq.free_symbols}
                static_kws = {key: val for key, val in self.kwargs.items() if key in reqs}
                return static_kws, 'pi_0' in reqs, 'pi_1' in reqs, 'F_0' in reqs, 'F_1' in reqs

            self.da0_reqs.append(_get_reqs(self.da_dF0_eqs[k]))
            self.da1_reqs.append(_get_reqs(self.da_dF1_eqs[k]))

    def __call__(self, y_score: FloatNDArray, alpha: float) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the stochastic objective."""
        y_score_arr = expit(np.asarray(y_score, dtype=np.float64).reshape(-1))

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

        # Vectorized Thresholds
        rates = np.clip(segment_tprs_arr * self.pi0 + segment_fprs_arr * self.pi1, 0.0, 1.0)
        T_M = np.quantile(y_score_arr, 1.0 - rates)  # noqa: N806

        # Precompute logistic derivatives for instances
        s_pos = y_score_arr[self.pos_mask]
        s_neg = y_score_arr[self.neg_mask]

        sig_pos = expit(alpha * np.subtract.outer(s_pos, T_M))
        sig_neg = expit(alpha * np.subtract.outer(s_neg, T_M))

        sig_prime_pos = alpha * sig_pos * (1.0 - sig_pos)
        sig_prime_neg = alpha * sig_neg * (1.0 - sig_neg)

        sig_sec_pos = alpha**2 * sig_pos * (1.0 - sig_pos) * (1.0 - 2.0 * sig_pos)
        sig_sec_neg = alpha**2 * sig_neg * (1.0 - sig_neg) * (1.0 - 2.0 * sig_neg)

        weight_F0_M = np.zeros(M)  # noqa: N806
        weight_F1_M = np.zeros(M)  # noqa: N806

        for k in range(len(self.score_function.coefficient_eqs)):
            k_mom, cdf_diffs = self.score_function._get_kth_integration_components(bounds, k, self.dist_params)
            R_kM = float(k_mom) * np.asarray(cdf_diffs)  # noqa: N806

            if not np.any(R_kM):
                continue

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

        # Convert to minimization constants per segment
        c_pos_M = -weight_F0_M / self.n_pos  # noqa: N806
        c_neg_M = -weight_F1_M / self.n_neg  # noqa: N806

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
