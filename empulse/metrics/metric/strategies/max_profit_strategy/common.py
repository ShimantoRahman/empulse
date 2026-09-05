import copy
from collections.abc import Iterable
from typing import Any, Literal, Self, overload

import numpy as np
import sympy
from scipy.special import expit
from sympy.utilities import lambdify

from ....._types import Float64Array, FloatNDArray, IntNDArray
from ...._cy_convex_hull import convex_hull
from ..metric_strategy import LogitObjective


def _convex_hull(y_true: IntNDArray, y_score: FloatNDArray) -> tuple[IntNDArray, FloatNDArray]:
    return convex_hull(y_true.astype(np.int32), y_score.astype(np.float64))  # type: ignore[no-any-return]


def extract_distribution_parameters(
    parameters: dict[str, Any], distribution_args: Iterable[sympy.Symbol]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the distribution parameters from the other parameters."""
    distribution_parameters = {
        str(key): parameters.pop(str(key)) for key in distribution_args if str(key) in parameters
    }
    return distribution_parameters, parameters


def _substitute_integrand(
    expr: sympy.Expr,
    kwargs: dict[str, Any],
    dist_params: dict[str, Any],
    pi0: float,
    pi1: float,
) -> sympy.Expr:
    """Substitute deterministic parameters, distribution parameters, and class priors into *expr*.

    ``dist_params={}`` is a no-op substitution, so this covers both the "distribution parameters
    are fixed numeric literals" and "distribution parameters were just resolved from kwargs" cases
    used by the Monte-Carlo, Quasi-Monte-Carlo, and quadrature integration backends.
    """
    return expr.subs(kwargs).subs(dist_params).subs('pi_0', pi0).subs('pi_1', pi1)


def _evaluate_sampled_integrands(
    profit_integrand: sympy.Expr,
    rate_integrand: sympy.Expr | None,
    true_positive_rates: Iterable[float],
    false_positive_rates: Iterable[float],
    random_symbols: Iterable[sympy.Symbol],
    param_grid: list[Any],
    n_samples: int,
) -> float:
    """Evaluate a profit (and optional rate) integrand over a sampled parameter grid.

    Used by both the Monte-Carlo and Quasi-Monte-Carlo integration backends, which differ only
    in how *param_grid* is generated (plain sympy sampling vs. a Sobol sequence mapped through
    each distribution's inverse CDF). For each convex-hull point, lambdifies the integrand at
    that point's F_0/F_1 and evaluates it over every sample in *param_grid*.

    Returns
    -------
    float
        When *rate_integrand* is ``None``: the mean, over samples, of the per-sample maximum
        profit. Otherwise: the mean, over samples, of the predicted-positive rate at each
        sample's profit-maximizing convex-hull point.
    """
    profit_integrands = [
        lambdify(random_symbols, profit_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
        for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
    ]

    results = np.empty((len(profit_integrands), n_samples))
    for i, integrand in enumerate(profit_integrands):
        results[i, :] = integrand(*param_grid)
    if rate_integrand is None:
        return float(results.max(axis=0).mean())

    rate_integrands = [
        lambdify(random_symbols, rate_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
        for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
    ]
    rate_results = np.empty((len(profit_integrands), n_samples))
    best_indices = results.argmax(axis=0)
    for i, integrand in enumerate(rate_integrands):
        rate_results[i, :] = integrand(*param_grid)
    return float(rate_results[best_indices, np.arange(n_samples)].mean())


@overload
def _smooth_step_derivatives(
    diff: FloatNDArray, alpha: float, order: Literal[1]
) -> tuple[FloatNDArray, FloatNDArray]: ...
@overload
def _smooth_step_derivatives(
    diff: FloatNDArray, alpha: float, order: Literal[2] = 2
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]: ...
def _smooth_step_derivatives(
    diff: FloatNDArray, alpha: float, order: Literal[1, 2] = 2
) -> tuple[FloatNDArray, FloatNDArray] | tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    """Sigmoid approximation of a step function, and its derivative factors.

    *diff* is ``scores - threshold`` (or ``np.subtract.outer(scores, thresholds)`` for a matrix
    of per-segment thresholds). Returns ``sigma = expit(alpha * diff)`` together with the
    *unscaled* logistic factors ``sigma * (1 - sigma)`` (``order=1``) and, for ``order=2``
    (default), also ``sigma * (1 - sigma) * (1 - 2 * sigma)``.

    The true first and second derivatives of ``sigma`` with respect to the *scores* axis are
    these factors multiplied by ``alpha`` and ``alpha**2`` respectively - callers apply that
    scaling themselves, so it can be combined with any other per-caller ``alpha`` factor (e.g.
    an outer ``alpha / n_pos`` term) without multiplying by ``alpha`` twice.
    """
    sigma = expit(alpha * diff)
    factor1 = sigma * (1.0 - sigma)
    if order == 1:
        return sigma, factor1
    factor2 = factor1 * (1.0 - 2.0 * sigma)
    return sigma, factor1, factor2


class _BaseMaxProfitLogitObjective(LogitObjective):
    """Shared state and helpers for MaxProfit's logistic-regression objectives.

    Both the deterministic and piecewise-stochastic MaxProfit logit objectives need identical
    soft-thresholding, elastic-net regularization, alpha-based smoothing, and index-based
    mini-batch slicing; this base class holds that shared machinery so the two subclasses only
    implement their differing profit/gradient computation.
    """

    def __init__(
        self,
        *,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        alpha: float,
    ) -> None:
        self.features = features
        self.y_true = y_true.ravel().astype(np.int32)
        self.C = C
        self.l1_ratio = l1_ratio
        self.soft_threshold = soft_threshold
        self.fit_intercept = fit_intercept
        self.alpha = float(alpha)
        self._refresh_sample_state()

    def _refresh_sample_state(self) -> None:
        """(Re)derive sample masks, counts, and per-class feature matrices from features/y_true."""
        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = int(self.pos_mask.sum())
        self.n_neg = int(self.neg_mask.sum())
        self.X_pos: Float64Array = np.asarray(self.features[self.pos_mask], dtype=np.float64)
        self.X_neg: Float64Array = np.asarray(self.features[self.neg_mask], dtype=np.float64)
        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

    @property
    def _start_coef(self) -> int:
        return 1 if self.fit_intercept else 0

    def set_alpha(self, alpha: float) -> None:
        """Override the smoothing parameter used for subsequent gradient computations.

        Parameters
        ----------
        alpha : float
            Smoothing parameter value.
        """
        self.alpha = float(alpha)

    def _apply_soft_threshold(self, weights: FloatNDArray) -> FloatNDArray:
        """Return a copy of *weights* with soft-thresholding applied (if enabled)."""
        start_coef = self._start_coef
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

    def with_indices(self, indices: FloatNDArray) -> Self:
        """Return a shallow copy of this objective restricted to *indices*.

        Parameters
        ----------
        indices : array-like of int
            Row indices into the full training set.

        Returns
        -------
        Self
            A new objective for the selected samples.
        """
        obj = copy.copy(self)
        obj.features = self.features[indices]
        obj.y_true = self.y_true[indices]  # already int32
        obj._refresh_sample_state()
        return obj
