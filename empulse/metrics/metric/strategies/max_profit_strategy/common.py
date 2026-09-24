import copy
from collections.abc import Iterable
from typing import Any, Literal, Self, overload

import numpy as np
import sympy
from scipy.special import expit
from sympy.utilities import lambdify

from ....._common._objective import ElasticNetPenalty, LogitObjective
from ....._types import Float64Array, FloatNDArray, IntNDArray
from ...._cy_convex_hull import convex_hull, convex_hull_from_counts
from ..._compile import CountScoreFn
from ..._parameter_domain import _check_parameters


def _convex_hull(y_true: IntNDArray, y_score: FloatNDArray) -> tuple[IntNDArray, FloatNDArray]:
    return convex_hull(y_true.astype(np.int32), y_score.astype(np.float64))  # type: ignore[no-any-return]


def _convex_hull_from_counts(
    y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray
) -> tuple[FloatNDArray, FloatNDArray]:
    return convex_hull_from_counts(  # type: ignore[no-any-return]
        np.asarray(y_score, dtype=np.float64),
        np.asarray(n_positive, dtype=np.int64),
        np.asarray(n_negative, dtype=np.int64),
    )


class _HullScoreFunction:
    """
    Mixin for score functions that see the samples only through their ROC convex hull and class prior.

    Subclasses implement :meth:`_score_hull`. Since that is all they need, they can also score samples
    grouped by score (:meth:`_count_scorer`), such as the leaves of a decision tree, without
    expanding the groups back into samples.
    """

    deterministic_symbols: Iterable[sympy.Symbol]
    dist_params: list[sympy.Expr]

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        raise NotImplementedError

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)
        positive_class_prior = float(np.mean(y_true))
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)
        return self._score_hull(true_positive_rates, false_positive_rates, positive_class_prior, kwargs)

    def _count_scorer(self, **kwargs: Any) -> CountScoreFn:
        """
        Prepare the score of samples grouped by score, for parameter values fixed across many calls.

        The parameters are checked once here rather than on every call. The returned function takes
        each group's score and its numbers of positive and negative samples, and gives the same score
        as calling this score function on the samples themselves.
        """
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        def score(y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray) -> float:
            n_positives = int(np.sum(n_positive))
            positive_class_prior = n_positives / (n_positives + int(np.sum(n_negative)))
            true_positive_rates, false_positive_rates = _convex_hull_from_counts(y_score, n_positive, n_negative)
            # _score_hull may consume entries of the parameters, so each call gets its own copy.
            return self._score_hull(true_positive_rates, false_positive_rates, positive_class_prior, dict(kwargs))

        return score


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
    elastic-net regularization, alpha-based smoothing, and index-based
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
        fit_intercept: bool,
        alpha: float,
        objective_scale: float = 1.0,
    ) -> None:
        self.features = features
        self.y_true = y_true.ravel().astype(np.int32)
        self.fit_intercept = fit_intercept
        self.alpha = float(alpha)
        self._refresh_sample_state()
        self.penalty = ElasticNetPenalty.from_scale(
            objective_scale=objective_scale,
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_samples=len(self.y_true),
        )

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

    def _compute_y_score(self, w: FloatNDArray) -> FloatNDArray:
        """Compute logistic scores for every sample."""
        return expit(self.features @ w)  # type: ignore[return-value]

    def _regularization_value(self, coef: FloatNDArray) -> float:
        """Regularization contribution to the scalar objective, for already-sliced coefficients."""
        coef_f = np.asarray(coef, dtype=np.float64)
        penalty = self.penalty
        if penalty is None or not penalty.is_active:
            return 0.0
        return penalty.l1_weight * float(np.sum(np.abs(coef_f))) + penalty.l2_value(coef_f)

    def _regularization_gradient(self, coef: FloatNDArray) -> Float64Array:
        """Regularization contribution to the gradient, for already-sliced coefficients."""
        coef_f: Float64Array = np.asarray(coef, dtype=np.float64)
        penalty = self.penalty
        if penalty is None or not penalty.is_active:
            return np.zeros_like(coef_f)
        return penalty.l1_weight * np.sign(coef_f) + penalty.l2_gradient(coef_f)

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
