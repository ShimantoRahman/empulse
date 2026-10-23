import copy
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from typing import Any, Literal, Self, overload

import numpy as np
import sympy
from scipy.special import expit

from ....._common._objective import ElasticNetPenalty, LogitObjective
from ....._types import Float64Array, FloatNDArray, IntNDArray
from ...._cy_convex_hull import convex_hull, convex_hull_from_counts, convex_hull_full, convex_hull_full_from_counts
from ..._compile import CountScoreFn, _safe_lambdify
from ..._parameter_domain import _check_parameters
from .envelope import _probe


def _convex_hull(y_true: IntNDArray, y_score: FloatNDArray, *, full: bool = False) -> tuple[FloatNDArray, FloatNDArray]:
    """
    Return the ROC convex hull as (TPR, FPR), for labels and scores of any numeric dtype and layout.

    The compiled hull only accepts int32 labels and float64 scores, so any other dtype is converted
    here. Arrays that already have them are passed on without a copy. With ``full``, the vertices of
    the lower hull follow, see :func:`~empulse.metrics._cy_convex_hull.convex_hull_full`.
    """
    hull = convex_hull_full if full else convex_hull
    return hull(  # type: ignore[no-any-return]
        np.asarray(y_true, dtype=np.int32),
        np.asarray(y_score, dtype=np.float64),
    )


def _convex_hull_from_counts(
    y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray, *, full: bool = False
) -> tuple[FloatNDArray, FloatNDArray]:
    hull = convex_hull_full_from_counts if full else convex_hull_from_counts
    return hull(  # type: ignore[no-any-return]
        np.asarray(y_score, dtype=np.float64),
        np.asarray(n_positive, dtype=np.int64),
        np.asarray(n_negative, dtype=np.int64),
    )


class _LowerChainGate:
    """
    Decide whether the lower hull of the ROC curve can hold the most profitable operating point.

    The profit is linear in the rates, ``constant + a * F_0 + b * F_1``, with
    ``a = pi_0 * (tp_benefit + fn_cost)`` and ``b = -pi_1 * (tn_benefit + fp_cost)``. Its maximum over
    the ROC curve lies on the upper hull, which includes (0, 0) and (1, 1), unless ``a < 0 < b``:
    targeting a positive costs more than missing it, while targeting a negative gains. Only then can
    it lie below the diagonal, on the lower hull. That takes a cost matrix whose benefits are
    inverted, so the lower hull is built only when the coefficients can take those signs.

    Parameters
    ----------
    profit_function : sympy.Expr
        The profit, linear in ``F_0`` and ``F_1``.
    random_symbols : sequence of sympy.stats.rv.RandomSymbol
        The stochastic variables, in the order of the sample grids the gate is given.
    """

    def __init__(self, profit_function: sympy.Expr, random_symbols: Sequence[sympy.Symbol]) -> None:
        pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
        # The priors are positive, so they do not change the signs.
        coefficients = (
            sympy.diff(profit_function, tpr).subs(pos_prior, 1),
            sympy.diff(profit_function, fpr).subs(neg_prior, 1),
        )
        self.random_symbols = list(random_symbols)
        parameters = sorted(
            {symbol for coefficient in coefficients for symbol in coefficient.free_symbols} - set(self.random_symbols),
            key=str,
        )
        self.parameter_names = [str(symbol) for symbol in parameters]
        arguments = [*parameters, *self.random_symbols]
        self._coefficients = [_safe_lambdify(coefficient, arguments, dummify=True) for coefficient in coefficients]
        self._is_random = any(coefficient.has(*self.random_symbols) for coefficient in coefficients)
        # Polynomial coefficients in a single stochastic variable, in ascending powers.
        self._polynomials: list[list[Callable[..., Any]]] | None = None
        if len(self.random_symbols) == 1:
            try:
                polynomials = [sympy.Poly(sympy.expand(c), self.random_symbols[0]) for c in coefficients]
            except (sympy.polys.polyerrors.PolynomialError, TypeError, ValueError, NotImplementedError):
                polynomials = None
            if polynomials is not None:
                self._polynomials = [
                    [_safe_lambdify(term, parameters, dummify=True) for term in reversed(polynomial.all_coeffs())]
                    for polynomial in polynomials
                ]
        # The last parameters and their answer, published as one tuple for concurrent callers.
        self._cache: tuple[tuple[Any, ...], bool] | None = None

    def _values(self, parameters: dict[str, Any]) -> list[Any] | None:
        try:
            return [parameters[name] for name in self.parameter_names]
        except KeyError:
            return None

    def _cached(self, key: tuple[Any, ...], decide: Callable[[], bool]) -> bool:
        cached = self._cache
        if cached is not None and cached[0] == key:
            return cached[1]
        answer = decide()
        self._cache = (key, answer)
        return answer

    def on_support(self, parameters: dict[str, Any], lower_bound: float, upper_bound: float) -> bool:
        """Whether ``a < 0 < b`` anywhere in the support of a single stochastic variable."""
        values = self._values(parameters)
        if values is None:
            return True
        if not self._is_random:
            return self._at(values, [])
        if self._polynomials is None:
            return True  # not a polynomial: assume it can
        try:
            key = (*(float(value) for value in values), float(lower_bound), float(upper_bound))
        except (TypeError, ValueError):
            return True
        return self._cached(key, lambda: self._polynomials_invert(values, lower_bound, upper_bound))

    def _polynomials_invert(self, values: list[Any], lower_bound: float, upper_bound: float) -> bool:
        assert self._polynomials is not None
        polynomials = [
            np.array([float(term(*values)) for term in polynomial], dtype=np.float64)
            for polynomial in self._polynomials
        ]
        edges = [lower_bound, upper_bound]
        for coefficients in polynomials:
            nonzero = np.flatnonzero(coefficients)
            if len(nonzero) and nonzero[-1] > 0:
                roots = np.roots(coefficients[: nonzero[-1] + 1][::-1])
                real = roots.real[np.abs(roots.imag) <= 1e-12 * np.maximum(1.0, np.abs(roots.real))]
                edges.extend(root for root in real if lower_bound < root < upper_bound)
        edges.sort()
        for lower, upper in pairwise(edges):
            if lower == upper:
                continue
            x = _probe(lower, upper)
            tpr_coefficient, fpr_coefficient = (np.polynomial.polynomial.polyval(x, c) for c in polynomials)
            if tpr_coefficient < 0 < fpr_coefficient:
                return True
        return False

    def on_samples(self, parameters: dict[str, Any], param_grid: Sequence[Any]) -> bool:
        """Whether ``a < 0 < b`` at any of the samples of the stochastic variables."""
        values = self._values(parameters)
        if values is None:
            return True
        return self._at(values, param_grid)

    def _at(self, values: list[Any], param_grid: Sequence[Any]) -> bool:
        if not self._is_random:
            param_grid = [np.nan] * len(self.random_symbols)
        tpr_coefficient, fpr_coefficient = (
            np.asarray(function(*values, *param_grid), dtype=np.float64) for function in self._coefficients
        )
        return bool(np.any((tpr_coefficient < 0) & (fpr_coefficient > 0)))


class _HullScoreFunction:
    """
    Mixin for score functions that see the samples only through their ROC convex hull and class prior.

    Subclasses implement :meth:`_score_hull`. Since that is all they need, they can also score samples
    grouped by score (:meth:`_count_scorer`), such as the leaves of a decision tree, without
    expanding the groups back into samples.
    """

    deterministic_symbols: Iterable[sympy.Symbol]
    dist_params: list[sympy.Symbol]

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        raise NotImplementedError

    def _needs_lower_hull(self, kwargs: dict[str, Any]) -> bool:
        """
        Whether the lower hull can hold the most profitable vertex for these parameters.

        :meth:`_score_hull` then gets the whole convex hull, see :class:`_LowerChainGate`. True
        unless a subclass can tell otherwise.
        """
        return True

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)
        positive_class_prior = float(np.mean(y_true))
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score, full=self._needs_lower_hull(kwargs))
        return self._score_hull(true_positive_rates, false_positive_rates, positive_class_prior, kwargs)

    def _count_scorer(self, **kwargs: Any) -> CountScoreFn:
        """
        Prepare the score of samples grouped by score, for parameter values fixed across many calls.

        The parameters are checked once here rather than on every call. The returned function takes
        each group's score and its numbers of positive and negative samples, and gives the same score
        as calling this score function on the samples themselves.
        """
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)
        full = self._needs_lower_hull(dict(kwargs))

        def score(y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray) -> float:
            n_positives = int(np.sum(n_positive))
            positive_class_prior = n_positives / (n_positives + int(np.sum(n_negative)))
            true_positive_rates, false_positive_rates = _convex_hull_from_counts(
                y_score, n_positive, n_negative, full=full
            )
            # _score_hull may consume entries of the parameters, so each call gets its own copy.
            return self._score_hull(true_positive_rates, false_positive_rates, positive_class_prior, dict(kwargs))

        return score

    def _sample_scorer(self, y_true: IntNDArray, **kwargs: Any) -> Callable[[FloatNDArray], float]:
        """
        Prepare the score of fixed labels, for parameter values fixed across many calls.

        The parameters are checked, and the labels converted and counted, once here. The returned
        function takes the scores and gives the same result as calling this score function.
        """
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)
        y_true = np.ascontiguousarray(y_true, dtype=np.int32).reshape(-1)
        positive_class_prior = float(np.mean(y_true))
        full = self._needs_lower_hull(dict(kwargs))

        def score(y_score: FloatNDArray) -> float:
            true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score, full=full)
            # _score_hull may consume entries of the parameters, so each call gets its own copy.
            return self._score_hull(true_positive_rates, false_positive_rates, positive_class_prior, dict(kwargs))

        return score


def _distribution_parameter_symbols(distribution_args: Iterable[sympy.Expr]) -> list[sympy.Symbol]:
    """Find the symbols the caller gives values for in the distributions' arguments.

    An argument can be an expression of them (``Beta('v', 2 * a, b)``), so these are the arguments'
    free symbols rather than the arguments themselves: each named once, in order of appearance.
    """
    symbols: dict[str, sympy.Symbol] = {}
    for argument in distribution_args:
        for symbol in sorted(sympy.sympify(argument).free_symbols, key=lambda symbol: symbol.name):
            symbols.setdefault(symbol.name, symbol)
    return list(symbols.values())


def extract_distribution_parameters(
    parameters: dict[str, Any], distribution_args: Iterable[sympy.Expr]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the distribution parameters from the other parameters.

    ``distribution_args`` may be the distributions' arguments or their symbols, see
    :func:`_distribution_parameter_symbols`.
    """
    distribution_parameters = {
        symbol.name: parameters.pop(symbol.name)
        for symbol in _distribution_parameter_symbols(distribution_args)
        if symbol.name in parameters
    }
    return distribution_parameters, parameters


#: Upper bound on the number of (hull point, sample) values held in memory at once. The samples are
#: evaluated in chunks of at most this many values divided by the number of hull points (32 MiB).
_MAX_HULL_EVALUATIONS = 2**22


class _SampledIntegrand:
    """
    A profit or rate function compiled for evaluation over samples of its stochastic variables.

    The function is a polynomial in the ROC point ``F_0``, ``F_1`` and the class priors ``pi_0``,
    ``pi_1``, with coefficients in the stochastic and deterministic variables. The coefficients are
    compiled once, here. Their values over the samples depend on neither the ROC convex hull nor
    the class priors, so the values for the last parameters are cached: scoring again with the same
    parameters, as a training loop does, only combines them with the hull and the priors.

    Parameters
    ----------
    expression : sympy.Expr
        The profit or rate function.
    random_symbols : sequence of sympy.stats.rv.RandomSymbol
        The stochastic variables, in the order of the sample grid's entries.
    """

    def __init__(self, expression: sympy.Expr, random_symbols: Iterable[sympy.Symbol]) -> None:
        generators = sympy.symbols('pi_0 pi_1 F_0 F_1')
        polynomial = sympy.Poly(sympy.expand(expression), *generators)
        random_symbols = list(random_symbols)
        coefficients = polynomial.coeffs()
        # A random symbol's free symbols are itself, not its distribution's parameters, so this
        # finds the deterministic variables (and any distribution parameter used as one).
        parameters = sorted(
            {symbol for coefficient in coefficients for symbol in coefficient.free_symbols} - set(random_symbols),
            key=str,
        )
        self.parameter_names = [str(symbol) for symbol in parameters]
        #: The exponents of ``(pi_0, pi_1, F_0, F_1)`` of each term, matching :attr:`coefficients`.
        self.monomials: list[tuple[int, ...]] = polynomial.monoms()
        self.coefficients = [
            _safe_lambdify(coefficient, [*parameters, *random_symbols], dummify=True) for coefficient in coefficients
        ]
        # The parameter values, the grid and the coefficients' values over it are cached as one
        # tuple, so a concurrent caller can never pair one call's parameters with another's values.
        self._cache: tuple[list[Any], list[Any], list[Float64Array]] | None = None

    def _coefficient_values(self, parameters: dict[str, Any], param_grid: list[Any]) -> list[Float64Array]:
        """Return each coefficient's values over *param_grid*, of shape (n_samples,) or () if constant."""
        values = [parameters[name] for name in self.parameter_names]
        cached = self._cache
        if cached is not None and cached[1] is param_grid and cached[0] == values:
            return cached[2]
        coefficient_values = [
            np.asarray(coefficient(*values, *param_grid), dtype=np.float64) for coefficient in self.coefficients
        ]
        self._cache = (values, param_grid, coefficient_values)
        return coefficient_values

    def terms(
        self,
        parameters: dict[str, Any],
        param_grid: list[Any],
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
    ) -> list[tuple[FloatNDArray, FloatNDArray]]:
        """
        Split the function over the ROC convex hull and the samples into a sum of outer products.

        Returns
        -------
        list of (ndarray of shape (n_hull_points, 1), ndarray of shape (n_samples,) or ())
            Pairs whose products, summed, are the function at every hull point (rows) and sample
            (columns): one pair per power of ``F_0`` and ``F_1``, with the priors folded into the
            samples' side.
        """
        priors = (positive_class_prior, 1.0 - positive_class_prior)
        sample_sides: dict[tuple[int, ...], Any] = {}
        for (pos_power, neg_power, *roc_powers), value in zip(
            self.monomials, self._coefficient_values(parameters, param_grid), strict=True
        ):
            term = priors[0] ** pos_power * priors[1] ** neg_power * value
            key = tuple(roc_powers)
            sample_sides[key] = sample_sides[key] + term if key in sample_sides else term
        return [
            ((true_positive_rates**tpr_power * false_positive_rates**fpr_power)[:, np.newaxis], sample_side)
            for (tpr_power, fpr_power), sample_side in sample_sides.items()
        ]


def _evaluate_terms(terms: list[tuple[FloatNDArray, FloatNDArray]], samples: slice, shape: tuple[int, int]) -> Any:
    """Sum the outer products of :meth:`_SampledIntegrand.terms` over a slice of the samples."""
    total: Any = 0.0
    for hull_side, sample_side in terms:
        total = total + hull_side * (sample_side[samples] if sample_side.ndim else sample_side)
    return np.broadcast_to(total, shape)


def _evaluate_sampled_integrands(
    profit_integrand: _SampledIntegrand,
    rate_integrand: _SampledIntegrand | None,
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    positive_class_prior: float,
    parameters: dict[str, Any],
    param_grid: list[Any],
    n_samples: int,
    controls: Sequence[FloatNDArray] = (),
) -> float:
    """Evaluate a profit (and optional rate) integrand over a sampled parameter grid.

    Used by both the Monte-Carlo and Quasi-Monte-Carlo integration backends, which differ only
    in how *param_grid* is generated (plain sympy sampling vs. a Sobol sequence mapped through
    each distribution's inverse CDF). Evaluates the integrand at every convex-hull point and every
    sample in *param_grid*, in chunks of samples to bound the memory this takes.

    *controls* are optional control variates, see :func:`_control_variate_mean`.

    Returns
    -------
    float
        When *rate_integrand* is ``None``: the mean, over samples, of the per-sample maximum
        profit. Otherwise: the mean, over samples, of the predicted-positive rate at each
        sample's profit-maximizing convex-hull point.
    """
    true_positive_rates = np.asarray(true_positive_rates, dtype=np.float64)
    false_positive_rates = np.asarray(false_positive_rates, dtype=np.float64)
    profit_terms = profit_integrand.terms(
        parameters, param_grid, true_positive_rates, false_positive_rates, positive_class_prior
    )
    rate_terms = (
        rate_integrand.terms(parameters, param_grid, true_positive_rates, false_positive_rates, positive_class_prior)
        if rate_integrand is not None
        else None
    )

    n_hull_points = len(true_positive_rates)
    chunk_size = max(1, _MAX_HULL_EVALUATIONS // n_hull_points)
    # The control variates need every sample's value; without them a running sum is enough.
    values = np.empty(n_samples) if controls else None
    total = 0.0
    for start in range(0, n_samples, chunk_size):
        samples = slice(start, min(start + chunk_size, n_samples))
        shape = (n_hull_points, samples.stop - samples.start)
        profits = _evaluate_terms(profit_terms, samples, shape)
        if rate_terms is None:
            sample_values = profits.max(axis=0)
        else:
            best_indices = profits.argmax(axis=0)
            rates = _evaluate_terms(rate_terms, samples, shape)
            sample_values = rates[best_indices, np.arange(shape[1])]
        if values is not None:
            values[samples] = sample_values
        else:
            total += float(sample_values.sum())
    if values is not None:
        return _control_variate_mean(values, controls)
    return total / n_samples


def _control_variate_mean(values: FloatNDArray, controls: Sequence[FloatNDArray]) -> float:
    """
    Estimate the expectation of *values* with linear control variates.

    Each control is a sampled stochastic variable minus its exact mean, so its expectation is zero.
    Regressing *values* on the controls and taking the intercept subtracts the part of the sample
    mean's error that the controls explain linearly. The maximum profit is close to linear in the
    stochastic variables, so that is most of it. The estimate is biased by O(1 / n_samples), far less
    than the sampling error it removes.

    Falls back to the sample mean when there are too few samples for the regression, or when any
    value is not finite.
    """
    design = np.column_stack([np.ones_like(values), *controls])
    if len(values) <= 2 * design.shape[1] or not (np.isfinite(values).all() and np.isfinite(design).all()):
        return float(values.mean())
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(coefficients[0])


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

    def data_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated MaxProfit and its gradient without the penalty, for solvers that apply it themselves."""
        w = np.asarray(weights, dtype=np.float64)
        value, gradient = self.logit_loss_gradient(w)
        coef = w[self._start_coef :]
        data_gradient = np.array(gradient, dtype=np.float64)
        data_gradient[self._start_coef :] -= self._regularization_gradient(coef)
        return value - self._regularization_value(coef), data_gradient

    def data_loss(self, weights: FloatNDArray) -> float:
        """Return the negated MaxProfit without the penalty."""
        return self.data_loss_gradient(weights)[0]

    def data_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return the gradient of the negated MaxProfit without the penalty."""
        return self.data_loss_gradient(weights)[1]

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
