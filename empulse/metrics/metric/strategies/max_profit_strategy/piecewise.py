import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from typing import Any, NamedTuple, Self

import numpy as np
import sympy
from sympy.stats import pspace

from ....._types import FloatNDArray, IntNDArray
from ...._cy_max_profit import Distribution, expected_max_profit, expected_max_profit_from_counts, partial_moments
from ..._compile import CountScoreFn, MetricFn, _safe_lambdify, _safe_run_lambda
from ..._distributions import spec_for
from ..._parameter_domain import _check_parameters
from ..._symbolic import _subs_by_name
from ._anchoring import Anchor, DensityOf, anchored_quad, warn_if_inaccurate
from .common import _build_rate_function, _distribution_parameter_symbols, _HullScoreFunction, _LowerChainGate
from .envelope import (
    CallableEnvelope,
    Partition,
    PolynomialEnvelope,
    ProfitEnvelope,
    partition_support,
)


def _is_polynomial_in(expression: sympy.Expr, symbol: sympy.Symbol) -> bool:
    """
    Test whether *expression* is a polynomial in *symbol*.

    This is the one place that decides between the two integration regimes: a polynomial profit
    function gets closed-form partial moments, anything else gets quadrature per region.

    ``sympy.Poly`` is the test rather than ``sympy.factor``, which raises instead of answering
    ``False`` for exactly the inputs this is asked about -- ``PolynomialError`` on ``exp(x)``,
    ``log(x)`` or ``sqrt(x)``, and ``TypeError`` from its own internals on some float coefficients.
    ``Poly`` succeeds only for non-negative integer powers of *symbol*, which is precisely the
    property that makes the ``a_k`` decomposition exist.
    """
    try:
        sympy.Poly(sympy.expand(expression), symbol)
    except (sympy.polys.polyerrors.PolynomialError, TypeError, ValueError, NotImplementedError):
        return False
    return True


def _closed_form_of(random_symbol: sympy.Symbol) -> str | None:
    """Return the distribution's name in the compiled partial moments, or ``None`` if they lack it."""
    spec = spec_for(random_symbol)
    if spec is None or spec.closed_form is None or partial_moments is None:
        return None
    return spec.closed_form


def _build_max_profit_score_piecewise(
    profit_function: sympy.Expr,
    random_symbol: sympy.Symbol,
    deterministic_symbols: Iterable[sympy.Symbol],
) -> MetricFn:
    closed_form = _closed_form_of(random_symbol)
    if closed_form is None or not _is_polynomial_in(profit_function, random_symbol):
        # Without the a_k decomposition, or the distribution's partial moments, each region is
        # integrated numerically. The regions are found the same way in both cases.
        return MaxProfitScorePiecewise(profit_function, random_symbol, deterministic_symbols)
    return ClosedFormMaxProfitScorePiecewise(profit_function, random_symbol, deterministic_symbols, closed_form)


class _PreparedIntegrand:
    """An integrand compiled once per parameter set, ready to be integrated over any region.

    Compiling means substituting everything except the rates and the stochastic variable, then
    lambdifying over those three. Doing both once per parameter set instead of once per region avoids
    a fresh ``subs`` and ``lambdify`` for each of the tens of regions in a partition, which dominated
    the runtime of the numerical path.

    The substitution has to happen before lambdifying, not be replaced by passing the parameters as
    extra arguments. A density can name a special function that collides with a symbol: the Chi
    density is ``... * gamma**(k - 1) / gamma(k/2)``, where the first ``gamma`` is the variable and
    the second is the gamma function. Substituting ``k`` collapses the call to a number and removes
    the ambiguity; passing ``k`` as an argument leaves generated source that cannot be read either
    way, whether or not the arguments are dummified.

    The integrand is the product of a ``factor`` and a density, compiled as the factor and the
    log-density, whose exponential is taken numerically (see :func:`~._anchoring.log_density`).
    """

    def __init__(self, factor: sympy.Expr, log_density: sympy.Expr, random_symbol: sympy.Symbol) -> None:
        self.integrand = sympy.Tuple(factor, log_density)
        self.random_symbol = random_symbol
        self.arguments = [*sympy.symbols('F_0 F_1'), random_symbol]
        self._compiled: tuple[tuple[tuple[str, float], ...], Callable[..., Any]] | None = None

    def _compile(self, parameters: dict[str, Any]) -> Callable[..., Any]:
        """Return the lambdified integrand for *parameters*, reusing the previous one if unchanged.

        A metric used as a fitness function is called repeatedly with the same business parameters
        and a different score vector, so in that loop this compiles exactly once. Published as one
        tuple so a concurrent reader never pairs one call's key with another call's function.
        """
        try:
            key = tuple(sorted((name, float(value)) for name, value in parameters.items()))
        except (TypeError, ValueError):
            # A non-scalar parameter cannot key the cache; compile without caching.
            return self._lambdify(parameters)
        cached = self._compiled
        if cached is not None and cached[0] == key:
            return cached[1]
        function = self._lambdify(parameters)
        self._compiled = (key, function)
        return function

    def _lambdify(self, parameters: dict[str, Any]) -> Callable[..., Any]:
        # evalf: a substituted normalisation such as loggamma(100) is otherwise left for NumPy.
        substituted = sympy.Tuple(*(_subs_by_name(part, parameters).evalf() for part in self.integrand))
        return _safe_lambdify(substituted, self.arguments)

    def integrate_regions(
        self,
        bounds: Sequence[float],
        true_positive_rates: Sequence[float],
        false_positive_rates: Sequence[float],
        parameters: dict[str, Any],
        anchor: Anchor | None,
    ) -> float:
        """
        Sum the integral of the prepared integrand over every region.

        Each region is split around ``anchor``, where the density's mass lies (see
        :func:`~._anchoring.anchored_quad`), and a large estimated error is reported once.
        """
        function = self._compile(parameters)
        total = error = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for (lower, upper), tpr, fpr in zip(
                pairwise(bounds), true_positive_rates, false_positive_rates, strict=True
            ):
                if lower == upper:
                    continue

                def integrand(x: float, tpr: float = tpr, fpr: float = fpr) -> Any:
                    return _product_with_density(function(tpr, fpr, x))

                value, estimate = anchored_quad(integrand, lower, upper, anchor)
                total += value
                error += estimate
        warn_if_inaccurate(total, error)
        return total


def _product_with_density(parts: tuple[Any, Any]) -> Any:
    """Return the integrand from its factor and log-density, see :class:`_PreparedIntegrand`."""
    factor, log_density = parts
    return factor * np.exp(log_density)


def _resolve_support(
    random_var_bounds: tuple[float | sympy.Expr, ...], distribution_parameters: dict[str, Any]
) -> tuple[float, float]:
    """Turn the distribution's symbolic support into a pair of floats, mapping ``oo`` to ``np.inf``."""
    lower_bound, upper_bound = random_var_bounds[:2]
    if isinstance(lower_bound, sympy.Expr):
        lower_bound = _subs_by_name(lower_bound, distribution_parameters)
        lower_bound = -np.inf if lower_bound == -sympy.oo else lower_bound
    if isinstance(upper_bound, sympy.Expr):
        upper_bound = _subs_by_name(upper_bound, distribution_parameters)
        upper_bound = np.inf if upper_bound == sympy.oo else upper_bound
    return float(lower_bound), float(upper_bound)


def _extract_polynomial_coefficients(
    profit_function: sympy.Expr, random_symbol: sympy.Symbol
) -> tuple[list[sympy.Expr], list[Callable[..., Any]]]:
    """
    Decompose the profit into coefficients in ascending powers of the stochastic variable.

    ``P(t, x) = sum_k a_k(F_0, F_1) x**k`` is what lets each term be integrated against a closed-form
    partial moment, and what makes the analytic gradient factor into ``grad a_k`` times ``R_k``.

    Parameters
    ----------
    profit_function : sympy.Expr
        The profit function.
    random_symbol : sympy.Symbol
        The stochastic variable to collect powers of.

    Returns
    -------
    coefficient_eqs : list of sympy.Expr
        The coefficients ``a_0, ..., a_n``.
    coefficient_fns : list of callable
        The same coefficients, lambdified.
    """
    expanded_profit = sympy.expand(profit_function)
    polynomial_degree = sympy.degree(expanded_profit, random_symbol)
    collected = sympy.collect(expanded_profit, random_symbol, evaluate=False)

    coefficient_eqs = [
        # random_symbol**0 is 1, which is the key sympy uses for the constant term.
        collected.get(random_symbol**k if k > 0 else 1, sympy.S.Zero)
        for k in range(polynomial_degree + 1)
    ]
    return coefficient_eqs, [_safe_lambdify(eq) for eq in coefficient_eqs]


def _evaluate_coefficient_matrix(
    coefficient_eqs: Sequence[sympy.Expr],
    coefficient_fns: Sequence[Callable[..., Any]],
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    positive_class_prior: float,
    negative_class_prior: float,
    parameters: dict[str, Any],
) -> FloatNDArray:
    """Evaluate every polynomial coefficient at every hull vertex, shape (n_vertices, degree + 1)."""
    columns = []
    for function, equation in zip(coefficient_fns, coefficient_eqs, strict=True):
        value = _safe_run_lambda(
            function,
            equation,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=true_positive_rates,
            F_1=false_positive_rates,
            **parameters,
        )
        # A coefficient that does not involve the rates evaluates to a scalar.
        columns.append(np.broadcast_to(np.asarray(value, dtype=np.float64), true_positive_rates.shape))
    return np.stack(columns, axis=-1)


def _callable_envelope(
    profit_eq: sympy.Expr,
    profit_fn: Callable[..., Any],
    random_name: str,
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    positive_class_prior: float,
    negative_class_prior: float,
    parameters: dict[str, Any],
    anchor: Anchor | None = None,
) -> CallableEnvelope:
    """Wrap a profit function of any shape as the family of candidate curves over the hull."""
    # Resolve which arguments the expression needs once: finding the region boundaries evaluates
    # this for every hull pair over a whole sweep, so re-deriving it per call would dominate.
    required = {str(symbol) for symbol in profit_eq.free_symbols}
    static = {
        name: value
        for name, value in {'pi_0': positive_class_prior, 'pi_1': negative_class_prior, **parameters}.items()
        if name in required
    }
    needs_tpr = 'F_0' in required
    needs_fpr = 'F_1' in required
    needs_random = random_name in required

    def profit_at(tpr: FloatNDArray | float, fpr: FloatNDArray | float, x: FloatNDArray | float) -> Any:
        arguments = dict(static)
        if needs_tpr:
            arguments['F_0'] = tpr
        if needs_fpr:
            arguments['F_1'] = fpr
        if needs_random:
            arguments[random_name] = x
        return profit_fn(**arguments)

    return CallableEnvelope(profit_at, true_positive_rates, false_positive_rates, anchor)


class _AffineCoefficients:
    """
    The profit's polynomial coefficients as affine functions of the rates, compiled to one call.

    Each coefficient is split as ``a_k = constant_k + tpr_slope_k * F_0 + fpr_slope_k * F_1``,
    where the three parts depend only on the class priors and the other parameters. Evaluating the
    parts once per call replaces evaluating every coefficient at every hull vertex, and is what the
    compiled :func:`~empulse.metrics._cy_max_profit.expected_max_profit` takes.
    """

    def __init__(self, parts: Sequence[sympy.Expr], n_powers: int) -> None:
        self.n_powers = n_powers
        expression = sympy.Tuple(*parts)
        variables = sorted(expression.free_symbols, key=str)
        self.names = [str(variable) for variable in variables]
        self.function = _safe_lambdify(expression, variables) if variables else None
        self.constant_parts = None if variables else np.array([float(part) for part in parts]).reshape(3, n_powers)

    @classmethod
    def from_coefficients(cls, coefficient_eqs: Sequence[sympy.Expr]) -> Self | None:
        """Split every coefficient, or return None if one of them is not affine in the rates."""
        tpr, fpr = sympy.symbols('F_0 F_1')
        constants, tpr_slopes, fpr_slopes = [], [], []
        for equation in coefficient_eqs:
            tpr_slope = sympy.diff(equation, tpr)
            fpr_slope = sympy.diff(equation, fpr)
            if tpr_slope.has(tpr, fpr) or fpr_slope.has(tpr, fpr):
                return None
            constants.append(equation.subs({tpr: 0, fpr: 0}))
            tpr_slopes.append(tpr_slope)
            fpr_slopes.append(fpr_slope)
        return cls([*constants, *tpr_slopes, *fpr_slopes], len(coefficient_eqs))

    def __call__(
        self, positive_class_prior: float, negative_class_prior: float, parameters: dict[str, Any]
    ) -> FloatNDArray:
        """Return the constants, TPR slopes and FPR slopes as the rows of a (3, n_powers) array."""
        if self.function is None:
            return self.constant_parts  # type: ignore[return-value]
        values = {'pi_0': positive_class_prior, 'pi_1': negative_class_prior, **parameters}
        parts = self.function(*[values[name] for name in self.names])
        return np.asarray(parts, dtype=np.float64).reshape(3, self.n_powers)


class ClosedFormExpectedMaxProfit(NamedTuple):
    """
    Everything the compiled expected maximum profit needs besides the samples, for fixed class priors.

    ``coefficient_parts`` holds the profit's constant, TPR slope and FPR slope as the rows of a
    C-contiguous ``(3, degree + 1)`` array (see :func:`~empulse.metrics._cy_max_profit.expected_max_profit`).
    """

    coefficient_parts: FloatNDArray
    lower_bound: float
    upper_bound: float
    distribution: int
    distribution_parameters: FloatNDArray


class _PiecewiseBase:
    """
    Shared plumbing for the piecewise classes: the support, the parameters and the envelope.

    Subclasses differ only in what they integrate over the regions, not in how the regions are
    found, so the envelope construction lives here.
    """

    # None when the profit is not a polynomial in the stochastic variable, in which case the
    # region boundaries are bracketed numerically through `profit_fn` instead.
    poly_eqs: list[sympy.Expr] | None
    poly_fns: list[Callable[..., Any]] | None
    profit_fn: Callable[..., Any] | None

    def _init_common(self, profit_function: sympy.Expr, random_symbol: sympy.Symbol) -> None:
        self.random_symbol = random_symbol
        self.random_var_bounds = pspace(random_symbol).domain.set.args
        self.distribution_args = pspace(random_symbol).distribution.args

        # Most supports are plain numbers; resolve those once instead of substituting on every call.
        if not any(isinstance(bound, sympy.Expr) and bound.free_symbols for bound in self.random_var_bounds):
            self._static_support: tuple[float, float] | None = _resolve_support(self.random_var_bounds, {})
        else:
            self._static_support = None
        self._support_cache: tuple[tuple[tuple[str, float], ...], tuple[float, float]] | None = None

        self.profit_eq = profit_function
        if _is_polynomial_in(profit_function, random_symbol):
            self.poly_eqs, self.poly_fns = _extract_polynomial_coefficients(profit_function, random_symbol)
            self.profit_fn = None
        else:
            # The region boundaries have to be bracketed numerically rather than read off the
            # coefficients, and each region integrated by quadrature rather than by moments.
            self.poly_eqs = None
            self.poly_fns = None
            self.profit_fn = _safe_lambdify(profit_function)

        self.dist_params = _distribution_parameter_symbols(self.distribution_args)
        self._lower_chain_gate = _LowerChainGate(profit_function, [random_symbol])
        self._density = DensityOf(random_symbol)

        # The rate on each region weighted by the region's probability: the compiled partial moment of
        # power zero, or SciPy's CDF. A distribution without either has the rate integrated numerically.
        self.rate_eq = _build_rate_function()
        self.rate_fn = _safe_lambdify(self.rate_eq)
        self._spec = spec_for(random_symbol)
        closed_form = _closed_form_of(random_symbol)
        self._rate_distribution = None if closed_form is None else int(getattr(Distribution, closed_form))
        self._prepared_rate = (
            _PreparedIntegrand(self.rate_eq, self._density.log_expression, random_symbol)
            if self._spec is None
            else None
        )

        # Resolved once because naming a sympy symbol means printing it, which costs more than
        # scoring a small hull.
        self._distribution_parameter_names = tuple(symbol.name for symbol in self.dist_params)
        self._distribution_value_sources: list[float | str | sympy.Expr] = [
            float(argument) if argument.is_number else str(argument) if isinstance(argument, sympy.Symbol) else argument
            for argument in self.distribution_args
        ]

    def _envelope(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        negative_class_prior: float,
        parameters: dict[str, Any],
        anchor: Anchor | None = None,
    ) -> ProfitEnvelope:
        """
        Build the candidate profit curves for this call's ROC hull and parameter values.

        ``anchor`` locates the density's mass, around which curves that are not polynomials are
        searched for crossings most finely.
        """
        if self.poly_eqs is not None and self.poly_fns is not None:
            return PolynomialEnvelope(
                _evaluate_coefficient_matrix(
                    self.poly_eqs,
                    self.poly_fns,
                    true_positive_rates,
                    false_positive_rates,
                    positive_class_prior,
                    negative_class_prior,
                    parameters,
                )
            )
        assert self.profit_fn is not None
        return _callable_envelope(
            self.profit_eq,
            self.profit_fn,
            str(self.random_symbol),
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            parameters,
            anchor,
        )

    def _partition(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        parameters: dict[str, Any],
        distribution_parameters: dict[str, Any],
        anchor: Anchor | None,
    ) -> Partition:
        """Split the support into the regions where each hull vertex is optimal, see ``partition_support``."""
        envelope = self._envelope(
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            1 - positive_class_prior,
            parameters,
            anchor,
        )
        return partition_support(
            envelope, true_positive_rates, false_positive_rates, *self._support(distribution_parameters)
        )

    def _anchor(self, distribution_parameters: dict[str, Any]) -> Anchor | None:
        """Return where the density's mass lies for these parameters, see :class:`~._anchoring.DensityOf`."""
        return self._density.anchor(distribution_parameters, *self._support(distribution_parameters))

    def _needs_lower_hull(self, kwargs: dict[str, Any]) -> bool:
        """Whether the lower hull can hold the most profitable vertex anywhere in the support."""
        distribution_parameters = self._resolve_distribution_parameters(kwargs)
        return bool(self._lower_chain_gate.on_support(kwargs, *self._support(distribution_parameters)))

    def _resolve_distribution_parameters(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Pick the distribution's own parameters out of the call's parameters.

        Only the parameters named by a symbol are passed in by the caller; hardcoded numeric ones
        are read from the distribution itself by :meth:`_distribution_values`. The call's
        parameters keep them too, since the profit can use a distribution parameter as well
        (``Gamma('v', a, b) * clv - a``).
        """
        return {name: kwargs[name] for name in self._distribution_parameter_names if name in kwargs}

    def _distribution_values(self, distribution_parameters: dict[str, Any]) -> list[float]:
        """Every parameter of the distribution, in the order its constructor takes them.

        The closed-form integrals read the parameters by position. Reading them off the caller's
        values instead would drop the hardcoded ones (``Beta('g', 6, b)`` would see only ``b``), or
        merge equal ones (``Beta('g', 6, 6)``), so each argument is resolved on its own.
        """
        values = []
        for source in self._distribution_value_sources:
            if isinstance(source, float):
                values.append(source)
            elif isinstance(source, str):
                values.append(float(distribution_parameters[source]))
            else:
                values.append(float(_subs_by_name(source, distribution_parameters)))
        return values

    def _support(self, distribution_parameters: dict[str, Any]) -> tuple[float, float]:
        """Resolve the support, reusing the previous answer when the parameters have not changed.

        Published as one tuple so a concurrent reader sees a key and a support that belong
        together, never a key from one parameter set beside a support from another.
        """
        if self._static_support is not None:
            return self._static_support
        key = tuple(sorted((name, float(value)) for name, value in distribution_parameters.items()))
        cached = self._support_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        support = _resolve_support(self.random_var_bounds, distribution_parameters)
        self._support_cache = (key, support)
        return support

    def _rate_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """
        Compute the optimal rate from the ROC convex hull and the positive class prior.

        The rate does not depend on the stochastic variable, only on which vertex is optimal, so the
        expected rate is the rate on each region weighted by the probability of that region, whatever
        the shape of the profit function.
        """
        negative_class_prior = 1 - positive_class_prior
        distribution_parameters = self._resolve_distribution_parameters(kwargs)
        # Only a profit that is not a polynomial needs the anchor to search for its crossings, and only
        # a numeric integration needs it to integrate.
        needs_anchor = self.poly_eqs is None or self._prepared_rate is not None
        anchor = self._anchor(distribution_parameters) if needs_anchor else None
        partition = self._partition(
            true_positive_rates, false_positive_rates, positive_class_prior, kwargs, distribution_parameters, anchor
        )
        if self._prepared_rate is not None:
            return self._prepared_rate.integrate_regions(
                partition.bounds,
                partition.tprs,
                partition.fprs,
                {**kwargs, **distribution_parameters, 'pi_0': positive_class_prior, 'pi_1': negative_class_prior},
                anchor,
            )

        rate = _safe_run_lambda(
            self.rate_fn,
            self.rate_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=np.array(partition.tprs),
            F_1=np.array(partition.fprs),
            **kwargs,
        )
        bounds = [float(bound) for bound in partition.bounds]
        values = self._distribution_values(distribution_parameters)
        if self._rate_distribution is not None:
            probabilities = partial_moments(  # type: ignore[misc]
                np.ascontiguousarray(bounds, dtype=np.float64),
                1,
                *self._support(distribution_parameters),
                self._rate_distribution,
                np.asarray(values, dtype=np.float64),
            )[0]
        else:
            assert self._spec is not None
            probabilities = np.diff(self._spec.freeze(values).cdf(bounds))
        return float((rate * probabilities).sum())


class MaxProfitScorePiecewise(_PiecewiseBase, _HullScoreFunction):
    """
    Compute the maximum profit for a single stochastic variable using piecewise integration.

    The support is split into the regions where a single ROC convex-hull vertex maximises profit,
    and the profit is integrated numerically over each one. Used when the distribution has no
    closed-form partial moments, or when the profit function is not a polynomial in the stochastic
    variable. Splitting first matters: ``max_t P(t, x)`` is non-smooth at every region boundary,
    which is exactly what makes a single global quadrature slow and inaccurate.
    """

    def __init__(
        self, profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self._init_common(profit_function, random_symbol)
        self._prepared = _PreparedIntegrand(profit_function, self._density.log_expression, random_symbol)

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        negative_class_prior = 1 - positive_class_prior

        distribution_parameters = self._resolve_distribution_parameters(kwargs)
        anchor = self._anchor(distribution_parameters)

        partition = self._partition(
            true_positive_rates, false_positive_rates, positive_class_prior, kwargs, distribution_parameters, anchor
        )

        return self._prepared.integrate_regions(
            partition.bounds,
            partition.tprs,
            partition.fprs,
            {**kwargs, **distribution_parameters, 'pi_0': positive_class_prior, 'pi_1': negative_class_prior},
            anchor,
        )


#: The closed forms whose stochastic variable is strictly positive. The analytic gradients of the logit
#: and boosting objectives are derived for these.
_POSITIVE_CLOSED_FORMS = frozenset({'GAMMA', 'PARETO', 'EXPONENTIAL', 'CHI_SQUARED', 'LOG_NORMAL', 'BETA', 'WEIBULL'})


class ClosedFormMaxProfitScorePiecewise(_PiecewiseBase, _HullScoreFunction):
    """
    Compute the maximum profit for a single stochastic variable with closed-form partial moments.

    The support is split into the regions where a single ROC convex-hull vertex maximises profit.
    On each region the profit is a polynomial in the stochastic variable, so every term integrates
    to a coefficient times a partial moment of the distribution, which
    :func:`~empulse.metrics._cy_max_profit.partial_moments` computes in closed form.

    A profit of degree at most two, whose region edges have closed-form roots, is scored entirely by
    the compiled :func:`~empulse.metrics._cy_max_profit.expected_max_profit`. Any other is
    partitioned here and integrated with the same compiled partial moments.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        random_symbol: sympy.Symbol,
        deterministic_symbols: Iterable[sympy.Symbol],
        closed_form: str,
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self._init_common(profit_function, random_symbol)
        if self.poly_eqs is None or self.poly_fns is None:
            raise ValueError(
                'The exact piecewise integration method requires a profit function that is a '
                f'polynomial in {random_symbol}. Use a MaxProfit() integration method other than '
                "'auto', or build the metric with a polynomial profit function."
            )
        # Narrowed from Optional: the check above proves the decomposition exists.
        self.coefficient_eqs: list[sympy.Expr] = self.poly_eqs
        self.coefficient_fns: list[Callable[..., Any]] = self.poly_fns
        #: The name of the distribution in :class:`~empulse.metrics._cy_max_profit.Distribution`.
        self.closed_form = closed_form
        self._distribution = int(getattr(Distribution, closed_form))
        self._compiled = _AffineCoefficients.from_coefficients(self.poly_eqs) if len(self.poly_eqs) <= 3 else None
        if closed_form in _POSITIVE_CLOSED_FORMS:
            self.gradient = 'piecewise'

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        negative_class_prior = 1 - positive_class_prior

        distribution_parameters = self._resolve_distribution_parameters(kwargs)

        if self._compiled is not None:
            constant, tpr_slope, fpr_slope = self._compiled(positive_class_prior, negative_class_prior, kwargs)
            return float(
                expected_max_profit(  # type: ignore[misc]
                    np.asarray(true_positive_rates, dtype=np.float64),
                    np.asarray(false_positive_rates, dtype=np.float64),
                    constant,
                    tpr_slope,
                    fpr_slope,
                    *self._support(distribution_parameters),
                    self._distribution,
                    np.asarray(self._distribution_values(distribution_parameters), dtype=np.float64),
                )
            )

        # The envelope needs the coefficients at every hull vertex; each region then selects one vertex.
        coefficient_matrix = _evaluate_coefficient_matrix(
            self.coefficient_eqs,
            self.coefficient_fns,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            kwargs,
        )
        partition = partition_support(
            PolynomialEnvelope(coefficient_matrix),
            true_positive_rates,
            false_positive_rates,
            *self._support(distribution_parameters),
        )
        segment_coefficients = list(coefficient_matrix[partition.vertex_indices].T)

        return self._integrate(
            bounds=[float(bound) for bound in partition.bounds],
            coefficients=segment_coefficients,
            distribution_parameters=distribution_parameters,
        )

    def _count_scorer(self, **kwargs: Any) -> CountScoreFn:
        """
        Prepare the score of samples grouped by score, for parameter values fixed across many calls.

        With the compiled score, everything that depends only on the parameters is resolved here,
        and the profit's coefficients are reused for as long as the class prior stays the same (as
        it does for every tree a ProfTree evaluates), so a call is a single compiled function call.
        """
        if self._compiled is None or self._needs_lower_hull(dict(kwargs)):
            # The compiled score builds the upper hull only.
            return super()._count_scorer(**kwargs)
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)
        affine, parameters, lower_bound, upper_bound, distribution_values = self._resolve_compiled(kwargs)
        distribution = self._distribution
        last_parts: tuple[int, int, FloatNDArray] | None = None

        def coefficient_parts(n_positives: int, n_negatives: int) -> FloatNDArray:
            nonlocal last_parts
            cached = last_parts
            if cached is not None and cached[0] == n_positives and cached[1] == n_negatives:
                return cached[2]
            positive_class_prior = n_positives / (n_positives + n_negatives)
            parts = np.ascontiguousarray(affine(positive_class_prior, 1 - positive_class_prior, parameters))
            last_parts = (n_positives, n_negatives, parts)
            return parts

        def score(y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray) -> float:
            return expected_max_profit_from_counts(  # type: ignore[no-any-return,misc]
                np.asarray(y_score, dtype=np.float64),
                np.asarray(n_positive, dtype=np.int64),
                np.asarray(n_negative, dtype=np.int64),
                coefficient_parts,
                lower_bound,
                upper_bound,
                distribution,
                distribution_values,
            )

        return score

    def _closed_form(self, n_positives: int, n_negatives: int, **kwargs: Any) -> ClosedFormExpectedMaxProfit | None:
        """
        Return what the compiled score needs for samples with these class counts, or None without it.

        For a caller that scores many groupings of the same samples in compiled code, such as the
        trees of a ProfTree population, whose leaves always hold every training sample.
        """
        if self._compiled is None or self._needs_lower_hull(dict(kwargs)):
            # The compiled score builds the upper hull only.
            return None
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)
        affine, parameters, lower_bound, upper_bound, distribution_values = self._resolve_compiled(kwargs)
        positive_class_prior = n_positives / (n_positives + n_negatives)
        return ClosedFormExpectedMaxProfit(
            np.ascontiguousarray(affine(positive_class_prior, 1 - positive_class_prior, parameters)),
            lower_bound,
            upper_bound,
            self._distribution,
            distribution_values,
        )

    def _resolve_compiled(
        self, kwargs: dict[str, Any]
    ) -> tuple['_AffineCoefficients', dict[str, Any], float, float, FloatNDArray]:
        """Resolve the parts of the compiled score that depend only on the parameters."""
        assert self._compiled is not None
        parameters = dict(kwargs)
        distribution_parameters = self._resolve_distribution_parameters(parameters)
        lower_bound, upper_bound = self._support(distribution_parameters)
        distribution_values = np.asarray(self._distribution_values(distribution_parameters), dtype=np.float64)
        return self._compiled, parameters, lower_bound, upper_bound, distribution_values

    def _partial_moments(
        self, bounds: Sequence[float] | FloatNDArray, n_powers: int, distribution_parameters: dict[str, Any]
    ) -> FloatNDArray:
        """Return the integral of ``x**k`` times the density over each region, at ``[k, region]``."""
        return partial_moments(  # type: ignore[no-any-return,misc]
            np.ascontiguousarray(bounds, dtype=np.float64),
            n_powers,
            *self._support(distribution_parameters),
            self._distribution,
            np.asarray(self._distribution_values(distribution_parameters), dtype=np.float64),
        )

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[FloatNDArray],
        distribution_parameters: dict[str, Any],
    ) -> float:
        """Integrate the profit, with these coefficients of each power on each region, over the regions."""
        moments = self._partial_moments(bounds, len(coefficients), distribution_parameters)
        total_score = 0.0
        for k, a_k in enumerate(coefficients):
            # A power the profit does not have contributes nothing, also where its moment is not finite.
            if np.all(np.asarray(a_k) == 0.0):
                continue
            total_score += (a_k * moments[k]).sum()
        return float(total_score)
