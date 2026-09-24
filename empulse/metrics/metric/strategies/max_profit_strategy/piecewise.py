import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from typing import Any, ClassVar, Self

import numpy as np
import scipy.special as sp
import scipy.stats as st
import sympy
from scipy.integrate import IntegrationWarning, quad
from sympy.stats import density, pspace

from ....._types import FloatNDArray, IntNDArray
from ...._cy_max_profit import Distribution, expected_max_profit
from ..._compile import MetricFn, RateFn, _safe_lambdify, _safe_run_lambda
from ..._parameter_domain import _check_parameters
from ..._symbolic import _subs_by_name
from ._distributions import ADAPTERS, adapter_for
from .common import _convex_hull, _distribution_parameter_symbols, _HullScoreFunction
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


def _lookup_by_distribution_type(table: dict[type, Any], distribution: Any) -> Any | None:
    """Look *distribution* up in *table*, by exact type first, then by ``isinstance``.

    The ``isinstance`` fallback matches a subclass of a registered distribution that
    ``sympy.stats`` does not register in the table verbatim -- the same thing a plain
    ``isinstance(distribution, X)`` chain would do, which this replaces at every call site.
    """
    match = table.get(type(distribution))
    if match is not None:
        return match
    for distribution_type, candidate in table.items():
        if isinstance(distribution, distribution_type):
            return candidate
    return None


def _build_max_profit_score_piecewise(
    profit_function: sympy.Expr,
    random_symbol: sympy.Symbol,
    deterministic_symbols: Iterable[sympy.Symbol],
) -> MetricFn:
    if not _is_polynomial_in(profit_function, random_symbol):
        # No a_k decomposition, so no closed-form partial moments; each region is integrated
        # numerically instead. The regions themselves are found the same way either way.
        return MaxProfitScorePiecewise(profit_function, random_symbol, deterministic_symbols)

    distribution = pspace(random_symbol).distribution
    score_class = _lookup_by_distribution_type(_SCORE_CLASSES, distribution) or MaxProfitScorePiecewise
    return score_class(profit_function, random_symbol, deterministic_symbols)


def _build_max_profit_rate_piecewise(
    profit_function: sympy.Expr,
    rate_function: sympy.Expr,
    random_symbol: sympy.Symbol,
    deterministic_symbols: Iterable[sympy.Symbol],
) -> RateFn:
    adapter = adapter_for(random_symbol)
    if adapter is None:
        return MaxProfitRatePiecewise(profit_function, rate_function, random_symbol, deterministic_symbols)
    return ExactMaxProfitRatePiecewise(
        profit_function, rate_function, random_symbol, deterministic_symbols, adapter.scipy_dist, adapter.params
    )


class _PreparedIntegrand:
    """An integrand compiled once per parameter set, ready to be integrated over any region.

    Compiling means substituting everything except the rates and the stochastic variable, then
    lambdifying over those three. Both steps are kept out of the region loop: doing them per region
    -- as this code used to -- meant a fresh ``subs`` and a fresh ``lambdify`` for every one of the
    tens of regions in a partition, which dominated the runtime of the numerical path.

    The substitution has to happen before lambdifying, not be replaced by passing the parameters as
    extra arguments. A density can name a special function that collides with a symbol: the Chi
    density is ``... * gamma**(k - 1) / gamma(k/2)``, where the first ``gamma`` is the variable and
    the second is the gamma function. Substituting ``k`` collapses the call to a number and removes
    the ambiguity; passing ``k`` as an argument leaves generated source that cannot be read either
    way, whether or not the arguments are dummified.
    """

    def __init__(self, integrand: sympy.Expr, random_symbol: sympy.Symbol) -> None:
        self.integrand = integrand
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
            return _safe_lambdify(_subs_by_name(self.integrand, parameters), self.arguments)
        cached = self._compiled
        if cached is not None and cached[0] == key:
            return cached[1]
        function = _safe_lambdify(_subs_by_name(self.integrand, parameters), self.arguments)
        self._compiled = (key, function)
        return function

    def integrate_regions(
        self,
        bounds: Sequence[float],
        true_positive_rates: Sequence[float],
        false_positive_rates: Sequence[float],
        parameters: dict[str, Any],
    ) -> float:
        """Sum the integral of the prepared integrand over every region."""
        function = self._compile(parameters)
        total = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationWarning)
            warnings.simplefilter('ignore', RuntimeWarning)
            for (lower, upper), tpr, fpr in zip(
                pairwise(bounds), true_positive_rates, false_positive_rates, strict=True
            ):
                if lower == upper:
                    continue
                total += quad(lambda x, tpr=tpr, fpr=fpr: function(tpr, fpr, x), lower, upper)[0]
        return total


def _resolve_support(
    random_var_bounds: tuple[float | sympy.Expr, ...],
    distribution_parameters: dict[str, Any],
    lower_bound: float | None,
    upper_bound: float | None,
) -> tuple[float, float]:
    """Turn the distribution's symbolic support into a pair of floats, mapping ``oo`` to ``np.inf``."""
    if upper_bound is None:
        bound = random_var_bounds[1]
        if isinstance(bound, sympy.Expr):
            bound = _subs_by_name(bound, distribution_parameters)
            upper_bound = np.inf if bound == sympy.oo else float(bound)
        else:
            upper_bound = float(bound)
    if lower_bound is None:
        bound = random_var_bounds[0]
        if isinstance(bound, sympy.Expr):
            bound = _subs_by_name(bound, distribution_parameters)
            lower_bound = -np.inf if bound == -sympy.oo else float(bound)
        else:
            lower_bound = float(bound)
    return float(lower_bound), float(upper_bound)


def compute_piecewise_bounds(
    envelope: ProfitEnvelope,
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    random_var_bounds: tuple[float | sympy.Expr, ...],
    distribution_parameters: dict[str, Any],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> Partition:
    """
    Split the stochastic variable's support into the regions where each hull vertex is optimal.

    Parameters
    ----------
    envelope : ProfitEnvelope
        The candidate profit curves, one per ROC convex-hull vertex.
    true_positive_rates : NDArray of shape (n_vertices,)
        True positive rates of the ROC convex hull.
    false_positive_rates : NDArray of shape (n_vertices,)
        False positive rates of the ROC convex hull.
    random_var_bounds : tuple
        The distribution's support, possibly still symbolic.
    distribution_parameters : dict
        Values for any free symbols appearing in `random_var_bounds`.
    lower_bound : float, optional
        Precomputed lower end of the support, to skip the sympy substitution on a hot path.
    upper_bound : float, optional
        Precomputed upper end of the support, to skip the sympy substitution on a hot path.

    Returns
    -------
    partition : Partition
        Region edges together with the optimal hull vertex on each region.
    """
    lower, upper = _resolve_support(random_var_bounds, distribution_parameters, lower_bound, upper_bound)
    return partition_support(envelope, true_positive_rates, false_positive_rates, lower, upper)


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

    return CallableEnvelope(profit_at, true_positive_rates, false_positive_rates)


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

        # Most supports are plain numbers, so resolving them symbolically on every call costs a
        # sympy substitution for no gain. Resolve once here when nothing in them is symbolic.
        if not any(isinstance(bound, sympy.Expr) and bound.free_symbols for bound in self.random_var_bounds):
            self._static_support: tuple[float, float] | None = _resolve_support(self.random_var_bounds, {}, None, None)
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

        # Resolved once here, since every call needs them: naming a sympy symbol means printing it,
        # which cost more than the rest of scoring a small hull.
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
    ) -> ProfitEnvelope:
        """Build the candidate profit curves for this call's ROC hull and parameter values."""
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
        )

    def _resolve_distribution_parameters(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split the call's parameters into the distribution's own and everything else.

        Only the parameters named by a symbol are passed in by the caller; hardcoded numeric ones
        are read from the distribution itself by :meth:`_distribution_values`.
        """
        distribution_parameters = {
            name: kwargs.pop(name) for name in self._distribution_parameter_names if name in kwargs
        }
        return distribution_parameters, kwargs

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
        support = _resolve_support(self.random_var_bounds, distribution_parameters, None, None)
        self._support_cache = (key, support)
        return support


class MaxProfitRatePiecewise(_PiecewiseBase):
    """
    Compute the maximum profit rate for a single stochastic variable using piecewise integration.

    The support is split into the regions where a single ROC convex-hull vertex maximises profit,
    and the rate is integrated numerically over each one.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr,
        random_symbol: sympy.Symbol,
        deterministic_symbols: Iterable[sympy.Symbol],
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self._init_common(profit_function, random_symbol)
        self.integrand = rate_function * density(random_symbol).pdf(random_symbol)
        self._prepared = _PreparedIntegrand(self.integrand, random_symbol)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the optimal rate."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        distribution_parameters, kwargs = self._resolve_distribution_parameters(kwargs)

        partition = compute_piecewise_bounds(
            self._envelope(
                true_positive_rates, false_positive_rates, positive_class_prior, negative_class_prior, kwargs
            ),
            true_positive_rates,
            false_positive_rates,
            self.random_var_bounds,
            distribution_parameters,
            *self._support(distribution_parameters),
        )

        return self._prepared.integrate_regions(
            partition.bounds,
            partition.tprs,
            partition.fprs,
            {**kwargs, **distribution_parameters, 'pi_0': positive_class_prior, 'pi_1': negative_class_prior},
        )


class ExactMaxProfitRatePiecewise(_PiecewiseBase):
    """
    Compute the optimal rate exactly for a single stochastic variable using piecewise integration.

    The rate itself does not depend on the stochastic variable -- only on which vertex is optimal --
    so the expected rate is the rate on each region weighted by the probability of that region. That
    is a difference of the distribution's own CDF, whatever the shape of the profit function.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr,
        random_symbol: sympy.Symbol,
        deterministic_symbols: Iterable[sympy.Symbol],
        cdf_function: st.rv_continuous,
        sympy_to_scipy_params_fn: Callable[..., dict[str, float]],
    ) -> None:
        self.cdf_function = cdf_function
        self.sympy_to_scipy_params_fn = sympy_to_scipy_params_fn
        self.deterministic_symbols = deterministic_symbols
        self._init_common(profit_function, random_symbol)

        self.rate_eq = rate_function
        self.rate_fn = _safe_lambdify(self.rate_eq)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit rate."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        distribution_parameters, kwargs = self._resolve_distribution_parameters(kwargs)

        partition = compute_piecewise_bounds(
            self._envelope(
                true_positive_rates, false_positive_rates, positive_class_prior, negative_class_prior, kwargs
            ),
            true_positive_rates,
            false_positive_rates,
            self.random_var_bounds,
            distribution_parameters,
            *self._support(distribution_parameters),
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

        return self._integrate(
            bounds=[float(bound) for bound in partition.bounds],
            rate=rate,
            distribution_parameters=distribution_parameters,
        )

    def _integrate(
        self,
        bounds: list[float],
        rate: FloatNDArray,
        distribution_parameters: dict[str, Any],
    ) -> float:
        """Integrate distribution-specific piecewise math."""
        cdf_diff = np.diff(
            self.cdf_function.cdf(
                bounds, **self.sympy_to_scipy_params_fn(*self._distribution_values(distribution_parameters))
            )
        )
        optimal_rate = (rate * cdf_diff).sum()
        return float(optimal_rate)


class MaxProfitScorePiecewise(_HullScoreFunction, _PiecewiseBase):
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
        self.integrand = profit_function * density(random_symbol).pdf(random_symbol)
        self._prepared = _PreparedIntegrand(self.integrand, random_symbol)

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        negative_class_prior = 1 - positive_class_prior

        distribution_parameters, kwargs = self._resolve_distribution_parameters(kwargs)

        partition = compute_piecewise_bounds(
            self._envelope(
                true_positive_rates, false_positive_rates, positive_class_prior, negative_class_prior, kwargs
            ),
            true_positive_rates,
            false_positive_rates,
            self.random_var_bounds,
            distribution_parameters,
            *self._support(distribution_parameters),
        )

        return self._prepared.integrate_regions(
            partition.bounds,
            partition.tprs,
            partition.fprs,
            {**kwargs, **distribution_parameters, 'pi_0': positive_class_prior, 'pi_1': negative_class_prior},
        )


class BaseMaxProfitScorePiecewise(_HullScoreFunction, _PiecewiseBase):
    """
    Base class to compute the maximum profit for a single stochastic variable using piecewise integration.

    The support is split into the regions where a single ROC convex-hull vertex maximises profit.
    On each region the profit is a polynomial in the stochastic variable, so every term integrates
    to a coefficient times a closed-form partial moment that the subclass supplies.

    Subclasses that name their distribution in ``_compiled_distribution`` are scored by the compiled
    :func:`~empulse.metrics._cy_max_profit.expected_max_profit` instead, for profits up to degree two
    in the stochastic variable; their ``_integrate`` remains the reference it is tested against.
    """

    #: The name of the distribution in :class:`~empulse.metrics._cy_max_profit.Distribution`, if the
    #: compiled score covers it.
    _compiled_distribution: ClassVar[str | None] = None

    def __init__(
        self, profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self._init_common(profit_function, random_symbol)
        if self.poly_eqs is None or self.poly_fns is None:
            raise ValueError(
                'The exact piecewise integration method requires a profit function that is a '
                f'polynomial in {random_symbol}. Use a MaxProfit() integration method other than '
                "'auto', or build the metric with a polynomial profit function."
            )
        # Republished without the Optional: reaching here proves the decomposition exists, and the
        # gradient objectives differentiate these directly.
        self.coefficient_eqs: list[sympy.Expr] = self.poly_eqs
        self.coefficient_fns: list[Callable[..., Any]] = self.poly_fns

        self._compiled: tuple[_AffineCoefficients, int] | None = None
        if expected_max_profit is not None and self._compiled_distribution is not None and len(self.poly_eqs) <= 3:
            affine = _AffineCoefficients.from_coefficients(self.poly_eqs)
            if affine is not None:
                self._compiled = (affine, int(getattr(Distribution, self._compiled_distribution)))

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        negative_class_prior = 1 - positive_class_prior

        distribution_parameters, kwargs = self._resolve_distribution_parameters(kwargs)

        if self._compiled is not None:
            affine, distribution = self._compiled
            constant, tpr_slope, fpr_slope = affine(positive_class_prior, negative_class_prior, kwargs)
            return float(
                expected_max_profit(  # type: ignore[misc]
                    np.asarray(true_positive_rates, dtype=np.float64),
                    np.asarray(false_positive_rates, dtype=np.float64),
                    constant,
                    tpr_slope,
                    fpr_slope,
                    *self._support(distribution_parameters),
                    distribution,
                    np.asarray(self._distribution_values(distribution_parameters), dtype=np.float64),
                )
            )

        # The coefficients are needed at every hull vertex to build the envelope, and the regions
        # then select which vertex's coefficients apply where.
        coefficient_matrix = _evaluate_coefficient_matrix(
            self.coefficient_eqs,
            self.coefficient_fns,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            kwargs,
        )
        partition = compute_piecewise_bounds(
            PolynomialEnvelope(coefficient_matrix),
            true_positive_rates,
            false_positive_rates,
            self.random_var_bounds,
            distribution_parameters,
            *self._support(distribution_parameters),
        )
        segment_coefficients = list(coefficient_matrix[partition.vertex_indices].T)

        return self._integrate(
            bounds=[float(bound) for bound in partition.bounds],
            coefficients=segment_coefficients,
            distribution_parameters=distribution_parameters,
            upper_bound=partition.upper_bound,
            lower_bound=partition.lower_bound,
        )

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[FloatNDArray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        """Integrate distribution-specific piecewise math."""
        raise NotImplementedError('Subclasses must implement the `_integrate` method.')


class MaxProfitScorePiecewiseUniform(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single uniform distributed variable using piecewise polynomial integration."""

    _compiled_distribution = 'UNIFORM'

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[np.ndarray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:

        lower_bounds = np.asarray(bounds[:-1])
        upper_bounds = np.asarray(bounds[1:])

        # Max and min of the Uniform distribution support
        max_val = float(upper_bound)
        min_val = float(lower_bound)

        # The constant PDF of the Uniform distribution: 1 / (beta - alpha)
        pdf_val = 1.0 / (max_val - min_val)

        total_score = 0.0

        # Iterate over each term in the polynomial: a_k * x^k
        for k, a_k in enumerate(coefficients):
            # Skip zero coefficients to save computation
            if np.all(np.asarray(a_k) == 0.0):
                continue

            # Integral of a_k * x^k * h(x)
            term_val = (a_k * pdf_val / (k + 1.0)) * (upper_bounds ** (k + 1) - lower_bounds ** (k + 1))

            total_score += term_val.sum()

        return float(total_score)


def _safe_pow_phi(x: np.ndarray, exp: int, phi: float | np.ndarray) -> np.ndarray:
    """Compute x^exp * phi(x), treating inf^exp * 0 as 0."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = (x**exp) * phi
    # Where phi is 0 (i.e., x is ±inf), the product should be 0, not NaN
    result[np.asarray(phi) == 0] = 0.0
    return result


class MaxProfitScorePiecewiseNormal(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single normal distributed variable using piecewise polynomial integration."""

    _compiled_distribution = 'NORMAL'

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[np.ndarray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = self._distribution_values(distribution_parameters)
        mu = float(dist_params_list[0])
        sigma = float(dist_params_list[1])

        lower_bounds = np.asarray(bounds[:-1])
        upper_bounds = np.asarray(bounds[1:])

        z_lower = (lower_bounds - mu) / sigma
        z_upper = (upper_bounds - mu) / sigma

        phi_lower = st.norm.pdf(z_lower)
        phi_upper = st.norm.pdf(z_upper)

        # R will store the integrated value of x^k for the segment
        r = []

        # R_0: The basic integral of the PDF (the CDF difference)
        r0 = st.norm.cdf(z_upper) - st.norm.cdf(z_lower)
        r.append(r0)

        max_degree = len(coefficients) - 1

        if max_degree >= 1:
            # R_1: mu * R_0 - sigma * (phi(z_d) - phi(z_c))
            e0 = phi_upper - phi_lower
            r1 = mu * r0 - sigma * e0
            r.append(r1)

        # R_k: Recurrence relation for k >= 2
        for k in range(2, max_degree + 1):
            r_k_minus_1 = _safe_pow_phi(upper_bounds, k - 1, phi_upper) - _safe_pow_phi(lower_bounds, k - 1, phi_lower)
            r_k = mu * r[-1] + (k - 1.0) * (sigma**2) * r[-2] - sigma * r_k_minus_1
            r.append(r_k)

        total_score = 0.0

        # Multiply each raw integral R_k by its polynomial coefficient a_k
        for k, a_k in enumerate(coefficients):
            if np.all(np.asarray(a_k) == 0.0):
                continue
            total_score += (a_k * r[k]).sum()

        return float(total_score)


class BasePositiveDistribution(BaseMaxProfitScorePiecewise):
    """
    Intermediate base class for continuous, strictly positive distributions.

    Leverages the k-th order size-biased distribution property for polynomials:
    integral(x^k * h(x)) = E[x^k] * shifted_cdf_k_diff.
    """

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[FloatNDArray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        total_score = 0.0

        # Iterate over each term in the polynomial: a_k * x^k
        for k, a_k in enumerate(coefficients):
            # Fetch the k-th moment and the CDF shifted by degree k
            kth_moment, shifted_cdf_k_diff = self._get_kth_integration_components(bounds, k, distribution_parameters)

            # Add the evaluated term to the total integral sum
            total_score += (a_k * kth_moment * shifted_cdf_k_diff).sum()

        return float(total_score)

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        """Return (kth_moment, shifted_cdf_k_diff) for a specific degree k."""
        raise NotImplementedError('Subclasses must implement `_get_kth_integration_components`.')


class MaxProfitScorePiecewiseGamma(BasePositiveDistribution):
    """Compute the maximum profit for a single gamma distributed variable using piecewise integration."""

    _compiled_distribution = 'GAMMA'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        alpha = float(dist_params_list[0])
        lambda_rate = float(dist_params_list[1])

        shifted_cdf_k_diff = np.diff(st.gamma.cdf(bounds, a=alpha + k, scale=lambda_rate))
        kth_moment = 1.0 if k == 0 else (lambda_rate**k) * (sp.gamma(alpha + k) / sp.gamma(alpha))

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewisePareto(BasePositiveDistribution):
    """Compute the maximum profit for a single triangular distributed variable using piecewise integration."""

    _compiled_distribution = 'PARETO'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        x_m = float(dist_params_list[0])  # Scale
        alpha = float(dist_params_list[1])  # Shape

        # Guardrail: The k-th moment only exists if alpha > k
        if alpha <= k:
            raise ValueError(
                f'The Pareto shape parameter (alpha={alpha}) must be strictly greater than degree k={k} '
                f'for the moment to exist.'
            )

        # H(x; x_m, alpha - k)
        shifted_cdf_k_diff = np.diff(st.pareto.cdf(bounds, b=alpha - k, scale=x_m))
        # E[x^k] = (alpha * x_m^k) / (alpha - k)
        kth_moment = 1.0 if k == 0 else (alpha * (x_m**k)) / (alpha - float(k))

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseTriangular(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single triangular distributed variable using piecewise integration."""

    _compiled_distribution = 'TRIANGULAR'

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[np.ndarray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = self._distribution_values(distribution_parameters)

        a = float(dist_params_list[0])
        b = float(dist_params_list[1])
        m = float(dist_params_list[2])

        loc = a
        scale = b - a
        c_shape = (m - a) / scale

        bounds_arr = np.clip(np.asarray(bounds), a, b)

        # Pre-calculate masks for the two segments of the triangle
        mask1 = bounds_arr <= m
        mask2 = bounds_arr > m

        total_score = 0.0

        for k, a_k in enumerate(coefficients):
            # Skip zero coefficients
            if np.all(np.asarray(a_k) == 0.0):
                continue

            if k == 0:
                # k = 0 is just the standard CDF difference
                expected_diff = np.diff(st.triang.cdf(bounds_arr, c=c_shape, loc=loc, scale=scale))
                total_score += (a_k * expected_diff).sum()
                continue

            # For k >= 1, use the exact polynomial integral A_k(x)
            a_x = np.zeros_like(bounds_arr)

            # 1. Evaluate bounds falling in the first segment: [a, m]
            if np.any(mask1) and m > a:
                x1 = bounds_arr[mask1]
                a_x[mask1] = (2.0 / ((b - a) * (m - a))) * (
                    (x1 ** (k + 2)) / (k + 2.0)
                    - (a * x1 ** (k + 1)) / (k + 1.0)
                    + (a ** (k + 2)) / ((k + 1.0) * (k + 2.0))
                )

            # 2. Evaluate bounds falling in the second segment: (m, b]
            if np.any(mask2) and b > m:
                x2 = bounds_arr[mask2]

                # A_k(m) is the integral up to the mode
                a_m = 0.0
                if m > a:
                    a_m = (2.0 / ((b - a) * (m - a))) * (
                        (m ** (k + 2)) / (k + 2.0)
                        - (a * m ** (k + 1)) / (k + 1.0)
                        + (a ** (k + 2)) / ((k + 1.0) * (k + 2.0))
                    )

                a_x[mask2] = a_m + (2.0 / ((b - a) * (b - m))) * (
                    b * (x2 ** (k + 1) - m ** (k + 1)) / (k + 1.0) - (x2 ** (k + 2) - m ** (k + 2)) / (k + 2.0)
                )

            # The expected value integral between bounds d and c is A_k(d) - A_k(c)
            expected_diff = np.diff(a_x)
            total_score += (a_k * expected_diff).sum()

        return float(total_score)


class MaxProfitScorePiecewiseExponential(BasePositiveDistribution):
    """Compute the maximum profit for a single exponential distributed variable using piecewise integration."""

    _compiled_distribution = 'EXPONENTIAL'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        lambda_rate = float(dist_params_list[0])
        scale = 1.0 / lambda_rate

        if k == 0:
            shifted_cdf_k_diff = np.diff(st.expon.cdf(bounds, scale=scale))
            kth_moment = 1.0
        else:
            # Shifted CDF: H_k*(x) evaluates to a Gamma CDF with shape = 1 + k
            shifted_cdf_k_diff = np.diff(st.gamma.cdf(bounds, a=1.0 + k, scale=scale))
            # E[x^k] = k! / lambda^k
            kth_moment = sp.gamma(1.0 + k) * (scale**k)

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseChi2(BasePositiveDistribution):
    """Compute the maximum profit for a single chi-squared distributed variable using piecewise integration."""

    _compiled_distribution = 'CHI_SQUARED'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        df = float(dist_params_list[0])

        # Shifted CDF: adding 2*k to degrees of freedom
        shifted_cdf_k_diff = np.diff(st.chi2.cdf(bounds, df=df + 2.0 * k))

        kth_moment = 1.0 if k == 0 else (2.0**k) * sp.gamma(k + df / 2.0) / sp.gamma(df / 2.0)

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseLogNormal(BasePositiveDistribution):
    """Compute the maximum profit for a single log normal distributed variable using piecewise integration."""

    _compiled_distribution = 'LOG_NORMAL'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        mu = float(dist_params_list[0])
        sigma = float(dist_params_list[1])

        # Shifted CDF: scale shifts to exp(mu + k * sigma^2)
        shifted_scale = np.exp(mu + k * sigma**2)
        shifted_cdf_k_diff = np.diff(st.lognorm.cdf(bounds, s=sigma, scale=shifted_scale))

        kth_moment = 1.0 if k == 0 else np.exp(k * mu + (k**2 * sigma**2) / 2.0)

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseBeta(BasePositiveDistribution):
    """Compute the maximum profit for a single beta distributed variable using piecewise integration."""

    _compiled_distribution = 'BETA'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        alpha = float(dist_params_list[0])
        beta = float(dist_params_list[1])

        # Shifted CDF: shape alpha becomes alpha + k
        shifted_cdf_k_diff = np.diff(sp.betainc(alpha + k, beta, bounds))

        if k == 0:
            kth_moment = 1.0
        else:
            # E[x^k] = Beta(alpha + k, beta) / Beta(alpha, beta)
            # Equivalently: (Gamma(alpha + k) * Gamma(alpha + beta)) / (Gamma(alpha) * Gamma(alpha + beta + k))
            kth_moment = (sp.gamma(alpha + k) * sp.gamma(alpha + beta)) / (sp.gamma(alpha) * sp.gamma(alpha + beta + k))

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseWeibull(BasePositiveDistribution):
    """Compute the maximum profit for a single weibull distributed variable using piecewise polynomial integration."""

    _compiled_distribution = 'WEIBULL'

    def _get_kth_integration_components(
        self, bounds: list[float] | FloatNDArray, k: int, distribution_parameters: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        dist_params_list = self._distribution_values(distribution_parameters)
        lambda_scale = float(dist_params_list[0])
        k_shape = float(dist_params_list[1])
        bounds_arr = np.asarray(bounds)

        if k == 0:
            shifted_cdf_k_diff = np.diff(st.weibull_min.cdf(bounds_arr, c=k_shape, scale=lambda_scale))
            kth_moment = 1.0
        else:
            # u = (x / lambda)^k_shape
            u_bounds = (bounds_arr / lambda_scale) ** k_shape
            gamma_shape = 1.0 + (float(k) / k_shape)

            # Shifted CDF uses regularized incomplete gamma trick
            shifted_cdf_k_diff = np.diff(st.gamma.cdf(u_bounds, a=gamma_shape))

            # E[x^k] = lambda^k * Gamma(1 + k / k_shape)
            kth_moment = (lambda_scale**k) * sp.gamma(gamma_shape)

        return kth_moment, shifted_cdf_k_diff


#: The score-side counterpart to `_distributions.ADAPTERS`'s rate-side table, keyed by the same
#: sympy.stats distribution types. Kept here rather than in `_distributions.py` because the
#: classes it names are defined in this module; `_distributions.py` importing them back would
#: make the two modules import each other.
_SCORE_CLASSES: dict[type, type[BaseMaxProfitScorePiecewise]] = {
    sympy.stats.crv_types.UniformDistribution: MaxProfitScorePiecewiseUniform,
    sympy.stats.crv_types.BetaDistribution: MaxProfitScorePiecewiseBeta,
    sympy.stats.crv_types.NormalDistribution: MaxProfitScorePiecewiseNormal,
    sympy.stats.crv_types.LogNormalDistribution: MaxProfitScorePiecewiseLogNormal,
    sympy.stats.crv_types.GammaDistribution: MaxProfitScorePiecewiseGamma,
    sympy.stats.crv_types.ExponentialDistribution: MaxProfitScorePiecewiseExponential,
    sympy.stats.crv_types.ChiSquaredDistribution: MaxProfitScorePiecewiseChi2,
    sympy.stats.crv_types.WeibullDistribution: MaxProfitScorePiecewiseWeibull,
    sympy.stats.crv_types.ParetoDistribution: MaxProfitScorePiecewisePareto,
    sympy.stats.crv_types.TriangularDistribution: MaxProfitScorePiecewiseTriangular,
}
assert _SCORE_CLASSES.keys() == ADAPTERS.keys(), (
    '_SCORE_CLASSES and _distributions.ADAPTERS must cover exactly the same distributions'
)
