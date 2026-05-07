import warnings
from collections.abc import Callable, Iterable
from itertools import pairwise
from typing import Any

import numpy as np
import scipy.special as sp
import scipy.stats as st
import sympy
from scipy.integrate import quad
from sympy import solve
from sympy.stats import density, pspace

from ....._types import FloatNDArray, IntNDArray
from ...common import (
    MetricFn,
    RateFn,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda,
)
from .common import _convex_hull, extract_distribution_parameters


class ComplexRootsError(ValueError):
    """
    Raised when the piecewise integration method encounters complex-valued roots.

    This occurs when the bound equation derived from setting two adjacent profit
    functions equal has no real solution, which means the piecewise integration
    method cannot be applied.  Use ``integration_method='quad'`` for the
    :class:`~empulse.metrics.MaxProfit` instance instead.
    """


def _build_max_profit_score_piecewise(
    profit_function: sympy.Expr,
    random_symbol: sympy.Symbol,
    deterministic_symbols: Iterable[sympy.Symbol],
) -> MetricFn:
    distribution = pspace(random_symbol).distribution
    if isinstance(distribution, sympy.stats.crv_types.UniformDistribution):
        return MaxProfitScorePiecewiseUniform(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.BetaDistribution):
        return MaxProfitScorePiecewiseBeta(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.NormalDistribution):
        return MaxProfitScorePiecewiseNormal(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.LogNormalDistribution):
        return MaxProfitScorePiecewiseLogNormal(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.GammaDistribution):
        return MaxProfitScorePiecewiseGamma(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.ExponentialDistribution):
        return MaxProfitScorePiecewiseExponential(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.ChiSquaredDistribution):
        return MaxProfitScorePiecewiseChi2(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.WeibullDistribution):
        return MaxProfitScorePiecewiseWeibull(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.ParetoDistribution):
        return MaxProfitScorePiecewisePareto(profit_function, random_symbol, deterministic_symbols)
    elif isinstance(distribution, sympy.stats.crv_types.TriangularDistribution):
        return MaxProfitScorePiecewiseTriangular(profit_function, random_symbol, deterministic_symbols)
    else:
        return MaxProfitScorePiecewise(profit_function, random_symbol, deterministic_symbols)


def _uniform_params(a: float, b: float) -> dict[str, float]:
    return {'loc': a, 'scale': b - a}


def _beta_params(a: float, b: float) -> dict[str, float]:
    return {'a': a, 'b': b}


def _normal_params(mu: float, sigma: float) -> dict[str, float]:
    return {'loc': mu, 'scale': sigma}


def _lognormal_params(mu: float, sigma: float) -> dict[str, float]:
    return {'s': sigma, 'scale': float(np.exp(mu))}


def _gamma_params(k: float, theta: float) -> dict[str, float]:
    return {'a': k, 'scale': theta}


def _expon_params(rate: float) -> dict[str, float]:
    return {'loc': 0.0, 'scale': 1.0 / rate}


def _chi2_params(df: float) -> dict[str, float]:
    return {'df': df}


def _weibull_params(a: float, b: float) -> dict[str, float]:
    return {'c': b, 'scale': a}


def _pareto_params(a: float, b: float) -> dict[str, float]:
    return {'b': b, 'scale': a}


def _triangular_params(a: float, b: float, c: float) -> dict[str, float]:
    return {'loc': a, 'scale': b - a, 'c': (c - a) / (b - a)}


def _build_max_profit_rate_piecewise(
    profit_function: sympy.Expr,
    rate_function: sympy.Expr,
    random_symbol: sympy.Symbol,
    deterministic_symbols: Iterable[sympy.Symbol],
) -> RateFn:
    distribution = pspace(random_symbol).distribution
    if isinstance(distribution, sympy.stats.crv_types.UniformDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.uniform, _uniform_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.BetaDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.beta, _beta_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.NormalDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.norm, _normal_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.LogNormalDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.lognorm, _lognormal_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.GammaDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.gamma, _gamma_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.ExponentialDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.expon, _expon_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.ChiSquaredDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.chi2, _chi2_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.WeibullDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.weibull_min, _weibull_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.ParetoDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.pareto, _pareto_params
        )
    elif isinstance(distribution, sympy.stats.crv_types.TriangularDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.triang, _triangular_params
        )
    else:
        return MaxProfitRatePiecewise(profit_function, rate_function, random_symbol, deterministic_symbols)


def compute_integral_quad(
    integrand: sympy.Expr,
    lower_bound: float,
    upper_bound: float,
    true_positive_rate: float,
    false_positive_rate: float,
    random_var: sympy.Symbol,
) -> float:
    """Compute the integral using scipy quadrature for one stochastic variable."""
    if lower_bound == upper_bound:
        return 0.0
    integrand = integrand.subs('F_0', true_positive_rate).subs('F_1', false_positive_rate).evalf()
    if not integrand.free_symbols:  # if the integrand is constant, no need to call quad
        if integrand == 0:  # need this separate path since sometimes upper or lower bound can be infinite
            return 0
        return float(integrand * (upper_bound - lower_bound))
    integrand_fn = _safe_lambdify(integrand, [random_var])
    result, _ = quad(integrand_fn, lower_bound, upper_bound)
    result: float
    return result


def compute_piecewise_bounds(
    compute_bounds_fns: list[Callable[..., float]],
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    positive_class_prior: float,
    negative_class_prior: float,
    random_var_bounds: tuple[float | sympy.Expr, ...],
    distribution_parameters: dict[str, Any],
    fix_inf: bool = True,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    **kwargs: Any,
) -> tuple[list[float], float, float, list[float], list[float]]:
    """Compute the consecutive bounds of the stochastic variable for which the expected profit is equal."""
    # 1. Prepare fully vectorized inputs
    # F_0, F_1 are the "current" vertices. F_2, F_3 are the "next" vertices.
    F_0 = true_positive_rates[:-1]  # noqa: N806
    F_1 = false_positive_rates[:-1]  # noqa: N806
    F_2 = true_positive_rates[1:]  # noqa: N806
    F_3 = false_positive_rates[1:]  # noqa: N806

    all_bounds = []
    all_tprs = []
    all_fprs = []

    # Bound Computation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for compute_bound in compute_bounds_fns:
            # We pass the full arrays in. NumPy executes this instantly.
            b_arr = compute_bound(
                F_0=F_0, F_1=F_1, F_2=F_2, F_3=F_3, pi_0=positive_class_prior, pi_1=negative_class_prior, **kwargs
            )
            # Guarantee 1D array even if b_arr returns a scalar for some reason
            b_arr = np.atleast_1d(b_arr)
            all_bounds.append(b_arr)
            all_tprs.append(F_2)
            all_fprs.append(F_3)

    if all_bounds:
        bounds_arr = np.concatenate(all_bounds)
        tprs_arr = np.concatenate(all_tprs)
        fprs_arr = np.concatenate(all_fprs)
    else:
        bounds_arr = np.array([], dtype=float)
        tprs_arr = np.array([], dtype=float)
        fprs_arr = np.array([], dtype=float)

    # --- Complex-root guards ---
    # Guard 1: lambdified I → numpy complex (e.g. 1j * ...).
    if np.iscomplexobj(bounds_arr) and np.any(np.iscomplex(bounds_arr)):
        raise ComplexRootsError(
            'The piecewise integration found complex-valued bounds for the current parameter values, '
            'indicating that the profit function polynomial has complex roots. '
            "Use integration_method='quad' for MaxProfit() to avoid this issue."
        )
    # Guard 2: sqrt(negative) → nan in numpy.
    # NaN bounds on *non-degenerate* hull segments (where both TPR **and** FPR change between
    # adjacent hull vertices) indicate that the bound equation has complex roots for those
    # parameter values.  Degenerate segments (only TPR or only FPR changes) trivially reduce
    # to 0 or ±∞ and are handled correctly.
    # We flag the issue the first time ANY non-degenerate segment yields a NaN bound so the
    # user gets a clear error rather than a silently-wrong result.
    nondegenerate_mask = np.concatenate([((F_0 != F_2) & (F_1 != F_3)) for _ in compute_bounds_fns])
    n_raw_bounds = len(bounds_arr)
    valid_mask = ~np.isnan(bounds_arr)
    if n_raw_bounds > 0 and nondegenerate_mask.any():
        nan_on_nondegenerate = ~valid_mask & nondegenerate_mask
        if nan_on_nondegenerate.any():
            raise ComplexRootsError(
                'Some piecewise integration bounds evaluated to NaN on non-degenerate ROC-hull '
                'segments, indicating the profit function polynomial has complex roots for these '
                'parameter values. '
                "Use integration_method='quad' for MaxProfit() to avoid this issue."
            )
    bounds_arr = bounds_arr[valid_mask]
    if n_raw_bounds > 0 and len(bounds_arr) == 0:
        raise ComplexRootsError(
            'All piecewise integration bounds evaluated to NaN for the current parameter values, '
            'indicating the profit function polynomial has complex roots for this ROC hull. '
            "Use integration_method='quad' for MaxProfit() to avoid this issue."
        )
    tprs_arr = tprs_arr[valid_mask]
    fprs_arr = fprs_arr[valid_mask]

    if fix_inf:
        bounds_arr[bounds_arr == -np.inf] = np.inf
    else:
        bounds_arr[bounds_arr == np.inf] = -np.inf

    sort_idx = np.argsort(bounds_arr)
    bounds_arr = bounds_arr[sort_idx]
    tprs_arr = tprs_arr[sort_idx]
    fprs_arr = fprs_arr[sort_idx]

    # Resolve Lower and Upper Bounds
    if upper_bound is None:
        upper_bound = random_var_bounds[1]
        if isinstance(upper_bound, sympy.Expr):
            upper_bound = upper_bound.subs(distribution_parameters)
            upper_bound = np.inf if upper_bound == sympy.oo else float(upper_bound)
        else:
            upper_bound = float(upper_bound)

    if lower_bound is None:
        lower_bound = random_var_bounds[0]
        if isinstance(lower_bound, sympy.Expr):
            lower_bound = lower_bound.subs(distribution_parameters)
            lower_bound = -np.inf if lower_bound == -sympy.oo else float(lower_bound)
        else:
            lower_bound = float(lower_bound)

    bounds_arr = np.clip(bounds_arr, lower_bound, upper_bound)
    bounds_list = bounds_arr.tolist()
    tprs_list = tprs_arr.tolist()
    fprs_list = fprs_arr.tolist()

    bounds_list.append(upper_bound)
    bounds_list.insert(0, lower_bound)

    # Handle start values
    if fix_inf:
        if len(fprs_list) >= 2:
            start_value = 0.0 if fprs_list[0] < fprs_list[1] else 1.0
        else:
            start_value = 0.0 if (len(fprs_list) > 0 and fprs_list[0] == 1.0) else 1.0
        fprs_list.insert(0, start_value)
        tprs_list.insert(0, start_value)
    else:
        if len(fprs_list) >= 2:
            start_value = 0.0 if fprs_list[0] > fprs_list[1] else 1.0
        else:
            start_value = 0.0 if (len(fprs_list) > 0 and fprs_list[0] == 1.0) else 1.0
        fprs_list.append(start_value)
        tprs_list.append(start_value)

    return bounds_list, upper_bound, lower_bound, tprs_list, fprs_list


class MaxProfitRatePiecewise:
    """
    Compute the maximum profit rate for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr,
        random_symbol: sympy.Symbol,
        deterministic_symbols: Iterable[sympy.Symbol],
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol
        self.derivative = (
            sympy.diff(profit_function, random_symbol).subs('F_0', 1).subs('F_1', 1).subs('pi_0', 1).subs('pi_1', 1)
        )
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        self.compute_bounds_eq = solve(profit_function - profit_prime, random_symbol)[0]
        self.compute_bounds = _safe_lambdify(self.compute_bounds_eq, list(self.compute_bounds_eq.free_symbols))

        self.random_var_bounds = pspace(random_symbol).domain.set.args
        self.distribution_args = pspace(random_symbol).distribution.args

        self.integrand = rate_function * density(random_symbol).pdf(random_symbol)

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            self.dist_params = []
        else:
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the optimal rate."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        # distribution parameters of the random variable
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)

        fix_inf = not self.derivative.subs(kwargs).is_negative

        bounds, _, _, tprs, fprs = compute_piecewise_bounds(
            [self.compute_bounds],
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            fix_inf=fix_inf,
            **kwargs,
        )

        integrand_ = (
            self.integrand
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lb, ub), tpr, fpr in zip(pairwise(bounds), tprs, fprs, strict=True):
            score += compute_integral_quad(integrand_, lb, ub, tpr, fpr, self.random_symbol)
        return score


class ExactMaxProfitRatePiecewise:
    """
    Base class to compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr,
        random_symbol: sympy.Symbol,
        deterministic_symbols: Iterable[sympy.Symbol],
        cdf_function: st.rv_continuous,
        sympy_to_scipy_params_fn: Callable[[float, ...], dict[str, float]],
    ) -> None:
        self.cdf_function = cdf_function
        self.sympy_to_scipy_params_fn = sympy_to_scipy_params_fn
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol
        self.derivative = (
            sympy.diff(profit_function, random_symbol).subs('F_0', 1).subs('F_1', 1).subs('pi_0', 1).subs('pi_1', 1)
        )

        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        self.compute_bounds_eqs = []
        self.compute_bounds_fns = []
        for root in solve(profit_function - profit_prime, random_symbol):
            # if not root.has(sympy.I):
            self.compute_bounds_eqs.append(root)
            self.compute_bounds_fns.append(_safe_lambdify(root, list(root.free_symbols)))
        if not self.compute_bounds_eqs:
            raise ValueError('No real roots found for the equation. Please check the profit function.')

        self.random_var_bounds = pspace(random_symbol).domain.set.args
        self.distribution_args = pspace(random_symbol).distribution.args

        self.rate_eq = rate_function
        self.rate_fn = _safe_lambdify(self.rate_eq)

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            self.dist_params = []
        else:
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit rate."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        # Distribution parameters of the random variable
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)

        fix_inf = not self.derivative.subs(kwargs).is_negative

        # We capture upper and lower bounds as the Uniform distribution needs them
        bounds, _, _, tprs, fprs = compute_piecewise_bounds(
            self.compute_bounds_fns,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            fix_inf=fix_inf,
            **kwargs,
        )
        bounds = [float(bound) for bound in bounds]

        rate = _safe_run_lambda(
            self.rate_fn,
            self.rate_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=np.array(tprs),
            F_1=np.array(fprs),
            **kwargs,
        )

        return self._integrate(
            bounds=bounds,
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
            self.cdf_function.cdf(bounds, **self.sympy_to_scipy_params_fn(*distribution_parameters.values()))
        )
        optimal_rate = (rate * cdf_diff).sum()
        return float(optimal_rate)


class MaxProfitScorePiecewise:
    """
    Compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    def __init__(
        self, profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol
        self.derivative = (
            sympy.diff(profit_function, random_symbol).subs('F_0', 1).subs('F_1', 1).subs('pi_0', 1).subs('pi_1', 1)
        )
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        root = solve(profit_function - profit_prime, random_symbol)[0]
        if root.has(sympy.I):
            raise ComplexRootsError(
                'The piecewise integration method requires real-valued roots, but the bound equation '
                'for this profit function has complex roots. '
                "Use integration_method='quad' for MaxProfit() to avoid this issue."
            )
        self.compute_bounds_eq = root
        self.compute_bounds = _safe_lambdify(self.compute_bounds_eq, list(self.compute_bounds_eq.free_symbols))

        self.random_var_bounds = pspace(random_symbol).domain.set.args
        self.distribution_args = pspace(random_symbol).distribution.args

        self.integrand = profit_function * density(random_symbol).pdf(random_symbol)

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            self.dist_params = []
        else:
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        # distribution parameters of the random variable
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)

        fix_inf = not self.derivative.subs(kwargs).is_negative

        bounds, _, _, tprs, fprs = compute_piecewise_bounds(
            [self.compute_bounds],
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            fix_inf=fix_inf,
            **kwargs,
        )

        integrand_ = (
            self.integrand
            .subs(kwargs)
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lb, ub), tpr, fpr in zip(pairwise(bounds), tprs, fprs, strict=False):
            score += compute_integral_quad(integrand_, lb, ub, tpr, fpr, self.random_symbol)
        return score


class BaseMaxProfitScorePiecewise:
    """
    Base class to compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    def __init__(
        self, profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol

        self.derivative = (
            sympy.diff(profit_function, random_symbol).subs('F_0', 1).subs('F_1', 1).subs('pi_0', 1).subs('pi_1', 1)
        )
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')

        self.compute_bounds_eqs = []
        self.compute_bounds_fns = []
        complex_roots = []
        for root in solve(profit_function - profit_prime, random_symbol):
            if root.has(sympy.I):
                complex_roots.append(root)
            else:
                self.compute_bounds_eqs.append(root)
                self.compute_bounds_fns.append(_safe_lambdify(root, list(root.free_symbols)))
        if complex_roots and not self.compute_bounds_eqs:
            raise ComplexRootsError(
                'The piecewise integration method requires real-valued roots, but all roots of the '
                'bound equation for this profit function are complex. '
                "Use integration_method='quad' for MaxProfit() to avoid this issue."
            )
        if not self.compute_bounds_eqs:
            raise ValueError('No real roots found for the equation. Please check the profit function.')

        self.random_var_bounds = pspace(random_symbol).domain.set.args
        self.distribution_args = pspace(random_symbol).distribution.args

        # Dynamically extract polynomial coefficients [a_0, a_1, ..., a_n]
        expanded_profit = sympy.expand(profit_function)
        polynomial_degree = sympy.degree(expanded_profit, random_symbol)
        collected = sympy.collect(expanded_profit, random_symbol, evaluate=False)

        self.coefficient_eqs = []
        self.coefficient_fns = []

        for k in range(polynomial_degree + 1):
            # Extract the term for random_symbol**k. (Note: random_symbol**0 is 1)
            term_key = random_symbol**k if k > 0 else 1
            eq = collected.get(term_key, sympy.S.Zero)
            self.coefficient_eqs.append(eq)
            self.coefficient_fns.append(_safe_lambdify(eq))

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            self.dist_params = []
        else:
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        # Distribution parameters of the random variable
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)

        fix_inf = not self.derivative.subs(kwargs).is_negative

        bounds, upper_bound, lower_bound, tprs, fprs = compute_piecewise_bounds(
            # bounds, upper_bound, lower_bound = compute_piecewise_bounds(
            self.compute_bounds_fns,
            # self.compute_bounds_fns[0],
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            fix_inf=fix_inf,
            **kwargs,
        )
        bounds = [float(bound) for bound in bounds]

        # Evaluate all polynomial coefficients dynamically
        evaluated_coefficients = [
            _safe_run_lambda(
                fn,
                eq,
                pi_0=positive_class_prior,
                pi_1=negative_class_prior,
                F_0=np.array(tprs),
                F_1=np.array(fprs),
                # F_0=true_positive_rates,
                # F_1=false_positive_rates,
                **kwargs,
            )
            for fn, eq in zip(self.coefficient_fns, self.coefficient_eqs, strict=True)
        ]

        return self._integrate(
            bounds=bounds,
            coefficients=evaluated_coefficients,
            distribution_parameters=distribution_parameters,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
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


def _safe_pow_phi(x: np.ndarray, exp: int, phi: float) -> np.ndarray:
    """Compute x^exp * phi(x), treating inf^exp * 0 as 0."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = (x**exp) * phi
    # Where phi is 0 (i.e., x is ±inf), the product should be 0, not NaN
    result[phi == 0] = 0.0
    return result


class MaxProfitScorePiecewiseNormal(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single normal distributed variable using piecewise polynomial integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[np.ndarray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
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
        coefficients: list[float],
        distribution_parameters: dict,
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

    def _get_kth_integration_components(self, bounds: list[float], k: int, distribution_parameters: dict) -> tuple:
        """Return (kth_moment, shifted_cdf_k_diff) for a specific degree k."""
        raise NotImplementedError('Subclasses must implement `_get_kth_integration_components`.')


class MaxProfitScorePiecewiseGamma(BasePositiveDistribution):
    """Compute the maximum profit for a single gamma distributed variable using piecewise integration."""

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
        alpha = float(dist_params_list[0])
        lambda_rate = float(dist_params_list[1])

        shifted_cdf_k_diff = np.diff(st.gamma.cdf(bounds, a=alpha + k, scale=lambda_rate))
        kth_moment = 1.0 if k == 0 else (lambda_rate**k) * (sp.gamma(alpha + k) / sp.gamma(alpha))

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewisePareto(BasePositiveDistribution):
    """Compute the maximum profit for a single triangular distributed variable using piecewise integration."""

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
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

    def _integrate(
        self,
        bounds: list[float],
        coefficients: list[np.ndarray],
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())

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

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
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

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
        df = float(dist_params_list[0])

        # Shifted CDF: adding 2*k to degrees of freedom
        shifted_cdf_k_diff = np.diff(st.chi2.cdf(bounds, df=df + 2.0 * k))

        kth_moment = 1.0 if k == 0 else (2.0**k) * sp.gamma(k + df / 2.0) / sp.gamma(df / 2.0)

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseLogNormal(BasePositiveDistribution):
    """Compute the maximum profit for a single log normal distributed variable using piecewise integration."""

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
        mu = float(dist_params_list[0])
        sigma = float(dist_params_list[1])

        # Shifted CDF: scale shifts to exp(mu + k * sigma^2)
        shifted_scale = np.exp(mu + k * sigma**2)
        shifted_cdf_k_diff = np.diff(st.lognorm.cdf(bounds, s=sigma, scale=shifted_scale))

        kth_moment = 1.0 if k == 0 else np.exp(k * mu + (k**2 * sigma**2) / 2.0)

        return kth_moment, shifted_cdf_k_diff


class MaxProfitScorePiecewiseBeta(BasePositiveDistribution):
    """Compute the maximum profit for a single beta distributed variable using piecewise integration."""

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
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

    def _get_kth_integration_components(self, bounds, k, distribution_parameters):
        dist_params_list = list(distribution_parameters.values())
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
