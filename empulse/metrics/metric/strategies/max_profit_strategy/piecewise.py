import warnings
from collections.abc import Callable, Iterable
from itertools import islice, pairwise
from typing import Any, ClassVar

import numpy as np
import scipy.special as sp
import scipy.stats as st
import sympy
from scipy.integrate import quad
from sympy import solve
from sympy.stats import density, pspace
from sympy.utilities import lambdify

from ....._types import FloatNDArray, IntNDArray
from ...common import (
    MetricFn,
    RateFn,
    SympyFnPickleMixin,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda,
)
from .common import _convex_hull, extract_distribution_parameters


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


def _build_max_profit_rate_piecewise(
    profit_function: sympy.Expr,
    rate_function: sympy.Expr,
    random_symbol: sympy.Symbol,
    deterministic_symbols: Iterable[sympy.Symbol],
) -> RateFn:
    distribution = pspace(random_symbol).distribution
    if isinstance(distribution, sympy.stats.crv_types.UniformDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.uniform,
            lambda a, b: {'loc': a, 'scale': b - a},
        )
    elif isinstance(distribution, sympy.stats.crv_types.BetaDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.beta, lambda a, b: {'a': a, 'b': b}
        )
    elif isinstance(distribution, sympy.stats.crv_types.NormalDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.norm,
            lambda a, b: {'loc': a, 'scale': b},
        )
    elif isinstance(distribution, sympy.stats.crv_types.LogNormalDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.lognorm,
            lambda mu, sigma: {'s': sigma, 'scale': np.exp(mu)},
        )
    elif isinstance(distribution, sympy.stats.crv_types.GammaDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.gamma,
            lambda k, theta: {'a': k, 'scale': theta},
        )
    elif isinstance(distribution, sympy.stats.crv_types.ExponentialDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.expon,
            lambda rate: {'loc': 0, 'scale': 1 / rate},
        )
    elif isinstance(distribution, sympy.stats.crv_types.ChiSquaredDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function, rate_function, random_symbol, deterministic_symbols, st.chi2, lambda a: {'df': a}
        )
    elif isinstance(distribution, sympy.stats.crv_types.WeibullDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.weibull_min,
            lambda a, b: {'c': b, 'scale': a},
        )
    elif isinstance(distribution, sympy.stats.crv_types.ParetoDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.pareto,
            lambda a, b: {'b': b, 'scale': a},
        )
    elif isinstance(distribution, sympy.stats.crv_types.TriangularDistribution):
        return ExactMaxProfitRatePiecewise(
            profit_function,
            rate_function,
            random_symbol,
            deterministic_symbols,
            st.triang,
            lambda a, b, c: {'loc': a, 'scale': b - a, 'c': (c - a) / (b - a)},
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
    integrand_fn = lambdify(random_var, integrand)
    result, _ = quad(integrand_fn, lower_bound, upper_bound)
    result: float
    return result


def compute_piecewise_bounds(
    compute_bounds: Callable[..., float],
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    positive_class_prior: float,
    negative_class_prior: float,
    random_var_bounds: tuple[float | sympy.Expr, ...],
    distribution_parameters: dict[str, Any],
    **kwargs: Any,
) -> tuple[list[float], float, float]:
    """
    Compute the consecutive bounds of the stochastic variable for which the expected profit is equal.

    These bounds can then be used during piecewise integration.
    """
    bounds = []
    for (tpr0, fpr0), (tpr1, fpr1) in islice(
        pairwise(zip(true_positive_rates, false_positive_rates, strict=False)), len(true_positive_rates) - 1
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            computed_bounds = compute_bounds(
                F_0=tpr0,
                F_1=fpr0,
                F_2=tpr1,
                F_3=fpr1,
                pi_0=positive_class_prior,
                pi_1=negative_class_prior,
                **kwargs,
            )
            # TODO: temporary fix, make sure infinity signs are always correct
            computed_bounds = np.inf if computed_bounds == -np.inf else computed_bounds
        bounds.append(computed_bounds)

    # the compute_bounds function only computes the internal bounds,
    # so we need to add the lower and upper bounds of the random variable
    if isinstance(upper_bound := random_var_bounds[1], sympy.Expr):
        upper_bound = upper_bound.subs(distribution_parameters)
        if upper_bound == sympy.oo:
            upper_bound = np.inf
    bounds.append(upper_bound)
    if isinstance(lower_bound := random_var_bounds[0], sympy.Expr):
        lower_bound = lower_bound.subs(distribution_parameters)
        if lower_bound == -sympy.oo:
            lower_bound = -np.inf
    bounds.insert(0, lower_bound)

    # it is possible that some of the computed bounds are outside the accepted interval [lower_bound, upper_bound]
    # replace values that are outside the interval with the respective bounds
    # this will have the effect of essentially setting that part to zero
    for i in range(len(bounds)):
        if bounds[i] < lower_bound:
            bounds[i] = lower_bound
        elif bounds[i] > upper_bound:
            bounds[i] = upper_bound
    return bounds, upper_bound, lower_bound


class MaxProfitRatePiecewise(SympyFnPickleMixin):
    """
    Compute the maximum profit rate for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {'compute_bounds': 'compute_bounds_eq'}

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr,
        random_symbol: sympy.Symbol,
        deterministic_symbols: Iterable[sympy.Symbol],
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        self.compute_bounds_eq = solve(profit_function - profit_prime, random_symbol)[0]
        self.compute_bounds = lambdify(list(self.compute_bounds_eq.free_symbols), self.compute_bounds_eq)

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
        bounds, _, _ = compute_piecewise_bounds(
            self.compute_bounds,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            **kwargs,
        )

        integrand_ = (
            self.integrand
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lb, ub), tpr, fpr in zip(pairwise(bounds), true_positive_rates, false_positive_rates, strict=False):
            score += compute_integral_quad(integrand_, lb, ub, tpr, fpr, self.random_symbol)
        return score


class ExactMaxProfitRatePiecewise(SympyFnPickleMixin):
    """
    Base class to compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {'compute_bounds': 'compute_bounds_eq'}

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

        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        self.compute_bounds_eq = solve(profit_function - profit_prime, random_symbol)[0]
        self.compute_bounds = lambdify(list(self.compute_bounds_eq.free_symbols), self.compute_bounds_eq)

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

        # We capture upper and lower bounds as the Uniform distribution needs them
        bounds, _, _ = compute_piecewise_bounds(
            self.compute_bounds,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            **kwargs,
        )
        bounds = [float(bound) for bound in bounds]

        rate = _safe_run_lambda(
            self.rate_fn,
            self.rate_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=true_positive_rates,
            F_1=false_positive_rates,
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


class MaxProfitScorePiecewise(SympyFnPickleMixin):
    """
    Compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {'compute_bounds': 'compute_bounds_eq'}

    def __init__(
        self, profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        self.compute_bounds_eq = solve(profit_function - profit_prime, random_symbol)[0]
        self.compute_bounds = lambdify(list(self.compute_bounds_eq.free_symbols), self.compute_bounds_eq)

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
        bounds, _, _ = compute_piecewise_bounds(
            self.compute_bounds,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
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
        for (lb, ub), tpr, fpr in zip(pairwise(bounds), true_positive_rates, false_positive_rates, strict=False):
            score += compute_integral_quad(integrand_, lb, ub, tpr, fpr, self.random_symbol)
        return score


class BaseMaxProfitScorePiecewise(SympyFnPickleMixin):
    """
    Base class to compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {'compute_bounds': 'compute_bounds_eq'}

    def __init__(
        self, profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
    ) -> None:
        self.deterministic_symbols = deterministic_symbols
        self.random_symbol = random_symbol
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        self.compute_bounds_eq = solve(profit_function - profit_prime, random_symbol)[0]
        self.compute_bounds = lambdify(list(self.compute_bounds_eq.free_symbols), self.compute_bounds_eq)

        self.random_var_bounds = pspace(random_symbol).domain.set.args
        self.distribution_args = pspace(random_symbol).distribution.args

        collected = sympy.collect(sympy.factor(profit_function, random_symbol), random_symbol, evaluate=False)
        self.coefficient_eq = collected.get(random_symbol, 0)
        self.coefficient_fn = _safe_lambdify(self.coefficient_eq)
        self.intercept_eq = collected.get(1, 0)
        self.intercept_fn = _safe_lambdify(self.intercept_eq)

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

        # We capture upper and lower bounds as the Uniform distribution needs them
        bounds, upper_bound, lower_bound = compute_piecewise_bounds(
            self.compute_bounds,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            negative_class_prior,
            self.random_var_bounds,
            distribution_parameters,
            **kwargs,
        )
        bounds = [float(bound) for bound in bounds]

        # Shared intercept and coefficient calculations
        coefficient = _safe_run_lambda(
            self.coefficient_fn,
            self.coefficient_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=true_positive_rates,
            F_1=false_positive_rates,
            **kwargs,
        )
        intercept = _safe_run_lambda(
            self.intercept_fn,
            self.intercept_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=true_positive_rates,
            F_1=false_positive_rates,
            **kwargs,
        )

        return self._integrate(
            bounds=bounds,
            coefficient=coefficient,
            intercept=intercept,
            distribution_parameters=distribution_parameters,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
        )

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        """Integrate distribution-specific piecewise math."""
        raise NotImplementedError('Subclasses must implement the `_integrate` method.')


class MaxProfitScorePiecewiseUniform(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single uniform distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        lower_bounds = np.asarray(bounds[:-1])
        upper_bounds = np.asarray(bounds[1:])
        max_val = float(upper_bound)
        min_val = float(lower_bound)

        score = (
            1
            / (max_val - min_val)
            * (coefficient / 2 * (upper_bounds**2 - lower_bounds**2) + intercept * (upper_bounds - lower_bounds)).sum()
        )
        return float(score)


class MaxProfitScorePiecewiseNormal(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single normal distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        mu = dist_params_list[0]
        sigma = dist_params_list[1]

        bounds = np.asarray(bounds)
        z = (bounds - mu) / sigma
        Phi_diffs = np.diff(st.norm.cdf(z))
        phi_diffs = np.diff(st.norm.pdf(z))

        score = ((coefficient * mu + intercept) * Phi_diffs - coefficient * sigma * phi_diffs).sum()
        return float(score)


class MaxProfitScorePiecewiseBeta(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single beta distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        alpha = dist_params_list[0]
        beta = dist_params_list[1]

        cdf_diff = np.diff(st.beta.cdf(bounds, a=alpha, b=beta))
        cdf_1_diff = np.diff(st.beta.cdf(bounds, a=alpha + 1, b=beta))

        mean_beta = st.beta.mean(a=alpha, b=beta)

        score = (coefficient * mean_beta * cdf_1_diff + intercept * cdf_diff).sum()
        return float(score)


class MaxProfitScorePiecewiseGamma(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single gamma distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        alpha = dist_params_list[0]
        lambda_rate = dist_params_list[1]

        cdf_diff = np.diff(st.gamma.cdf(bounds, a=alpha, scale=lambda_rate))
        cdf_1_diff = np.diff(st.gamma.cdf(bounds, a=alpha + 1, scale=lambda_rate))

        mean_gamma = alpha * lambda_rate

        score = (coefficient * mean_gamma * cdf_1_diff + intercept * cdf_diff).sum()
        return float(score)


class MaxProfitScorePiecewiseExponential(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single exponential distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        lambda_rate = dist_params_list[0]
        scale = 1.0 / lambda_rate

        cdf_diff = np.diff(st.expon.cdf(bounds, scale=scale))
        cdf_1_diff = np.diff(st.gamma.cdf(bounds, a=2, scale=scale))

        mean_expon = scale

        score = (coefficient * mean_expon * cdf_1_diff + intercept * cdf_diff).sum()
        return float(score)


class MaxProfitScorePiecewiseChi2(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single chi-squared distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        df = dist_params_list[0]

        cdf_diff = np.diff(st.chi2.cdf(bounds, df=df))
        cdf_1_diff = np.diff(st.chi2.cdf(bounds, df=df + 2))

        mean_chi2 = df

        score = (coefficient * mean_chi2 * cdf_1_diff + intercept * cdf_diff).sum()
        return float(score)


class MaxProfitScorePiecewiseLogNormal(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single log normal distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        mu = float(dist_params_list[0])
        sigma = float(dist_params_list[1])

        cdf_diff = np.diff(st.lognorm.cdf(bounds, s=sigma, scale=np.exp(mu)))
        shifted_scale = np.exp(mu + sigma**2)
        cdf_1_diff = np.diff(st.lognorm.cdf(bounds, s=sigma, scale=shifted_scale))

        mean_lognorm = np.exp(mu + (sigma**2) / 2.0)

        score = (coefficient * mean_lognorm * cdf_1_diff + intercept * cdf_diff).sum()
        return float(score)


class MaxProfitScorePiecewisePareto(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single pareto distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())

        # SymPy's Pareto generally takes shape (alpha) and scale (x_m)
        x_m = float(dist_params_list[0])
        alpha = float(dist_params_list[1])

        if alpha <= 1:
            raise ValueError('The shape parameter (alpha) must be greater than 1 for the mean to exist.')

        # H(x; x_m, alpha) - Standard Pareto CDF
        cdf_diff = np.diff(st.pareto.cdf(bounds, b=alpha, scale=x_m))

        # H(x; x_m, alpha - 1) - Shifted Pareto CDF for the expected value integral
        cdf_shifted_diff = np.diff(st.pareto.cdf(bounds, b=alpha - 1, scale=x_m))

        # The mean of a Pareto distribution is (alpha * x_m) / (alpha - 1)
        mean_pareto = (alpha * x_m) / (alpha - 1.0)

        # Integrate: a * E(x) * [H(d; alpha-1) - H(c; alpha-1)] + b * [H(d; alpha) - H(c; alpha)]
        score = (coefficient * mean_pareto * cdf_shifted_diff + intercept * cdf_diff).sum()
        return float(score)


class MaxProfitScorePiecewiseTriangular(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single triangular distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())

        # SymPy's Triangular distribution generally takes a (min), b (max), and c (mode/peak)
        # Note: I am naming the mode 'm' here to match our math, saving 'c' for SciPy's shape parameter.
        a = float(dist_params_list[0])
        b = float(dist_params_list[1])
        m = float(dist_params_list[2])

        # Mapping to SciPy's parameterization: loc = min, scale = max - min, c = (mode - min) / scale
        loc = a
        scale = b - a
        c_shape = (m - a) / scale

        # H(x) - Standard Triangular CDF
        cdf_diff = np.diff(st.triang.cdf(bounds, c=c_shape, loc=loc, scale=scale))

        # Vectorized calculation for the cumulative partial expectation A(x) = \int_a^x t*h(t) dt
        bounds_arr = np.clip(np.asarray(bounds), a, b)
        a_x = np.zeros_like(bounds_arr)

        # 1. Evaluate bounds falling in the first segment: [a, m]
        mask1 = bounds_arr <= m
        if np.any(mask1) and m > a:
            x1 = bounds_arr[mask1]
            a_x[mask1] = (2.0 / ((b - a) * (m - a))) * (x1**3 / 3.0 - a * x1**2 / 2.0 + a**3 / 6.0)

        # 2. Evaluate bounds falling in the second segment: (m, b]
        mask2 = bounds_arr > m
        if np.any(mask2) and b > m:
            x2 = bounds_arr[mask2]
            a_m = (2.0 / ((b - a) * (m - a))) * (m**3 / 3.0 - a * m**2 / 2.0 + a**3 / 6.0) if m > a else 0.0
            a_x[mask2] = a_m + (2.0 / ((b - a) * (b - m))) * (b * (x2**2 - m**2) / 2.0 - (x2**3 - m**3) / 3.0)

        # The expected value integral between bounds d and c is simply A(d) - A(c)
        expected_diff = np.diff(a_x)

        # Integrate: a * [A(d) - A(c)] + b * [H(d) - H(c)]
        # Note: We don't need to multiply by the mean here because A(x) already intrinsically computes E[x]
        score = (coefficient * expected_diff + intercept * cdf_diff).sum()

        return float(score)


class MaxProfitScorePiecewiseWeibull(BaseMaxProfitScorePiecewise):
    """Compute the maximum profit for a single weibull distributed variable using piecewise integration."""

    def _integrate(
        self,
        bounds: list[float],
        coefficient: FloatNDArray,
        intercept: FloatNDArray,
        distribution_parameters: dict[str, Any],
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        dist_params_list = list(distribution_parameters.values())
        lambda_scale = float(dist_params_list[0])
        k_shape = float(dist_params_list[1])
        bounds_arr = np.asarray(bounds)

        cdf_diff = np.diff(st.weibull_min.cdf(bounds_arr, c=k_shape, scale=lambda_scale))
        u_bounds = (bounds_arr / lambda_scale) ** k_shape
        gamma_shape = 1.0 + (1.0 / k_shape)
        cdf_expected_diff = np.diff(st.gamma.cdf(u_bounds, a=gamma_shape))

        mean_weibull = lambda_scale * sp.gamma(gamma_shape)

        score = (coefficient * mean_weibull * cdf_expected_diff + intercept * cdf_diff).sum()

        return float(score)
