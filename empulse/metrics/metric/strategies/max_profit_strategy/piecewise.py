import warnings
from collections.abc import Callable, Iterable
from itertools import islice, pairwise
from typing import Any, ClassVar

import numpy as np
import scipy.stats as st
import sympy
from scipy.integrate import quad
from sympy import solve
from sympy.stats import density, pspace
from sympy.utilities import lambdify

from ....._types import FloatNDArray, IntNDArray
from ...common import (
    MetricFn,
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
        return MaxProfitScorePiecewise(profit_function, random_symbol, deterministic_symbols)
    else:
        return MaxProfitScorePiecewise(profit_function, random_symbol, deterministic_symbols)


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

        integrand_ = (
            self.integrand
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lower_bound, upper_bound), tpr, fpr in zip(
            pairwise(bounds), true_positive_rates, false_positive_rates, strict=False
        ):
            score += compute_integral_quad(integrand_, lower_bound, upper_bound, tpr, fpr, self.random_symbol)
        return score


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

        integrand_ = (
            self.integrand
            .subs(kwargs)
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lower_bound, upper_bound), tpr, fpr in zip(
            pairwise(bounds), true_positive_rates, false_positive_rates, strict=False
        ):
            score += compute_integral_quad(integrand_, lower_bound, upper_bound, tpr, fpr, self.random_symbol)
        return score


class MaxProfitScorePiecewiseBeta(SympyFnPickleMixin):
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
        bounds = [float(bound) for bound in bounds]
        distribution_parameters = list(distribution_parameters.values())
        alpha = distribution_parameters[0]
        beta = distribution_parameters[1]

        cdf_diff = np.diff(st.beta.cdf(bounds, a=alpha, b=beta))
        cdf_1_diff = np.diff(st.beta.cdf(bounds, a=alpha + 1, b=beta))

        mean_gamma = st.beta.mean(a=alpha, b=beta)
        temp_1 = mean_gamma * _safe_run_lambda(
            self.coefficient_fn,
            self.coefficient_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=true_positive_rates,
            F_1=false_positive_rates,
            **kwargs,
        )
        temp_2 = _safe_run_lambda(
            self.intercept_fn,
            self.intercept_eq,
            pi_0=positive_class_prior,
            pi_1=negative_class_prior,
            F_0=true_positive_rates,
            F_1=false_positive_rates,
            **kwargs,
        )
        score = (temp_1 * cdf_1_diff + temp_2 * cdf_diff).sum()

        return float(score)


class MaxProfitScorePiecewiseUniform(SympyFnPickleMixin):
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

        # expression = sympy.simplify(profit_function)
        # expression = sympy.factor(profit_function)
        collected = sympy.collect(sympy.factor(profit_function, random_symbol), random_symbol, evaluate=False)
        self.coefficient_eq = collected.get(random_symbol, 0)
        self.coefficient_fn = _safe_lambdify(self.coefficient_eq)
        self.intercept_eq = collected.get(1, 0)
        self.intercept_fn = _safe_lambdify(self.intercept_eq)

        # self.integrand = profit_function * density(random_symbol).pdf(random_symbol)

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
        lower_bounds = np.asarray(bounds[:-1])
        upper_bounds = np.asarray(bounds[1:])
        max_val = float(upper_bound)
        min_val = float(lower_bound)

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
        score = (
            1
            / (max_val - min_val)
            * (coefficient / 2 * (upper_bounds**2 - lower_bounds**2) + intercept * (upper_bounds - lower_bounds)).sum()
        )

        return float(score)


class MaxProfitScorePiecewiseNormal(SympyFnPickleMixin):
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
        bounds = [float(bound) for bound in bounds]

        dist_params_list = list(distribution_parameters.values())
        mu = dist_params_list[0]
        sigma = dist_params_list[1]

        bounds = np.asarray(bounds)
        z = (bounds - mu) / sigma
        Phi = st.norm.cdf(z)
        phi = st.norm.pdf(z)

        Phi_diffs = np.diff(Phi)  # Φ(b) - Φ(a)
        phi_diffs = np.diff(phi)  # φ(b) - φ(a)

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
        score = ((coefficient * mu + intercept) * Phi_diffs - coefficient * sigma * phi_diffs).sum()

        return float(score)
