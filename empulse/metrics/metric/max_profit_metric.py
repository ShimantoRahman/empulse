from collections.abc import Callable, Iterable, Sequence
from itertools import islice, pairwise
from typing import Any

import numpy as np
import scipy
import sympy
from scipy.integrate import dblquad, nquad, quad, tplquad
from scipy.stats._qmc import Sobol
from sympy import solve
from sympy.stats import density, pspace
from sympy.stats.rv import is_random
from sympy.utilities import lambdify

from ..._types import FloatNDArray
from .._convex_hull import _compute_convex_hull
from ..common import classification_threshold
from .common import MetricFn, _check_parameters

_sympy_dist_to_scipy: dict[
    sympy.stats.crv_types.SingleContinuousDistribution | sympy.stats.drv_types.SingleDiscreteDistribution,
    scipy.stats.rv_continuous | scipy.stats.rv_discrete,
] = {
    sympy.stats.crv_types.ArcsinDistribution: scipy.stats.arcsine,
    sympy.stats.crv_types.BetaDistribution: scipy.stats.beta,
    sympy.stats.crv_types.BetaPrimeDistribution: scipy.stats.betaprime,
    sympy.stats.crv_types.ChiDistribution: scipy.stats.chi,
    sympy.stats.crv_types.ChiSquaredDistribution: scipy.stats.chi2,
    # sympy.stats.crv_types.Erlang: scipy.stats.chi2
    # Erlang internally calls GammaDistribution so should be supported
    sympy.stats.crv_types.ExGaussianDistribution: scipy.stats.exponnorm,
    sympy.stats.crv_types.ExponentialDistribution: scipy.stats.expon,
    sympy.stats.crv_types.FDistribution: scipy.stats.f,
    sympy.stats.crv_types.GammaDistribution: scipy.stats.gamma,
    sympy.stats.crv_types.GammaInverseDistribution: scipy.stats.invgamma,
    sympy.stats.crv_types.GompertzDistribution: scipy.stats.gompertz,
    sympy.stats.crv_types.LaplaceDistribution: scipy.stats.laplace,
    sympy.stats.crv_types.LevyDistribution: scipy.stats.levy,
    sympy.stats.crv_types.LogisticDistribution: scipy.stats.logistic,
    sympy.stats.crv_types.LogNormalDistribution: scipy.stats.lognorm,
    sympy.stats.crv_types.LomaxDistribution: scipy.stats.lomax,
    sympy.stats.crv_types.MaxwellDistribution: scipy.stats.maxwell,
    sympy.stats.crv_types.MoyalDistribution: scipy.stats.moyal,
    sympy.stats.crv_types.NakagamiDistribution: scipy.stats.nakagami,
    sympy.stats.crv_types.NormalDistribution: scipy.stats.norm,
    sympy.stats.crv_types.ParetoDistribution: scipy.stats.pareto,
    sympy.stats.crv_types.PowerFunctionDistribution: scipy.stats.powerlaw,
    sympy.stats.crv_types.StudentTDistribution: scipy.stats.t,
    sympy.stats.crv_types.TrapezoidalDistribution: scipy.stats.trapezoid,
    sympy.stats.crv_types.TriangularDistribution: scipy.stats.triang,
    sympy.stats.crv_types.UniformDistribution: scipy.stats.uniform,
    sympy.stats.crv_types.VonMisesDistribution: scipy.stats.vonmises,
    sympy.stats.crv_types.GaussianInverseDistribution: scipy.stats.wald,
}


def _build_max_profit_score(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    random_symbols, deterministic_symbols = _identify_symbols(tp_benefit, tn_benefit, fp_cost, fn_cost)
    n_random = len(random_symbols)

    profit_function = _build_profit_function(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    if n_random == 0:
        max_profit_score = _build_max_profit_deterministic(profit_function, deterministic_symbols)
    else:
        max_profit_score = _build_max_profit_stochastic(
            profit_function,
            random_symbols,
            deterministic_symbols,
            integration_method=integration_method,
            n_mc_samples=n_mc_samples,
            rng=rng,
        )
    return max_profit_score


def _build_max_profit_optimal_threshold(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    random_symbols, deterministic_symbols = _identify_symbols(tp_benefit, tn_benefit, fp_cost, fn_cost)
    n_random = len(random_symbols)

    profit_function = _build_profit_function(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    if n_random == 0:
        optimal_threshold = _build_max_profit_threshold_deterministic(profit_function, deterministic_symbols)
    else:
        optimal_threshold = _build_max_profit_threshold_deterministic(profit_function, deterministic_symbols)
    return optimal_threshold


def _build_max_profit_optimal_rate(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    random_symbols, deterministic_symbols = _identify_symbols(tp_benefit, tn_benefit, fp_cost, fn_cost)
    n_random = len(random_symbols)

    profit_function = _build_profit_function(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    if n_random == 0:
        optimal_rate = _build_max_profit_rate_deterministic(profit_function, deterministic_symbols)
    else:
        optimal_rate = _build_max_profit_rate_deterministic(profit_function, deterministic_symbols)
    return optimal_rate


def _identify_symbols(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    """Identify random and deterministic symbols in the profit function."""
    terms = tp_benefit + tn_benefit + fp_cost + fn_cost
    random_symbols = [symbol for symbol in terms.free_symbols if is_random(symbol)]
    deterministic_symbols = [symbol for symbol in terms.free_symbols if not is_random(symbol)]
    return random_symbols, deterministic_symbols


def _build_profit_function(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
    profit_function = (
        tp_benefit * pos_prior * tpr
        + tn_benefit * neg_prior * (1 - fpr)
        - fn_cost * pos_prior * (1 - tpr)
        - neg_prior * fp_cost * fpr
    )
    return profit_function


def _calculate_profits_deterministic(
    y_true: FloatNDArray, y_score: FloatNDArray, calculate_profit: Callable[..., float], **kwargs: Any
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray, float, float]:
    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0
    tprs, fprs = _compute_convex_hull(y_true, y_score)

    profits = np.zeros_like(tprs)
    for i, (tpr, fpr) in enumerate(zip(tprs, fprs, strict=False)):
        profits[i] = calculate_profit(pi_0=pi0, pi_1=pi1, F_0=tpr, F_1=fpr, **kwargs)

    return profits, tprs, fprs, pi0, pi1


def _build_max_profit_deterministic(
    profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]
) -> MetricFn:
    """Compute the maximum profit for all deterministic variables."""
    calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    @_check_parameters(*deterministic_symbols)
    def score_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        profits, *_ = _calculate_profits_deterministic(y_true, y_score, calculate_profit, **kwargs)
        return float(profits.max())

    return score_function


def _calculate_optimal_rate_deterministic(
    y_true: FloatNDArray, y_score: FloatNDArray, calculate_profit: Callable[..., float], **kwargs: Any
) -> float:
    profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(y_true, y_score, calculate_profit, **kwargs)
    best_index = np.argmax(profits)
    return float(tprs[best_index] * pi0 + fprs[best_index] * pi1)


def _build_max_profit_rate_deterministic(
    profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]
) -> MetricFn:
    """Calculate the optimal predicted positive rate."""
    calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    @_check_parameters(*deterministic_symbols)
    def rate_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        return _calculate_optimal_rate_deterministic(y_true, y_score, calculate_profit, **kwargs)

    return rate_function


def _build_max_profit_threshold_deterministic(
    profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]
) -> MetricFn:
    """Calculate the optimal classification threshold."""
    calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    @_check_parameters(*deterministic_symbols)
    def threshold_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        threshold = _calculate_optimal_rate_deterministic(y_true, y_score, calculate_profit, **kwargs)
        return classification_threshold(y_true, y_score, threshold)

    return threshold_function


def _support_all_distributions(random_symbols: Iterable[sympy.Symbol]) -> bool:
    return all(pspace(r).distribution.__class__ in _sympy_dist_to_scipy for r in random_symbols)


def _build_max_profit_stochastic(
    profit_function: sympy.Expr,
    random_symbols: Sequence[sympy.Symbol],
    deterministic_symbols: Iterable[sympy.Symbol],
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    """Compute the maximum profit for one or more stochastic variables."""
    n_random = len(random_symbols)
    if integration_method == 'auto':
        if n_random == 1:
            return _build_max_profit_stochastic_piecewise(profit_function, random_symbols[0], deterministic_symbols)
        elif n_random == 2:
            return _build_max_profit_stochastic_quad(profit_function, random_symbols, deterministic_symbols)
        elif _support_all_distributions(random_symbols):
            return _build_max_profit_stochastic_qmc(
                profit_function, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
        else:
            return _build_max_profit_stochastic_mc(
                profit_function, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
    elif integration_method == 'quad':
        return _build_max_profit_stochastic_quad(profit_function, random_symbols, deterministic_symbols)
    elif integration_method == 'monte-carlo':
        return _build_max_profit_stochastic_mc(
            profit_function, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    elif integration_method == 'quasi-monte-carlo':
        return _build_max_profit_stochastic_qmc(
            profit_function, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    else:
        raise ValueError(f'Integration method {integration_method} is not supported')


def _build_max_profit_stochastic_piecewise(
    profit_function: sympy.Expr, random_symbol: sympy.Symbol, deterministic_symbols: Iterable[sympy.Symbol]
) -> MetricFn:
    """
    Compute the maximum profit for a single stochastic variable using piecewise integration.

    For each convex hull segment, the bounds of the random variable are computed
    in which that decision threshold is optimal.
    For each segment, the profit is integrated over the bounds of the random variable.
    """
    profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
    bound_eq = solve(profit_function - profit_prime, random_symbol)[0]
    compute_bounds = lambdify(list(bound_eq.free_symbols), bound_eq)

    random_var_bounds = pspace(random_symbol).domain.set.args
    distribution_args = pspace(random_symbol).distribution.args

    integrand = profit_function * density(random_symbol).pdf(random_symbol)

    if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
        dist_params = []
    else:
        dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

    def compute_integral(
        integrand: sympy.Expr,
        lower_bound: float,
        upper_bound: float,
        true_positive_rate: float,
        false_positive_rate: float,
        random_var: sympy.Symbol,
    ) -> float:
        integrand = integrand.subs('F_0', true_positive_rate).subs('F_1', false_positive_rate).evalf()
        if not integrand.free_symbols:  # if the integrand is constant, no need to call quad
            if integrand == 0:  # need this separate path since sometimes upper or lower bound can be infinite
                return 0
            return float(integrand * (upper_bound - lower_bound))
        integrand_fn = lambdify(random_var, integrand)
        result, _ = quad(integrand_fn, lower_bound, upper_bound)
        result: float
        return result

    @_check_parameters(*deterministic_symbols, *dist_params)
    def score_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

        distribution_parameters = {  # distribution parameters of the random variable
            str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
        }
        bounds = []
        for (tpr0, fpr0), (tpr1, fpr1) in islice(
            pairwise(zip(true_positive_rates, false_positive_rates, strict=False)), len(true_positive_rates) - 2
        ):
            bounds.append(
                compute_bounds(
                    F_0=tpr0,
                    F_1=fpr0,
                    F_2=tpr1,
                    F_3=fpr1,
                    pi_0=positive_class_prior,
                    pi_1=negative_class_prior,
                    **kwargs,
                )
            )

        # bounds of the random variable can be parameterized by the user
        # if so substitute the parameters in the bounds with the user provided values
        if isinstance(upper_bound := random_var_bounds[1], sympy.Expr):
            upper_bound = upper_bound.subs(distribution_parameters)
        bounds.append(upper_bound)
        if isinstance(lower_bound := random_var_bounds[0], sympy.Expr):
            lower_bound = lower_bound.subs(distribution_parameters)
        bounds.insert(0, lower_bound)

        integrand_ = (
            integrand.subs(kwargs)
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lower_bound, upper_bound), tpr, fpr in zip(
            pairwise(bounds), true_positive_rates, false_positive_rates, strict=False
        ):
            score += compute_integral(integrand_, lower_bound, upper_bound, tpr, fpr, random_symbol)
        return score

    return score_function


def _build_max_profit_stochastic_quad(
    profit_function: sympy.Expr, random_symbols: Sequence[sympy.Symbol], deterministic_symbols: Iterable[sympy.Symbol]
) -> MetricFn:
    """
    Compute the maximum profit for one or more stochastic variables using quad integration.

    This method is very slow for more than 2 stochastic variables.
    It is recommended to use Quasi Monte Carlo integration for more than 2 stochastic variables.
    """
    n_random = len(random_symbols)
    random_variables_bounds = [pspace(random_symbol).domain.set.args for random_symbol in random_symbols]
    random_variables_bounds = [(lb, up) for (lb, up, *_) in random_variables_bounds]
    distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
    distribution_args = [arg for args in distributions_args for arg in args]

    integrand = profit_function
    for random_symbol in random_symbols:
        integrand *= density(random_symbol).pdf(random_symbol)

    if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
        dist_params = []
    else:
        dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

    def compute_integral(
        integrand: sympy.Expr,
        bounds: Iterable[float],
        true_positive_rates: Iterable[float],
        false_positive_rates: Iterable[float],
        random_variables: Iterable[sympy.Symbol],
    ) -> float:
        integrands = [
            lambdify(random_variables, integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
            for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
        ]

        def integrand_fn(*random_vars: float) -> float:
            return float(max(integrand(*reversed(random_vars)) for integrand in integrands))

        if n_random == 1:
            result, _ = quad(integrand_fn, *bounds)
        elif n_random == 2:
            result, _ = dblquad(integrand_fn, *bounds)
        elif n_random == 3:
            result, _ = tplquad(integrand_fn, *bounds)
        else:
            result, _ = nquad(integrand_fn, *bounds)
        return float(result)

    @_check_parameters(*deterministic_symbols, *dist_params)
    def score_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

        # certain distributions determine the bounds of the integral (e.g., uniform)
        # for those distributions we have to fill in the parameters of the distribution
        distribution_parameters = {str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs}
        bounds = [bound for bounds in random_variables_bounds for bound in bounds]
        bounds = [
            bounds.subs(distribution_parameters) if isinstance(bounds, sympy.Expr) else bounds for bounds in bounds
        ]

        integrand_ = (
            integrand.subs(kwargs)
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        return compute_integral(integrand_, bounds, true_positive_rates, false_positive_rates, random_symbols)

    return score_function


def _build_max_profit_stochastic_mc(
    profit_function: sympy.Expr,
    random_symbols: Sequence[sympy.Symbol],
    deterministic_symbols: Iterable[sympy.Symbol],
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    """
    Compute the maximum profit for one or more stochastic variables using Monte Carlo (MC) integration.

    This method is less accurate than quad integration but faster for many stochastic variables.
    The QMC method is preferred over the MC due to better accuracy.
    This method should only be used if there is no mapping of sympy distributions to scipy distributions.
    """
    distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
    distribution_args = [arg for args in distributions_args for arg in args]
    if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
        param_grid_needs_recompute = False
        param_grid = [sympy.stats.sample(random_var, size=(n_mc_samples,), seed=rng) for random_var in random_symbols]
        dist_params = []
    else:
        cached_dist_params = {str(arg): arg for arg in distribution_args}
        param_grid_needs_recompute = True
        param_grid = None
        dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

    @_check_parameters(*deterministic_symbols, *dist_params)
    def score_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

        nonlocal param_grid
        nonlocal param_grid_needs_recompute
        nonlocal cached_dist_params
        if param_grid_needs_recompute:
            distribution_parameters = {  # distribution parameters of the random variable
                str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
            }
            if cached_dist_params != distribution_parameters:
                cached_dist_params = distribution_parameters
                param_grid = [
                    sympy.stats.sample(random_var.subs(cached_dist_params), size=(n_mc_samples,), seed=rng)
                    for random_var in random_symbols
                ]

            integrand = (
                profit_function.subs(kwargs)
                .subs(cached_dist_params)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
        else:
            integrand = (
                profit_function.subs(kwargs).subs('pi_0', positive_class_prior).subs('pi_1', negative_class_prior)
            )
        integrands = [
            lambdify(random_symbols, integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
            for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
        ]

        results = np.empty((len(integrands), n_mc_samples))
        for i, integrand in enumerate(integrands):
            results[i, :] = integrand(*param_grid)
        result = results.max(axis=0).mean()

        return float(result)

    return score_function


def _build_max_profit_stochastic_qmc(
    profit_function: sympy.Expr,
    random_symbols: Sequence[sympy.Symbol],
    deterministic_symbols: Iterable[sympy.Symbol],
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
    distribution_args = [arg for args in distributions_args for arg in args]
    # Generate a Sobol sequence for QMC sampling
    sobol = Sobol(d=len(random_symbols), scramble=True, seed=rng)
    sobol_samples = sobol.random(n_mc_samples)
    if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
        # If all distribution parameters are fixed, then the param grid can be pre-computed.
        param_grid_needs_recompute = False
        # convert to scipy distributions
        scipy_distributions = [
            _sympy_dist_to_scipy[pspace(random_var).distribution.__class__](*[
                float(arg) for arg in pspace(random_var).distribution.args
            ])
            for random_var in random_symbols
        ]
        param_grid = [dist.ppf(sobol_samples[:, i]) for i, dist in enumerate(scipy_distributions)]
        dist_params = []

    else:
        cached_dist_params = {str(arg): arg for arg in distribution_args}
        scipy_distributions = None
        param_grid_needs_recompute = True
        param_grid = []
        dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

    @_check_parameters(*deterministic_symbols, *dist_params)
    def score_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

        nonlocal param_grid
        nonlocal param_grid_needs_recompute
        nonlocal cached_dist_params
        nonlocal scipy_distributions
        if param_grid_needs_recompute:
            distribution_parameters = {  # distribution parameters of the random variable
                str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
            }
            if cached_dist_params != distribution_parameters:
                cached_dist_params = distribution_parameters
                scipy_distributions = [
                    _sympy_dist_to_scipy[pspace(random_var).distribution.__class__](*[
                        float(arg) for arg in pspace(random_var.subs(cached_dist_params)).distribution.args
                    ])
                    for random_var in random_symbols
                ]
                param_grid = [dist.ppf(sobol_samples[:, i]) for i, dist in enumerate(scipy_distributions)]

            integrand = (
                profit_function.subs(kwargs)
                .subs(cached_dist_params)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
        else:
            integrand = (
                profit_function.subs(kwargs).subs('pi_0', positive_class_prior).subs('pi_1', negative_class_prior)
            )

        integrands = [
            lambdify(random_symbols, integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
            for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
        ]

        results = np.empty((len(integrands), len(param_grid[0])))
        for i, integrand in enumerate(integrands):
            results[i, :] = integrand(*param_grid)
        result = results.max(axis=0).mean()

        return float(result)

    return score_function


def _max_profit_score_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    profit_function = _build_profit_function(
        tp_benefit=-tp_benefit, tn_benefit=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    random_symbols = [symbol for symbol in profit_function.free_symbols if is_random(symbol)]

    if random_symbols:
        integrand = profit_function
        for random_symbol in random_symbols:
            integrand *= density(random_symbol).pdf(random_symbol)
        integral = integrand
        for random_symbol in random_symbols:
            lower_bound, upper_bound = pspace(random_symbol).domain.set.args[:2]
            integral = sympy.Integral(integral, (random_symbol, lower_bound, upper_bound))

        output = latex(integral, mode='plain', order=None)
    else:
        output = latex(profit_function, mode='plain', order=None)
    return f'$\\displaystyle {output}$'
