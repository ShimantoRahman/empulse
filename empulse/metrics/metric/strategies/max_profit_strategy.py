import sys
from collections.abc import Callable, Iterable, Sequence
from itertools import islice, pairwise
from typing import Any, ClassVar, Literal

import numpy as np
import scipy
import sympy
from scipy.integrate import dblquad, nquad, quad, tplquad
from scipy.stats._qmc import Sobol
from sympy import solve
from sympy.stats import density, pspace
from sympy.stats.rv import is_random
from sympy.utilities import lambdify

from ...._types import FloatNDArray, IntNDArray
from ..._cy_convex_hull import convex_hull
from ...common import classification_threshold
from ..common import MetricFn, RateFn, SympyFnPickleMixin, ThresholdFn, _check_parameters
from .metric_strategy import Direction, MetricStrategy

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


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
    sympy.stats.crv_types.FDistributionDistribution: scipy.stats.f,
    sympy.stats.crv_types.GammaDistribution: scipy.stats.gamma,
    sympy.stats.crv_types.GammaInverseDistribution: scipy.stats.invgamma,
    sympy.stats.crv_types.LaplaceDistribution: scipy.stats.laplace,
    sympy.stats.crv_types.LogisticDistribution: scipy.stats.logistic,
    sympy.stats.crv_types.LogNormalDistribution: scipy.stats.lognorm,
    sympy.stats.crv_types.LomaxDistribution: scipy.stats.lomax,
    sympy.stats.crv_types.MaxwellDistribution: scipy.stats.maxwell,
    sympy.stats.crv_types.MoyalDistribution: scipy.stats.moyal,
    sympy.stats.crv_types.NakagamiDistribution: scipy.stats.nakagami,
    sympy.stats.crv_types.NormalDistribution: scipy.stats.norm,
    sympy.stats.crv_types.PowerFunctionDistribution: scipy.stats.powerlaw,
    sympy.stats.crv_types.StudentTDistribution: scipy.stats.t,
    sympy.stats.crv_types.TrapezoidalDistribution: scipy.stats.trapezoid,
    sympy.stats.crv_types.TriangularDistribution: scipy.stats.triang,
    sympy.stats.crv_types.UniformDistribution: scipy.stats.uniform,
    sympy.stats.crv_types.GaussianInverseDistribution: scipy.stats.invgauss,
}

_sympy_dist_to_scipy_params: dict[
    sympy.stats.crv_types.SingleContinuousDistribution | sympy.stats.drv_types.SingleDiscreteDistribution,
    Callable[..., dict[str, float]],
] = {
    sympy.stats.crv_types.ExponentialDistribution: lambda rate: {'loc': 0, 'scale': 1 / rate},
    sympy.stats.crv_types.GammaDistribution: lambda k, theta: {'a': k, 'scale': theta},
    sympy.stats.crv_types.GammaInverseDistribution: lambda a, b: {'a': a, 'scale': b},
    sympy.stats.crv_types.FDistributionDistribution: lambda d1, d2: {'dfn': d1, 'dfd': d2},
    sympy.stats.crv_types.LomaxDistribution: lambda alpha, lamb: {'c': alpha, 'scale': lamb},
    sympy.stats.crv_types.LogNormalDistribution: lambda mu, sigma: {'s': sigma, 'loc': mu},
    sympy.stats.crv_types.MaxwellDistribution: lambda a: {'loc': 0, 'scale': a},
    sympy.stats.crv_types.NakagamiDistribution: lambda mu, omega: {'nu': mu, 'scale': np.sqrt(omega)},
    sympy.stats.crv_types.TrapezoidalDistribution: lambda a, b, c, d: {
        'loc': a,
        'scale': d - a,
        'c': (b - a) / (d - a),
        'd': (c - a) / (d - a),
    },
    sympy.stats.crv_types.TriangularDistribution: lambda a, b, c: {'loc': a, 'scale': b - a, 'c': (c - a) / (b - a)},
    sympy.stats.crv_types.UniformDistribution: lambda a, b: {'loc': a, 'scale': b - a},
    sympy.stats.crv_types.GaussianInverseDistribution: lambda mu, lamb: {'mu': mu / lamb, 'scale': lamb},
    sympy.stats.crv_types.ExGaussianDistribution: lambda mean, std, rate: {
        'loc': mean,
        'scale': std,
        'K': 1 / (std * rate),
    },
}


def _convex_hull(y_true: IntNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
    return convex_hull(y_true.astype(np.int32), y_score.astype(np.float64))


def _aggregate_instance_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Replace instance-dependent array-like parameter values with their mean (float).

    Leaves scalar parameters unchanged.
    """
    for key, value in list(parameters.items()):
        if isinstance(value, np.ndarray):
            parameters[key] = float(np.mean(value))
    return parameters


class MaxProfit(MetricStrategy):
    """
    Strategy for the Expected Maximum Profit (EMP) metric.

    Parameters
    ----------
    integration_method: {'auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'}, default='auto'
        The integration method to use when the metric has stochastic variables.

        - If ``'auto'``, the integration method is automatically chosen based on the number of stochastic variables,
          balancing accuracy with execution speed.
          For a single stochastic variable, piecewise integration is used. This is the most accurate method.
          For two stochastic variables, 'quad' is used,
          and for more than two stochastic variables, 'quasi-monte-carlo' is used if all distribution are supported.
          Otherwise, 'monte-carlo' is used.
        - If ``'quad'``, the metric is integrated using the quad function from scipy.
          Be careful, as this can be slow for more than 2 stochastic variables.
        - If ``'monte-carlo'``, the metric is integrated using a Monte Carlo simulation.
          The monte-carlo simulation is less accurate but faster than quad for many stochastic variables.
        - If ``'quasi-monte-carlo'``, the metric is integrated using a Quasi Monte Carlo simulation.
          The quasi-monte-carlo simulation is more accurate than monte-carlo but only supports a few distributions
          present in :mod:`sympy:sympy.stats`:

            - :class:`sympy.stats.Arcsin`
            - :class:`sympy.stats.Beta`
            - :class:`sympy.stats.BetaPrime`
            - :class:`sympy.stats.Chi`
            - :class:`sympy.stats.ChiSquared`
            - :class:`sympy.stats.Erlang`
            - :class:`sympy.stats.Exponential`
            - :class:`sympy.stats.ExGaussian`
            - :class:`sympy.stats.F`
            - :class:`sympy.stats.Gamma`
            - :class:`sympy.stats.GammaInverse`
            - :class:`sympy.stats.GaussianInverse`
            - :class:`sympy.stats.Laplace`
            - :class:`sympy.stats.Logistic`
            - :class:`sympy.stats.LogNormal`
            - :class:`sympy.stats.Lomax`
            - :class:`sympy.stats.Normal`
            - :class:`sympy.stats.Maxwell`
            - :class:`sympy.stats.Moyal`
            - :class:`sympy.stats.Nakagami`
            - :class:`sympy.stats.PowerFunction`
            - :class:`sympy.stats.StudentT`
            - :class:`sympy.stats.Trapezoidal`
            - :class:`sympy.stats.Triangular`
            - :class:`sympy.stats.Uniform`

    n_mc_samples_exp: int
        ``2**n_mc_samples_exp`` is the number of (Quasi-) Monte Carlo samples to use when
        ``integration_technique'monte-carlo'``.
        Increasing the number of samples improves the accuracy of the metric estimation but slows down the speed.
        This argument is ignored when the ``integration_technique='quad'``.

    random_state: int | np.random.RandomState | None, default=None
        The random state to use when ``integration_technique='monte-carlo'`` or
        ``integration_technique='quasi-monte-carlo'``.
        Determines the points sampled from the distribution of the stochastic variables.
        This argument is ignored when ``integration_technique='quad'``.
    """

    INTEGRATION_METHODS: ClassVar[list[Literal['auto', 'quad', 'quasi-monte-carlo', 'monte-carlo']]] = [
        'auto',
        'quad',
        'quasi-monte-carlo',
        'monte-carlo',
    ]

    def __init__(
        self,
        integration_method: Literal['auto', 'quad', 'quasi-monte-carlo', 'monte-carlo'] = 'auto',
        n_mc_samples_exp: int = 16,
        random_state: np.random.RandomState | int | None = None,
    ):
        super().__init__(name='max profit', direction=Direction.MAXIMIZE)
        if integration_method not in self.INTEGRATION_METHODS:
            raise ValueError(
                f'Integration method {integration_method} is not supported. '
                f'Supported values are {self.INTEGRATION_METHODS}'
            )
        self.integration_method = integration_method
        self.n_mc_samples: int = 2**n_mc_samples_exp
        if isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState(random_state)

    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""
        self._score_function = _build_max_profit_score(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            integration_method=self.integration_method,
            n_mc_samples=self.n_mc_samples,
            rng=self._rng,
        )
        self._optimal_rate: RateFn = _build_max_profit_optimal_rate(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            integration_method=self.integration_method,
            n_mc_samples=self.n_mc_samples,
            rng=self._rng,
        )
        return self

    def score(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the maximum profit score.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score: float
            The maximum profit score.
        """
        parameters = _aggregate_instance_parameters(parameters)
        return self._score_function(y_true, y_score, **parameters)

    def optimal_threshold(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> float | FloatNDArray:
        """
        Compute the classification threshold(s) to optimize the metric value.

        i.e., the score threshold at which an observation should be classified as positive to optimize the metric.
        For instance-dependent costs and benefits, this will return an array of thresholds, one for each sample.
        For class-dependent costs and benefits, this will return a single threshold value.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_threshold: float | FloatNDArray
            The optimal classification threshold(s).
        """
        rate = self.optimal_rate(y_true, y_score, **parameters)
        return classification_threshold(y_true, y_score, rate)

    def optimal_rate(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the predicted positive rate to optimize the metric value.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_rate: float
            The optimal predicted positive rate.
        """
        parameters = _aggregate_instance_parameters(parameters)
        return self._optimal_rate(y_true, y_score, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _max_profit_score_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(direction={self.direction}'
            f', integration_method={self.integration_method!r}, n_mc_samples={self.n_mc_samples}'
            f', random_state={self._rng})'
        )


def extract_distribution_parameters(
    parameters: dict[str, Any], distribution_args: Iterable[sympy.Symbol]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the distribution parameters from the other parameters."""
    distribution_parameters = {
        str(key): parameters.pop(str(key)) for key in distribution_args if str(key) in parameters
    }
    return distribution_parameters, parameters


def compute_integral_quad(
    integrand: sympy.Expr,
    lower_bound: float,
    upper_bound: float,
    true_positive_rate: float,
    false_positive_rate: float,
    random_var: sympy.Symbol,
) -> float:
    """Compute the integral using scipy quadrature for one stochastic variable."""
    integrand = integrand.subs('F_0', true_positive_rate).subs('F_1', false_positive_rate).evalf()
    if not integrand.free_symbols:  # if the integrand is constant, no need to call quad
        if integrand == 0:  # need this separate path since sometimes upper or lower bound can be infinite
            return 0
        return float(integrand * (upper_bound - lower_bound))
    integrand_fn = lambdify(random_var, integrand)
    result, _ = quad(integrand_fn, lower_bound, upper_bound)
    result: float
    return result


def compute_integral_multiple_quad(
    profit_integrand: sympy.Expr,
    rate_integrand: sympy.Expr,
    bounds: Iterable[float],
    true_positive_rates: Iterable[float],
    false_positive_rates: Iterable[float],
    random_variables: Iterable[sympy.Symbol],
    n_random: int,
) -> float:
    """Compute the integral using scipy quadrature for multiple stochastic variables."""
    profit_integrands = [
        lambdify(random_variables, profit_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
        for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
    ]

    if rate_integrand is None:  # compute maximum profit

        def integrand_fn(*random_vars: float) -> float:
            return float(max(integrand(*reversed(random_vars)) for integrand in profit_integrands))

    else:  # compute optimal rate
        rate_integrands = [
            lambdify(random_variables, rate_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
            for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
        ]

        def integrand_fn(*random_vars: float) -> float:
            best_index = np.argmax([integrand(*reversed(random_vars)) for integrand in profit_integrands])
            return float(rate_integrands[best_index](*reversed(random_vars)))

    if n_random == 1:
        result, _ = quad(integrand_fn, *bounds)
    elif n_random == 2:
        result, _ = dblquad(integrand_fn, *bounds)
    elif n_random == 3:
        result, _ = tplquad(integrand_fn, *bounds)
    else:
        result, _ = nquad(integrand_fn, *bounds)
    return float(result)


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
        max_profit_score = MaxProfitScoreDeterministic(profit_function, deterministic_symbols)
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


def _to_threshold_function(rate_function: RateFn) -> ThresholdFn:
    def threshold_function(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        rate = rate_function(y_true, y_score, **kwargs)
        return classification_threshold(y_true, y_score, rate)

    return threshold_function


def _build_max_profit_optimal_threshold(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> ThresholdFn:
    rate_fn = _build_max_profit_optimal_rate(
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        integration_method=integration_method,
        n_mc_samples=n_mc_samples,
        rng=rng,
    )
    return _to_threshold_function(rate_fn)


def _build_max_profit_optimal_rate(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> RateFn:
    random_symbols, deterministic_symbols = _identify_symbols(tp_benefit, tn_benefit, fp_cost, fn_cost)
    n_random = len(random_symbols)

    profit_function = _build_profit_function(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    rate_function = _build_rate_function()
    if n_random == 0:
        optimal_rate = MaxProfitRateDeterministic(profit_function, deterministic_symbols)
    else:
        optimal_rate = _build_max_profit_rate_stochastic(
            profit_function, rate_function, random_symbols, deterministic_symbols, integration_method, n_mc_samples, rng
        )
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


def _build_rate_function() -> sympy.Expr:
    pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
    return pos_prior * tpr + neg_prior * fpr


def _calculate_profits_deterministic(
    y_true: FloatNDArray, y_score: FloatNDArray, calculate_profit: Callable[..., float], **kwargs: Any
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray, float, float]:
    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0
    tprs, fprs = _convex_hull(y_true, y_score)

    profits = np.zeros_like(tprs)
    for i, (tpr, fpr) in enumerate(zip(tprs, fprs, strict=False)):
        profits[i] = calculate_profit(pi_0=pi0, pi_1=pi1, F_0=tpr, F_1=fpr, **kwargs)

    return profits, tprs, fprs, pi0, pi1


class MaxProfitScoreDeterministic(SympyFnPickleMixin):
    """Compute the maximum profit for all deterministic variables."""

    _sympy_functions: ClassVar[dict[str, str]] = {'calculate_profit': 'profit_function'}

    def __init__(self, profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        profits, *_ = _calculate_profits_deterministic(y_true, y_score, self.calculate_profit, **kwargs)
        return float(profits.max())


def _calculate_optimal_rate_deterministic(
    y_true: FloatNDArray, y_score: FloatNDArray, calculate_profit: Callable[..., float], **kwargs: Any
) -> float:
    profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(y_true, y_score, calculate_profit, **kwargs)
    best_index = np.argmax(profits)
    return float(tprs[best_index] * pi0 + fprs[best_index] * pi1)


class MaxProfitRateDeterministic(SympyFnPickleMixin):
    """Compute the maximum profit for all deterministic variables."""

    _sympy_functions: ClassVar[dict[str, str]] = {'calculate_profit': 'profit_function'}

    def __init__(self, profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        return _calculate_optimal_rate_deterministic(y_true, y_score, self.calculate_profit, **kwargs)


def _build_max_profit_rate_stochastic(
    profit_function: sympy.Expr,
    rate_function: sympy.Expr,
    random_symbols: Sequence[sympy.Symbol],
    deterministic_symbols: Iterable[sympy.Symbol],
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> RateFn:
    """Compute the maximum profit for one or more stochastic variables."""
    n_random = len(random_symbols)
    if integration_method == 'auto':
        if n_random == 1:
            return MaxProfitRatePiecewise(profit_function, rate_function, random_symbols[0], deterministic_symbols)
        elif n_random == 2:
            return MaxProfitScoreQuad(profit_function, rate_function, random_symbols, deterministic_symbols)
        elif _support_all_distributions(random_symbols):
            return MaxProfitScoreQuasiMonteCarlo(
                profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
        else:
            return MaxProfitScoreMonteCarlo(
                profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
    elif integration_method == 'quad':
        return MaxProfitScoreQuad(profit_function, rate_function, random_symbols, deterministic_symbols)
    elif integration_method == 'monte-carlo':
        return MaxProfitScoreMonteCarlo(
            profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    elif integration_method == 'quasi-monte-carlo':
        return MaxProfitScoreQuasiMonteCarlo(
            profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    else:
        raise ValueError(f'Integration method {integration_method} is not supported')


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
    # if so substitute the parameters in the bounds with the user-provided values
    if isinstance(upper_bound := random_var_bounds[1], sympy.Expr):
        upper_bound = upper_bound.subs(distribution_parameters)
    bounds.append(upper_bound)
    if isinstance(lower_bound := random_var_bounds[0], sympy.Expr):
        lower_bound = lower_bound.subs(distribution_parameters)
    bounds.insert(0, lower_bound)
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

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
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
            self.integrand.subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        score = 0.0
        for (lower_bound, upper_bound), tpr, fpr in zip(
            pairwise(bounds), true_positive_rates, false_positive_rates, strict=False
        ):
            score += compute_integral_quad(integrand_, lower_bound, upper_bound, tpr, fpr, self.random_symbol)
        return score


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
            return MaxProfitScorePiecewise(profit_function, random_symbols[0], deterministic_symbols)
        elif n_random == 2:
            return MaxProfitScoreQuad(profit_function, None, random_symbols, deterministic_symbols)
        elif _support_all_distributions(random_symbols):
            return MaxProfitScoreQuasiMonteCarlo(
                profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
        else:
            return MaxProfitScoreMonteCarlo(
                profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
    elif integration_method == 'quad':
        return MaxProfitScoreQuad(profit_function, None, random_symbols, deterministic_symbols)
    elif integration_method == 'monte-carlo':
        return MaxProfitScoreMonteCarlo(profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng)
    elif integration_method == 'quasi-monte-carlo':
        return MaxProfitScoreQuasiMonteCarlo(
            profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    else:
        raise ValueError(f'Integration method {integration_method} is not supported')


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

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
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
            self.integrand.subs(kwargs)
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


class MaxProfitScoreQuad(SympyFnPickleMixin):
    """
    Compute the optimal predicted positive rate for one or more stochastic variables using quad integration.

    This method is very slow for more than 2 stochastic variables.
    It is recommended to use Quasi Monte Carlo integration for more than 2 stochastic variables.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr | None,
        random_symbols: Sequence[sympy.Symbol],
        deterministic_symbols: Iterable[sympy.Symbol],
    ) -> None:
        self.profit_function = profit_function
        self.rate_function = rate_function
        self.random_symbols = random_symbols
        self.deterministic_symbols = deterministic_symbols

        self.n_random = len(random_symbols)
        self.random_variables_bounds = [pspace(random_symbol).domain.set.args for random_symbol in random_symbols]
        self.random_variables_bounds = [(lb, up) for (lb, up, *_) in self.random_variables_bounds]
        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        self.distribution_args = [arg for args in distributions_args for arg in args]

        for random_symbol in random_symbols:
            self.profit_function *= density(random_symbol).pdf(random_symbol)
        if self.rate_function is not None:
            for random_symbol in random_symbols:
                self.rate_function *= density(random_symbol).pdf(random_symbol)

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            self.dist_params = []
        else:
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        # certain distributions determine the bounds of the integral (e.g., uniform)
        # for those distributions we have to fill in the parameters of the distribution
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
        bounds = [bound for bounds in self.random_variables_bounds for bound in bounds]
        bounds = [
            bounds.subs(distribution_parameters) if isinstance(bounds, sympy.Expr) else bounds for bounds in bounds
        ]

        profit_integrand_ = (
            self.profit_function.subs(kwargs)
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        if self.rate_function is not None:
            rate_integrand_ = (
                self.rate_function.subs(kwargs)
                .subs(distribution_parameters)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
        else:
            rate_integrand_ = None
        return compute_integral_multiple_quad(
            profit_integrand_,
            rate_integrand_,
            bounds,
            true_positive_rates,
            false_positive_rates,
            self.random_symbols,
            self.n_random,
        )


class MaxProfitScoreMonteCarlo(SympyFnPickleMixin):
    """
    Compute the maximum profit for one or more stochastic variables using Monte Carlo (MC) integration.

    This method is less accurate than quad integration but faster for many stochastic variables.
    The QMC method is preferred over the MC due to better accuracy.
    This method should only be used if there is no mapping of sympy distributions to scipy distributions.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr | None,
        random_symbols: Iterable[sympy.Symbol],
        deterministic_symbols: Iterable[sympy.Symbol],
        n_mc_samples: int,
        rng: np.random.RandomState,
    ) -> None:
        self.profit_function = profit_function
        self.rate_function = rate_function
        self.random_symbols = random_symbols
        self.deterministic_symbols = deterministic_symbols
        self.n_mc_samples = n_mc_samples
        self.rng = rng

        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        self.distribution_args = [arg for args in distributions_args for arg in args]
        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            self.param_grid_needs_recompute = False
            self.param_grid = [
                sympy.stats.sample(random_var, size=(n_mc_samples,), seed=rng) for random_var in random_symbols
            ]
            self.dist_params = []
        else:
            self.cached_dist_params = {str(arg): arg for arg in self.distribution_args}
            self.param_grid_needs_recompute = True
            self.param_grid = None
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        if self.param_grid_needs_recompute:
            # distribution parameters of the random variable
            distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
            if self.cached_dist_params != distribution_parameters:
                self.cached_dist_params = distribution_parameters
                self.param_grid = [
                    sympy.stats.sample(
                        random_var.subs(self.cached_dist_params), size=(self.n_mc_samples,), seed=self.rng
                    )
                    for random_var in self.random_symbols
                ]

            profit_integrand = (
                self.profit_function.subs(kwargs)
                .subs(self.cached_dist_params)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
            if self.rate_function is not None:
                rate_integrand = (
                    self.rate_function.subs(self.cached_dist_params)
                    .subs('pi_0', positive_class_prior)
                    .subs('pi_1', negative_class_prior)
                )
        else:
            profit_integrand = (
                self.profit_function.subs(kwargs).subs('pi_0', positive_class_prior).subs('pi_1', negative_class_prior)
            )
            if self.rate_function is not None:
                rate_integrand = self.rate_function.subs('pi_0', positive_class_prior).subs(
                    'pi_1', negative_class_prior
                )
        profit_integrands = [
            lambdify(self.random_symbols, profit_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
            for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
        ]

        results = np.empty((len(profit_integrands), self.n_mc_samples))
        for i, integrand in enumerate(profit_integrands):
            results[i, :] = integrand(*self.param_grid)
        if self.rate_function is None:
            result = results.max(axis=0).mean()
        else:
            rate_integrands = [
                lambdify(self.random_symbols, rate_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
                for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
            ]
            rate_results = np.empty((len(profit_integrands), self.n_mc_samples))
            best_indices = results.argmax(axis=0)
            for i, integrand in enumerate(rate_integrands):
                rate_results[i, :] = integrand(*self.param_grid)
            result = float(rate_results[best_indices, np.arange(self.n_mc_samples)].mean())

        return float(result)


class MaxProfitScoreQuasiMonteCarlo(SympyFnPickleMixin):
    """
    Compute the maximum profit for one or more stochastic variables using Quasi Monte Carlo (QMC) integration.

    This method is less accurate than quad integration but faster for many stochastic variables.
    The QMC method is preferred over the MC due to better accuracy.
    """

    _sympy_functions: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr | None,
        random_symbols: Sequence[sympy.Symbol],
        deterministic_symbols: Iterable[sympy.Symbol],
        n_mc_samples: int,
        rng: np.random.RandomState,
    ) -> None:
        self.profit_function = profit_function
        self.rate_function = rate_function
        self.random_symbols = random_symbols
        self.deterministic_symbols = deterministic_symbols
        self.n_mc_samples = n_mc_samples
        self.rng = rng

        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        self.distribution_args = [arg for args in distributions_args for arg in args]
        # Generate a Sobol sequence for QMC sampling
        sobol = Sobol(d=len(random_symbols), scramble=True, seed=rng)
        self.sobol_samples = sobol.random(n_mc_samples)
        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            # If all distribution parameters are fixed, then the param grid can be pre-computed.
            self.param_grid_needs_recompute = False
            # convert to scipy distributions
            self.scipy_distributions = []
            for random_var in random_symbols:
                sympy_distribution = pspace(random_var).distribution.__class__
                scipy_distribution = _sympy_dist_to_scipy[sympy_distribution]
                sympy_dist_params = [float(arg) for arg in pspace(random_var).distribution.args]
                if sympy_distribution in _sympy_dist_to_scipy_params:
                    scipy_dist_kwargs = _sympy_dist_to_scipy_params[sympy_distribution](*sympy_dist_params)
                    self.scipy_distributions.append(scipy_distribution(**scipy_dist_kwargs))
                else:
                    scipy_dist_params = sympy_dist_params
                    self.scipy_distributions.append(scipy_distribution(*scipy_dist_params))
            self.param_grid = [dist.ppf(self.sobol_samples[:, i]) for i, dist in enumerate(self.scipy_distributions)]
            self.dist_params = []

        else:
            self.cached_dist_params = {str(arg): arg for arg in self.distribution_args}
            self.scipy_distributions = None
            self.param_grid_needs_recompute = True
            self.param_grid = []
            self.dist_params = [
                arg for arg in self.distribution_args if not isinstance(arg, sympy.core.numbers.Integer)
            ]

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        if self.param_grid_needs_recompute:
            # distribution parameters of the random variable
            distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
            if self.cached_dist_params != distribution_parameters:
                self.cached_dist_params = distribution_parameters
                scipy_distributions = []
                for random_var in self.random_symbols:
                    sympy_distribution = pspace(random_var).distribution.__class__
                    scipy_distribution = _sympy_dist_to_scipy[sympy_distribution]
                    sympy_dist_params = [
                        float(arg) for arg in pspace(random_var.subs(self.cached_dist_params)).distribution.args
                    ]
                    if sympy_distribution in _sympy_dist_to_scipy_params:
                        scipy_dist_kwargs = _sympy_dist_to_scipy_params[sympy_distribution](*sympy_dist_params)
                        scipy_distributions.append(scipy_distribution(**scipy_dist_kwargs))
                    else:
                        scipy_dist_params = sympy_dist_params
                        scipy_distributions.append(scipy_distribution(*scipy_dist_params))
                self.param_grid = [dist.ppf(self.sobol_samples[:, i]) for i, dist in enumerate(scipy_distributions)]

            profit_integrand = (
                self.profit_function.subs(kwargs)
                .subs(self.cached_dist_params)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
            if self.rate_function is not None:
                rate_integrand = (
                    self.rate_function.subs(self.cached_dist_params)
                    .subs('pi_0', positive_class_prior)
                    .subs('pi_1', negative_class_prior)
                )
        else:
            profit_integrand = (
                self.profit_function.subs(kwargs).subs('pi_0', positive_class_prior).subs('pi_1', negative_class_prior)
            )
            if self.rate_function is not None:
                rate_integrand = self.rate_function.subs('pi_0', positive_class_prior).subs(
                    'pi_1', negative_class_prior
                )

        profit_integrands = [
            lambdify(self.random_symbols, profit_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
            for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
        ]

        results = np.empty((len(profit_integrands), self.n_mc_samples))
        for i, integrand in enumerate(profit_integrands):
            results[i, :] = integrand(*self.param_grid)
        if self.rate_function is None:
            result = float(results.max(axis=0).mean())
        else:
            rate_integrands = [
                lambdify(self.random_symbols, rate_integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
                for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
            ]
            rate_results = np.empty((len(profit_integrands), self.n_mc_samples))
            best_indices = results.argmax(axis=0)
            for i, integrand in enumerate(rate_integrands):
                rate_results[i, :] = integrand(*self.param_grid)
            result = float(rate_results[best_indices, np.arange(self.n_mc_samples)].mean())

        return result


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
