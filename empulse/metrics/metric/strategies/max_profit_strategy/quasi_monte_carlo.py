from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import scipy
import sympy
from scipy.stats._qmc import Sobol
from sympy.stats import pspace
from sympy.utilities import lambdify

from ....._types import FloatNDArray, IntNDArray
from ...common import SympyFnPickleMixin, _check_parameters
from .common import _convex_hull, extract_distribution_parameters

FrozenScipyDist = (
    scipy.stats._distn_infrastructure.rv_continuous_frozen | scipy.stats._distn_infrastructure.rv_discrete_frozen
)

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
        sobol = Sobol(d=len(random_symbols), scramble=True, rng=rng)
        self.sobol_samples = sobol.random(n_mc_samples)
        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in self.distribution_args):
            # If all distribution parameters are fixed, then the param grid can be pre-computed.
            self.param_grid_needs_recompute = False
            # convert to scipy distributions
            self.scipy_distributions: list[FrozenScipyDist] | None = []
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

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
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
                self.profit_function
                .subs(kwargs)
                .subs(self.cached_dist_params)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
            if self.rate_function is not None:
                rate_integrand = (
                    self.rate_function
                    .subs(self.cached_dist_params)
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
