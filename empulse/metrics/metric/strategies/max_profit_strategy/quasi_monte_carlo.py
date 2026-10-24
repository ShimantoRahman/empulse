from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import sympy
from scipy.stats.qmc import Sobol
from sympy.stats import pspace

from ....._types import FloatNDArray
from ..._distributions import FrozenDistribution, frozen_distribution
from ..._symbolic import _subs_by_name
from .common import (
    _build_rate_function,
    _distribution_parameter_symbols,
    _evaluate_sampled_integrands,
    _HullScoreFunction,
    _LowerChainGate,
    _SampledIntegrand,
    extract_distribution_parameters,
)


class MaxProfitScoreQuasiMonteCarlo(_HullScoreFunction):
    """
    Compute the maximum profit for one or more stochastic variables using Quasi Monte Carlo (QMC) integration.

    This method is less accurate than quad integration but faster for many stochastic variables.
    The QMC method is preferred over the MC due to better accuracy.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        random_symbols: Sequence[sympy.Symbol],
        deterministic_symbols: Iterable[sympy.Symbol],
        n_mc_samples: int,
        rng: np.random.Generator,
    ) -> None:
        self.profit_function = profit_function
        self.random_symbols = random_symbols
        self.deterministic_symbols = deterministic_symbols
        self.n_mc_samples = n_mc_samples
        self._profit_integrand = _SampledIntegrand(profit_function, random_symbols)
        self._rate_integrand = _SampledIntegrand(_build_rate_function(), random_symbols)
        self._lower_chain_gate = _LowerChainGate(profit_function, random_symbols)

        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        self.distribution_args = [arg for args in distributions_args for arg in args]
        sobol = Sobol(d=len(random_symbols), scramble=True, rng=rng)
        self.sobol_samples = sobol.random(n_mc_samples)
        if not any(arg.free_symbols for arg in self.distribution_args):
            # With fixed distribution parameters, the grid can be computed up front.
            self.param_grid_needs_recompute = False
            self.scipy_distributions: list[FrozenDistribution] | None = [
                frozen_distribution(random_var) for random_var in random_symbols
            ]
            self.param_grid = [dist.ppf(self.sobol_samples[:, i]) for i, dist in enumerate(self.scipy_distributions)]
        else:
            self.scipy_distributions = None
            self.param_grid_needs_recompute = True
            self.param_grid = []
        self.dist_params = _distribution_parameter_symbols(self.distribution_args)
        # Parameters and the grid sampled for them are cached as one tuple, so a concurrent caller
        # can never pair one call's parameters with another call's samples.
        self._grid_cache: tuple[dict[str, Any], list[Any]] | None = None

    def _needs_lower_hull(self, kwargs: dict[str, Any]) -> bool:
        """Whether the lower hull can hold the most profitable vertex at any of the samples."""
        param_grid, parameters = self._grid(dict(kwargs))
        return bool(self._lower_chain_gate.on_samples(parameters, param_grid))

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        return self._evaluate(true_positive_rates, false_positive_rates, positive_class_prior, kwargs, rate=False)

    def _rate_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the optimal rate from the ROC convex hull and the positive class prior, on the same samples."""
        return self._evaluate(true_positive_rates, false_positive_rates, positive_class_prior, kwargs, rate=True)

    def _evaluate(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
        *,
        rate: bool,
    ) -> float:
        param_grid, parameters = self._grid(kwargs)
        return _evaluate_sampled_integrands(
            self._profit_integrand,
            self._rate_integrand if rate else None,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            parameters,
            param_grid,
            self.n_mc_samples,
        )

    def _grid(self, kwargs: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Return the samples for these parameters, and the parameters to evaluate them with."""
        dist_params: dict[str, Any] = {}
        param_grid = self.param_grid
        if self.param_grid_needs_recompute:
            distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
            cached = self._grid_cache
            if cached is not None and cached[0] == distribution_parameters:
                param_grid = cached[1]
            else:
                scipy_distributions = [
                    frozen_distribution(_subs_by_name(random_var, distribution_parameters))
                    for random_var in self.random_symbols
                ]
                param_grid = [dist.ppf(self.sobol_samples[:, i]) for i, dist in enumerate(scipy_distributions)]
                self._grid_cache = (distribution_parameters, param_grid)
            dist_params = distribution_parameters
        return param_grid, {**kwargs, **dist_params}
