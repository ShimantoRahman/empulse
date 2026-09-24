from collections.abc import Iterable
from typing import Any

import numpy as np
import sympy
from sympy.stats import pspace

from ....._types import FloatNDArray
from ..._symbolic import _subs_by_name
from .common import (
    _evaluate_sampled_integrands,
    _HullScoreFunction,
    _substitute_integrand,
    extract_distribution_parameters,
)


class MaxProfitScoreMonteCarlo(_HullScoreFunction):
    """
    Compute the maximum profit for one or more stochastic variables using Monte Carlo (MC) integration.

    This method is less accurate than quad integration but faster for many stochastic variables.
    The QMC method is preferred over the MC due to better accuracy.
    This method should only be used if there is no mapping of sympy distributions to scipy distributions.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        rate_function: sympy.Expr | None,
        random_symbols: Iterable[sympy.Symbol],
        deterministic_symbols: Iterable[sympy.Symbol],
        n_mc_samples: int,
        rng: np.random.Generator,
    ) -> None:
        self.profit_function = profit_function
        self.rate_function = rate_function
        self.random_symbols = random_symbols
        self.deterministic_symbols = deterministic_symbols
        self.n_mc_samples = n_mc_samples
        self.rng = rng

        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        self.distribution_args = [arg for args in distributions_args for arg in args]
        if not any(arg.free_symbols for arg in self.distribution_args):
            self.param_grid_needs_recompute = False
            self.param_grid: list[Any] | None = [
                sympy.stats.sample(random_var, size=(n_mc_samples,), seed=rng) for random_var in random_symbols
            ]
            self.dist_params = []
        else:
            self.param_grid_needs_recompute = True
            self.param_grid = None
            self.dist_params = [arg for arg in self.distribution_args if arg.free_symbols]
        # Parameters and the grid sampled for them are cached as one tuple, so a concurrent caller
        # can never pair one call's parameters with another call's samples.
        self._grid_cache: tuple[dict[str, Any], list[Any]] | None = None

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        negative_class_prior = 1 - positive_class_prior

        dist_params: dict[str, Any] = {}
        param_grid = self.param_grid
        if self.param_grid_needs_recompute:
            # distribution parameters of the random variable
            distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
            cached = self._grid_cache
            if cached is not None and cached[0] == distribution_parameters:
                param_grid = cached[1]
            else:
                param_grid = [
                    sympy.stats.sample(
                        _subs_by_name(random_var, distribution_parameters), size=(self.n_mc_samples,), seed=self.rng
                    )
                    for random_var in self.random_symbols
                ]
                self._grid_cache = (distribution_parameters, param_grid)
            dist_params = distribution_parameters

        profit_integrand = _substitute_integrand(
            self.profit_function, kwargs, dist_params, positive_class_prior, negative_class_prior
        )
        rate_integrand = (
            _substitute_integrand(self.rate_function, kwargs, dist_params, positive_class_prior, negative_class_prior)
            if self.rate_function is not None
            else None
        )

        assert param_grid is not None
        return _evaluate_sampled_integrands(
            profit_integrand,
            rate_integrand,
            true_positive_rates,
            false_positive_rates,
            self.random_symbols,
            param_grid,
            self.n_mc_samples,
        )
