from collections.abc import Iterable
from typing import Any

import numpy as np
import sympy
from sympy.stats import pspace

from ....._types import FloatNDArray, IntNDArray
from ...common import _check_parameters
from .common import _convex_hull, _evaluate_sampled_integrands, _substitute_integrand, extract_distribution_parameters


class MaxProfitScoreMonteCarlo:
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
            self.cached_dist_params = {str(arg): arg for arg in self.distribution_args}
            self.param_grid_needs_recompute = True
            self.param_grid = None
            self.dist_params = [arg for arg in self.distribution_args if arg.free_symbols]

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the maximum profit."""
        _check_parameters((*self.deterministic_symbols, *self.dist_params), kwargs)

        positive_class_prior = float(np.mean(y_true))
        negative_class_prior = 1 - positive_class_prior
        true_positive_rates, false_positive_rates = _convex_hull(y_true, y_score)

        dist_params: dict[str, Any] = {}
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
            dist_params = self.cached_dist_params

        profit_integrand = _substitute_integrand(
            self.profit_function, kwargs, dist_params, positive_class_prior, negative_class_prior
        )
        rate_integrand = (
            _substitute_integrand(self.rate_function, kwargs, dist_params, positive_class_prior, negative_class_prior)
            if self.rate_function is not None
            else None
        )

        assert self.param_grid is not None  # populated above whenever param_grid_needs_recompute, else at __init__
        return _evaluate_sampled_integrands(
            profit_integrand,
            rate_integrand,
            true_positive_rates,
            false_positive_rates,
            self.random_symbols,
            self.param_grid,
            self.n_mc_samples,
        )
