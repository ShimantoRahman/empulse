from collections.abc import Iterable
from typing import Any

import numpy as np
import sympy
from sympy.stats import pspace

from ....._types import FloatNDArray
from ..._symbolic import _subs_by_name
from .common import (
    _distribution_parameter_symbols,
    _evaluate_sampled_integrands,
    _HullScoreFunction,
    _SampledIntegrand,
    extract_distribution_parameters,
)
from .quasi_monte_carlo import _scipy_distribution, _sympy_dist_to_scipy


def _control_variates(
    random_symbols: Iterable[sympy.Expr], param_grid: list[Any], distribution_parameters: dict[str, Any]
) -> list[FloatNDArray]:
    """
    Return the samples of each stochastic variable minus its exact mean, for use as control variates.

    Only variables with a SciPy counterpart qualify, and only while its mean and variance are finite
    (SciPy reports ``inf`` or ``nan`` for a moment that does not exist, such as a Pareto mean with
    shape at most 1). sympy.stats's own expectation is not used: for some distributions it returns a
    wrong value (``QuadraticU``) or does not finish (``LogitNormal``), and a wrong mean would bias
    the estimate rather than merely fail to improve it.
    """
    controls: list[FloatNDArray] = []
    for random_symbol, samples in zip(random_symbols, param_grid, strict=True):
        if type(pspace(random_symbol).distribution) not in _sympy_dist_to_scipy:
            continue
        distribution = _scipy_distribution(_subs_by_name(random_symbol, distribution_parameters))
        with np.errstate(all='ignore'):
            mean, variance = float(distribution.mean()), float(distribution.var())
        if np.isfinite(mean) and np.isfinite(variance):
            controls.append(np.asarray(samples, dtype=np.float64) - mean)
    return controls


class MaxProfitScoreMonteCarlo(_HullScoreFunction):
    """
    Compute the maximum profit for one or more stochastic variables using Monte Carlo (MC) integration.

    This method is less accurate than quad integration but faster for many stochastic variables.
    The QMC method is preferred over the MC due to better accuracy.
    This method should only be used if there is no mapping of sympy distributions to scipy distributions.

    The stochastic variables that do have one, with a finite mean and variance, serve as linear
    control variates (see :func:`_control_variates`), which removes most of the sampling error.
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
        self._profit_integrand = _SampledIntegrand(profit_function, random_symbols)
        self._rate_integrand = _SampledIntegrand(rate_function, random_symbols) if rate_function is not None else None

        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        self.distribution_args = [arg for args in distributions_args for arg in args]
        if not any(arg.free_symbols for arg in self.distribution_args):
            self.param_grid_needs_recompute = False
            self.param_grid: list[Any] | None = [
                sympy.stats.sample(random_var, size=(n_mc_samples,), seed=rng) for random_var in random_symbols
            ]
            self.controls = _control_variates(random_symbols, self.param_grid, {})
        else:
            self.param_grid_needs_recompute = True
            self.param_grid = None
            self.controls = []
        self.dist_params = _distribution_parameter_symbols(self.distribution_args)
        # Parameters, the grid sampled for them and its control variates are cached as one tuple, so
        # a concurrent caller can never pair one call's parameters with another call's samples.
        self._grid_cache: tuple[dict[str, Any], list[Any], list[FloatNDArray]] | None = None

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        dist_params: dict[str, Any] = {}
        param_grid = self.param_grid
        controls = self.controls
        if self.param_grid_needs_recompute:
            # distribution parameters of the random variable
            distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
            cached = self._grid_cache
            if cached is not None and cached[0] == distribution_parameters:
                param_grid, controls = cached[1], cached[2]
            else:
                param_grid = [
                    sympy.stats.sample(
                        _subs_by_name(random_var, distribution_parameters), size=(self.n_mc_samples,), seed=self.rng
                    )
                    for random_var in self.random_symbols
                ]
                controls = _control_variates(self.random_symbols, param_grid, distribution_parameters)
                self._grid_cache = (distribution_parameters, param_grid, controls)
            dist_params = distribution_parameters

        assert param_grid is not None
        return _evaluate_sampled_integrands(
            self._profit_integrand,
            self._rate_integrand,
            true_positive_rates,
            false_positive_rates,
            positive_class_prior,
            {**kwargs, **dist_params},
            param_grid,
            self.n_mc_samples,
            controls,
        )
