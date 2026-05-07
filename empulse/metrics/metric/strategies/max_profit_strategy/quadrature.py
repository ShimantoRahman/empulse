from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import sympy
from scipy.integrate import dblquad, nquad, quad, tplquad
from sympy.stats import density, pspace
from sympy.utilities import lambdify

from ....._types import FloatNDArray, IntNDArray
from ...common import _check_parameters
from .common import _convex_hull, extract_distribution_parameters


def compute_integral_multiple_quad(
    profit_integrand: sympy.Expr,
    rate_integrand: sympy.Expr | None,
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
        result, _ = quad(integrand_fn, *bounds)  # type: ignore[call-overload]
    elif n_random == 2:
        result, _ = dblquad(integrand_fn, *bounds)  # type: ignore[call-overload]
    elif n_random == 3:
        result, _ = tplquad(integrand_fn, *bounds)  # type: ignore[call-overload]
    else:
        result, _ = nquad(integrand_fn, *bounds)  # type: ignore[call-overload]
    return float(result)


class MaxProfitScoreQuad:
    """
    Compute the optimal predicted positive rate for one or more stochastic variables using quad integration.

    This method is very slow for more than 2 stochastic variables.
    It is recommended to use Quasi Monte Carlo integration for more than 2 stochastic variables.
    """

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

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
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
            self.profit_function
            .subs(kwargs)
            .subs(distribution_parameters)
            .subs('pi_0', positive_class_prior)
            .subs('pi_1', negative_class_prior)
        )
        if self.rate_function is not None:
            rate_integrand_ = (
                self.rate_function
                .subs(kwargs)
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
