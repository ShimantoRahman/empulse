import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import sympy
from scipy.integrate import IntegrationWarning, dblquad, nquad, quad, tplquad
from sympy.stats import density, pspace
from sympy.utilities import lambdify

from ....._types import FloatNDArray
from .common import _HullScoreFunction, _substitute_integrand, extract_distribution_parameters


def compute_integral_multiple_quad(
    profit_integrand: sympy.Expr,
    rate_integrand: sympy.Expr | None,
    bounds: Sequence[float],
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        if n_random == 1:
            result, _ = quad(integrand_fn, *bounds)  # type: ignore[call-overload]
        elif n_random == 2:
            result, _ = dblquad(integrand_fn, *bounds)  # type: ignore[call-overload]
        elif n_random == 3:
            result, _ = tplquad(integrand_fn, *bounds)  # type: ignore[call-overload]
        else:
            # nquad expects ranges as a list of (lower, upper) pairs, not a flat unpacked list
            ranges = list(zip(bounds[::2], bounds[1::2], strict=True))
            # Unlike dblquad/tplquad, nquad passes the variables in the order of their ranges, so
            # undo the reversal integrand_fn applies for those two.
            result, _ = nquad(lambda *random_vars: integrand_fn(*reversed(random_vars)), ranges)  # type: ignore[call-overload]
    return float(result)


class MaxProfitScoreQuad(_HullScoreFunction):
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

        if not any(arg.free_symbols for arg in self.distribution_args):
            self.dist_params = []
        else:
            self.dist_params = [arg for arg in self.distribution_args if arg.free_symbols]

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        negative_class_prior = 1 - positive_class_prior

        # certain distributions determine the bounds of the integral (e.g., uniform)
        # for those distributions we have to fill in the parameters of the distribution
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
        bounds = [bound for bounds in self.random_variables_bounds for bound in bounds]
        bounds = [
            bounds.subs(distribution_parameters) if isinstance(bounds, sympy.Expr) else bounds for bounds in bounds
        ]

        profit_integrand_ = _substitute_integrand(
            self.profit_function, kwargs, distribution_parameters, positive_class_prior, negative_class_prior
        )
        rate_integrand_ = (
            _substitute_integrand(
                self.rate_function, kwargs, distribution_parameters, positive_class_prior, negative_class_prior
            )
            if self.rate_function is not None
            else None
        )
        return compute_integral_multiple_quad(
            profit_integrand_,
            rate_integrand_,
            bounds,
            true_positive_rates,
            false_positive_rates,
            self.random_symbols,
            self.n_random,
        )
