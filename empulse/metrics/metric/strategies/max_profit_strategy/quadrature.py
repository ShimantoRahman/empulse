import bisect
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import sympy
from scipy.integrate import IntegrationWarning, dblquad, nquad, quad, tplquad
from sympy.stats import pspace

from ....._types import FloatNDArray
from ..._compile import _safe_lambdify
from ..._symbolic import _subs_by_name
from .common import (
    _distribution_parameter_symbols,
    _HullScoreFunction,
    extract_distribution_parameters,
)


def compute_integral_multiple_quad(
    integrand: Callable[..., float],
    bounds: Sequence[float],
    n_random: int,
) -> float:
    """
    Integrate *integrand* over a box with scipy quadrature.

    *integrand* takes one value per stochastic variable, in the order of *bounds*, which holds each
    variable's lower and upper bound in turn.
    """

    def reversed_integrand(*random_vars: float) -> float:
        # dblquad and tplquad pass the innermost variable first, the reverse of the order of the bounds.
        return integrand(*reversed(random_vars))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        if n_random == 1:
            result, _ = quad(integrand, *bounds)  # type: ignore[call-overload]
        elif n_random == 2:
            result, _ = dblquad(reversed_integrand, *bounds)  # type: ignore[call-overload]
        elif n_random == 3:
            result, _ = tplquad(reversed_integrand, *bounds)  # type: ignore[call-overload]
        else:
            # nquad expects ranges as a list of (lower, upper) pairs, not a flat unpacked list, and
            # passes the variables in the order of their ranges.
            ranges = list(zip(bounds[::2], bounds[1::2], strict=True))
            result, _ = nquad(integrand, ranges)  # type: ignore[call-overload]
    return float(result)


def _roc_coefficients(expression: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr, sympy.Expr]:
    """Split *expression*, linear in the ROC point, into its constant and its ``F_0`` and ``F_1`` coefficients."""
    tpr, fpr = sympy.symbols('F_0 F_1')
    polynomial = sympy.Poly(sympy.expand(expression), tpr, fpr)
    if polynomial.total_degree() > 1:
        raise NotImplementedError('Quadrature supports profit functions linear in F_0 and F_1 only.')
    return polynomial.coeff_monomial(1), polynomial.coeff_monomial(tpr), polynomial.coeff_monomial(fpr)


class _Hull:
    """
    The vertices of a ROC convex hull, for finding the one that maximizes a linear function of them.

    Walking the hull from ``(0, 0)`` to ``(1, 1)``, the slopes ``dF_1 / dF_0`` of its segments
    increase. A step along segment ``k`` changes ``a * F_0 + b * F_1`` by
    ``dF_0 * (a + slope_k * b)``, which changes sign at most once along the hull:

    - ``a >= 0 >= b``: the function rises while ``slope_k < -a / b`` and falls after, so the maximum
      is where that threshold falls among the slopes, found by bisection.
    - ``a <= 0 <= b``: the function falls, then rises, so the maximum is at one of the two ends.
    - ``a, b > 0``: it rises all the way; ``a, b < 0``: it falls all the way.
    """

    def __init__(self, true_positive_rates: FloatNDArray, false_positive_rates: FloatNDArray) -> None:
        tprs = np.asarray(true_positive_rates, dtype=np.float64)
        fprs = np.asarray(false_positive_rates, dtype=np.float64)
        tpr_steps, fpr_steps = np.diff(tprs), np.diff(fprs)
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.where(tpr_steps > 0, fpr_steps / tpr_steps, np.inf)
        # Increasing by construction; this only guards the bisection against rounding.
        self.slopes: list[float] = np.maximum.accumulate(slopes).tolist()
        self.tprs: list[float] = tprs.tolist()
        self.fprs: list[float] = fprs.tolist()

    def best_vertex(self, tpr_coefficient: float, fpr_coefficient: float) -> int:
        """Return the index of the first vertex maximizing ``tpr_coefficient * F_0 + fpr_coefficient * F_1``."""
        if tpr_coefficient >= 0 >= fpr_coefficient:
            threshold = -tpr_coefficient / fpr_coefficient if fpr_coefficient < 0 else math.inf
            return bisect.bisect_left(self.slopes, threshold)
        last = len(self.tprs) - 1
        if tpr_coefficient <= 0 <= fpr_coefficient:
            first_value = tpr_coefficient * self.tprs[0] + fpr_coefficient * self.fprs[0]
            last_value = tpr_coefficient * self.tprs[last] + fpr_coefficient * self.fprs[last]
            return 0 if first_value >= last_value else last
        return last if tpr_coefficient > 0 else 0


class MaxProfitScoreQuad(_HullScoreFunction):
    """
    Compute the optimal predicted positive rate for one or more stochastic variables using quad integration.

    This method is very slow for more than 2 stochastic variables.
    It is recommended to use Quasi Monte Carlo integration for more than 2 stochastic variables.

    The profit (or rate) function is linear in the ROC point, so it is compiled once as its constant
    and its ``F_0`` and ``F_1`` coefficients, and the joint density separately. At each point of
    the integration the coefficients are evaluated once, and the hull vertex that maximizes the
    profit is found by bisection rather than by evaluating the profit at every vertex.
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
        self.dist_params = _distribution_parameter_symbols(self.distribution_args)

        # Each stochastic variable becomes a plain placeholder: its distribution's density is evaluated
        # at it, and it replaces the random symbol in the profit. A hand-built `ContinuousRV(x, ...)`
        # writes its density in the plain symbol `x`, so the random symbol itself cannot stand in.
        placeholders = [sympy.Dummy(str(random_symbol)) for random_symbol in random_symbols]
        to_placeholder = dict(zip(random_symbols, placeholders, strict=True))
        # evalf folds numeric constants such as the normalization `beta(6, 14)` into numbers, which
        # would otherwise be recomputed at every point of the integration.
        joint_density = sympy.Mul(
            *(
                pspace(random_symbol).distribution.pdf(placeholder)
                for random_symbol, placeholder in zip(random_symbols, placeholders, strict=True)
            )
        ).evalf()
        profit_coefficients = [
            coefficient.xreplace(to_placeholder) for coefficient in _roc_coefficients(profit_function)
        ]
        rate_coefficients = (
            [coefficient.xreplace(to_placeholder) for coefficient in _roc_coefficients(rate_function)]
            if rate_function is not None
            else []
        )
        expressions = (joint_density, *profit_coefficients, *rate_coefficients)
        # The class priors, the deterministic variables and the distribution parameters.
        parameters = sorted(
            {symbol for expression in expressions for symbol in expression.free_symbols} - set(placeholders),
            key=str,
        )
        self.parameter_names = [str(symbol) for symbol in parameters]
        arguments = [*parameters, *placeholders]
        # dummify: a density calls functions, such as `beta(alpha, beta)`, that a parameter can be named after.
        self._density = _safe_lambdify(joint_density, arguments, dummify=True)
        self._profit = tuple(
            _safe_lambdify(coefficient, arguments, dummify=True) for coefficient in profit_coefficients
        )
        self._rate = (
            tuple(_safe_lambdify(coefficient, arguments, dummify=True) for coefficient in rate_coefficients)
            if rate_function is not None
            else None
        )

    def _score_hull(
        self,
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
        positive_class_prior: float,
        kwargs: dict[str, Any],
    ) -> float:
        """Compute the maximum profit from the ROC convex hull and the positive class prior."""
        # certain distributions determine the bounds of the integral (e.g., uniform)
        # for those distributions we have to fill in the parameters of the distribution
        distribution_parameters, kwargs = extract_distribution_parameters(kwargs, self.distribution_args)
        bounds = [bound for bounds in self.random_variables_bounds for bound in bounds]
        bounds = [
            _subs_by_name(bounds, distribution_parameters) if isinstance(bounds, sympy.Expr) else bounds
            for bounds in bounds
        ]

        values = {
            **kwargs,
            **distribution_parameters,
            'pi_0': positive_class_prior,
            'pi_1': 1 - positive_class_prior,
        }
        fixed_arguments = tuple(values[name] for name in self.parameter_names)
        hull = _Hull(true_positive_rates, false_positive_rates)
        best_vertex, tprs, fprs = hull.best_vertex, hull.tprs, hull.fprs
        # The compiled functions themselves, without the pickling wrapper's extra call: the
        # integrand runs hundreds of thousands of times.
        joint_density = self._density.func
        profit_constant, profit_tpr_coefficient, profit_fpr_coefficient = (function.func for function in self._profit)
        rate = tuple(function.func for function in self._rate) if self._rate is not None else None

        def integrand(*random_values: float) -> float:
            arguments = (*fixed_arguments, *random_values)
            tpr_coefficient = profit_tpr_coefficient(*arguments)
            fpr_coefficient = profit_fpr_coefficient(*arguments)
            best = best_vertex(tpr_coefficient, fpr_coefficient)
            if rate is None:  # compute maximum profit
                value = profit_constant(*arguments) + tpr_coefficient * tprs[best] + fpr_coefficient * fprs[best]
            else:  # compute optimal rate
                rate_constant, rate_tpr_coefficient, rate_fpr_coefficient = rate
                value = (
                    rate_constant(*arguments)
                    + rate_tpr_coefficient(*arguments) * tprs[best]
                    + rate_fpr_coefficient(*arguments) * fprs[best]
                )
            return float(value * joint_density(*arguments))

        return compute_integral_multiple_quad(integrand, bounds, self.n_random)
