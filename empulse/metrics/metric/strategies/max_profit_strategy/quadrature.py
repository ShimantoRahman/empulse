import bisect
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import sympy
from sympy.stats import pspace

from ....._types import FloatNDArray
from ..._compile import _safe_lambdify
from ..._symbolic import _subs_by_name
from ._anchoring import Anchor, DensityOf, anchored_nquad, anchored_quad, warn_if_inaccurate
from .common import (
    _distribution_parameter_symbols,
    _HullScoreFunction,
    extract_distribution_parameters,
)


def compute_integral_multiple_quad(
    integrand: Callable[..., float],
    bounds: Sequence[float],
    n_random: int,
    anchors: Sequence[Anchor | None],
) -> float:
    """
    Integrate *integrand* over a box with scipy quadrature.

    *integrand* takes one value per stochastic variable, in the order of *bounds*, which holds each
    variable's lower and upper bound in turn. Each axis is split around its anchor in *anchors*,
    where that variable's mass lies (see :func:`~._anchoring.anchored_nquad`), and a large
    estimated error is reported.
    """
    ranges = list(zip(bounds[::2], bounds[1::2], strict=True))
    if n_random == 1:
        value, error = anchored_quad(integrand, *ranges[0], anchors[0])
    else:
        value, error = anchored_nquad(integrand, ranges, anchors)
    warn_if_inaccurate(value, error)
    return float(value)


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

    Walking the upper hull from ``(0, 0)`` to ``(1, 1)``, the slopes ``dF_1 / dF_0`` of its segments
    increase. A step along segment ``k`` changes ``a * F_0 + b * F_1`` by
    ``dF_0 * (a + slope_k * b)``, which changes sign at most once along the hull:

    - ``a >= 0 >= b``: the function rises while ``slope_k < -a / b`` and falls after, so the maximum
      is where that threshold falls among the slopes, found by bisection.
    - ``a <= 0 <= b``: along the upper hull the function falls, then rises. Its maximum is on the
      lower hull instead, whose slopes decrease from ``(0, 0)`` to ``(1, 1)``: it rises there while
      ``slope_k > -a / b``, which is again found by bisection. Without lower hull vertices (see
      :func:`~empulse.metrics._cy_convex_hull.convex_hull_full`), the lower hull is the diagonal and
      the maximum one of its two ends.
    - ``a, b > 0``: it rises all the way; ``a, b < 0``: it falls all the way.
    """

    def __init__(self, true_positive_rates: FloatNDArray, false_positive_rates: FloatNDArray) -> None:
        tprs = np.asarray(true_positive_rates, dtype=np.float64)
        fprs = np.asarray(false_positive_rates, dtype=np.float64)
        # The upper hull ends at its first (1, 1); any vertices after it are the lower hull's, walked back.
        top = np.flatnonzero((tprs == 1.0) & (fprs == 1.0))
        self.last = int(top[0]) if len(top) else len(tprs) - 1
        upper = slice(0, self.last + 1)
        self.slopes = self._slopes(tprs[upper], fprs[upper], np.maximum)
        # The lower hull from (0, 0) to (1, 1), as indices into the vertices.
        self.lower_indices: list[int] = [0, *range(len(tprs) - 1, self.last, -1), self.last]
        lower_slopes = self._slopes(tprs[self.lower_indices], fprs[self.lower_indices], np.minimum)
        # Negated, so that they increase as `bisect` needs.
        self.negated_lower_slopes = [-slope for slope in lower_slopes]
        self.tprs: list[float] = tprs.tolist()
        self.fprs: list[float] = fprs.tolist()

    @staticmethod
    def _slopes(tprs: FloatNDArray, fprs: FloatNDArray, monotone: np.ufunc) -> list[float]:
        tpr_steps, fpr_steps = np.diff(tprs), np.diff(fprs)
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = np.where(tpr_steps > 0, fpr_steps / tpr_steps, np.inf)
        # Monotone by construction; this only guards the bisection against rounding.
        return monotone.accumulate(slopes).tolist() if len(slopes) else []

    def best_vertex(self, tpr_coefficient: float, fpr_coefficient: float) -> int:
        """Return the index of the first vertex maximizing ``tpr_coefficient * F_0 + fpr_coefficient * F_1``."""
        if tpr_coefficient >= 0 >= fpr_coefficient:
            threshold = -tpr_coefficient / fpr_coefficient if fpr_coefficient < 0 else math.inf
            return bisect.bisect_left(self.slopes, threshold)
        if tpr_coefficient <= 0 <= fpr_coefficient:
            threshold = -tpr_coefficient / fpr_coefficient if fpr_coefficient > 0 else math.inf
            # The number of leading lower-hull segments steeper than the threshold, along which it rises.
            return self.lower_indices[bisect.bisect_left(self.negated_lower_slopes, -threshold)]
        return self.last if tpr_coefficient > 0 else 0


class MaxProfitScoreQuad(_HullScoreFunction):
    """
    Compute the expected maximum profit, or its predicted positive rate, for stochastic variables using quad.

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
        self._densities = [
            DensityOf(random_symbol, placeholder)
            for random_symbol, placeholder in zip(random_symbols, placeholders, strict=True)
        ]
        # The log of the joint density, whose exponential is taken numerically (see DensityOf). evalf
        # folds numeric constants such as the normalization `loggamma(6)` into numbers, which would
        # otherwise be recomputed at every point of the integration.
        joint_density = sympy.Add(*(density_of.log_expression for density_of in self._densities)).evalf()
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
        self._log_density = _safe_lambdify(joint_density, arguments, dummify=True)
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
            float(_subs_by_name(bound, distribution_parameters) if isinstance(bound, sympy.Expr) else bound)
            for bound in bounds
        ]
        anchors = [
            density_of.anchor(distribution_parameters, lower, upper)
            for density_of, lower, upper in zip(self._densities, bounds[::2], bounds[1::2], strict=True)
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
        log_density_fn = self._log_density.func

        def joint_density(*arguments: float) -> Any:
            return np.exp(log_density_fn(*arguments))

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

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return compute_integral_multiple_quad(integrand, bounds, self.n_random, anchors)
