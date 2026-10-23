"""
Integrate against a density wherever its mass lies, not only near zero at unit scale.

``scipy.integrate.quad`` maps an infinite interval onto ``(0, 1]`` with ``x = a + (1 - t) / t``, which
spreads its nodes at unit scale around the interval's finite end, or around zero. A density whose
mass sits far from there, such as a Laplace(200, 2), or that is much narrower or wider than one,
such as a Normal(0, 1e-4), is then sampled only where it is negligible: ``quad`` returns about zero
and reports convergence. A long finite interval has the same problem.

Each interval is therefore split around the density's mass, found numerically from the log-density
alone so that any distribution ``quad`` can integrate is supported: a core of ``2 * k`` scales
around the mode, integrated directly, and the tails on either side, integrated in units of the scale
from the edge nearest the mass. Only the density is needed, as for ``quad`` itself; no cumulative
distribution function or its inverse.
"""

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
import sympy
from scipy.integrate import IntegrationWarning, nquad, quad
from scipy.optimize import brentq, minimize_scalar
from sympy.stats import pspace

from ..._compile import _safe_lambdify
from ..._symbolic import _subs_by_name

#: The half-width of the core, in scales. A scale is about one standard deviation of a normal
#: density, so the tails start where a normal density has fallen by a factor of e**50.
_CORE_WIDTH = 10.0

#: The log-density drop that marks one scale on either side of the mode: 0.5 for one standard
#: deviation of a normal density.
_SCALE_DROP = 0.5

#: Breakpoints, in scales from the tail's edge, handed to ``quad`` on a long finite tail.
_TAIL_POINTS = (1.0, 10.0, 100.0, 1000.0)

#: The summed error ``quad`` estimates, relative to the integral, above which a warning is raised,
#: and the absolute error below which it is not, for an integral that is about zero.
_REPORTED_RELATIVE_ERROR = 1e-6
_REPORTED_ABSOLUTE_ERROR = 1e-9


class Anchor(NamedTuple):
    """Where a density's mass lies: its mode and a scale over which it falls off."""

    location: float
    scale: float


def log_density(distribution_pdf: sympy.Expr) -> sympy.Expr:
    """
    Return the logarithm of a density, expanded so that it can be evaluated without overflow.

    A density such as Gamma(100, 1)'s ``x**99 * exp(-x) / gamma(100)`` overflows a float in each
    factor long before the product does. Its expanded logarithm, ``99 * log(x) - x -
    loggamma(100)``, does not, and neither does its exponential once the terms are summed.

    The exponential has to be taken numerically: sympy turns ``exp(99 * log(x) - x)`` straight back
    into ``x**99 * exp(-x)``.
    """
    expanded = sympy.expand_log(sympy.log(distribution_pdf), force=True)
    # log(gamma(a)) and log(beta(a, b)) of a large argument overflow before the logarithm is taken.
    expanded = expanded.replace(
        lambda expression: isinstance(expression, sympy.log) and isinstance(expression.args[0], sympy.gamma),
        lambda expression: sympy.loggamma(expression.args[0].args[0]),
    )
    expanded = expanded.replace(
        lambda expression: isinstance(expression, sympy.log) and isinstance(expression.args[0], sympy.beta),
        lambda expression: (
            sympy.loggamma(expression.args[0].args[0])
            + sympy.loggamma(expression.args[0].args[1])
            - sympy.loggamma(expression.args[0].args[0] + expression.args[0].args[1])
        ),
    )
    # With numeric parameters, a normalisation such as gamma(100) is already an integer of 156 digits,
    # which neither NumPy nor Python's float can take the logarithm of. Its logarithm, as a float, can.
    return expanded.evalf()


class DensityOf:
    """
    The density of one stochastic variable, in a form safe to integrate, and where its mass lies.

    Parameters
    ----------
    random_symbol : sympy.stats.rv.RandomSymbol
        The stochastic variable.
    variable : sympy.Symbol, optional
        The symbol the density is a function of, the random symbol itself by default.
    """

    def __init__(self, random_symbol: sympy.Symbol, variable: sympy.Symbol | None = None) -> None:
        self.random_symbol = random_symbol
        self.variable = random_symbol if variable is None else variable
        pdf = pspace(random_symbol).distribution.pdf(self.variable)
        try:
            log_pdf = log_density(pdf)
        except Exception:  # ruff: ignore[blind-except] (any density sympy cannot take apart keeps its form)
            log_pdf = sympy.log(pdf)
        #: The log-density as an expression in the variable, see :func:`log_density`.
        self.log_expression = log_pdf
        self._parameter_names = {str(symbol) for symbol in log_pdf.free_symbols} - {str(self.variable)}
        # The last parameters, support and anchor, published as one tuple for concurrent callers.
        self._cache: tuple[tuple[Any, ...], Anchor | None] | None = None

    def anchor(self, parameters: dict[str, Any], lower: float, upper: float) -> Anchor | None:
        """Return where the density's mass lies on ``(lower, upper)`` for these parameter values."""
        relevant = {name: value for name, value in parameters.items() if name in self._parameter_names}
        try:
            key: tuple[Any, ...] | None = (
                *sorted((name, float(value)) for name, value in relevant.items()),
                float(lower),
                float(upper),
            )
        except (TypeError, ValueError):
            key = None
        cached = self._cache
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]
        function = _safe_lambdify(_subs_by_name(self.log_expression, relevant), [self.variable], dummify=True)
        anchor = locate_mass(function, lower, upper)
        if key is not None:
            self._cache = (key, anchor)
        return anchor


def _candidate_points(lower: float, upper: float) -> np.ndarray:
    """Points spread over the support at every scale, to look for the mode among."""
    magnitudes = np.geomspace(1e-8, 1e12, 81)
    if np.isfinite(lower) and np.isfinite(upper):
        fractions = np.concatenate([np.geomspace(1e-12, 0.5, 40), np.linspace(0.0, 1.0, 101)[1:-1]])
        fractions = np.concatenate([fractions, 1.0 - fractions])
        return np.asarray(lower + (upper - lower) * np.unique(fractions))
    if np.isfinite(lower):
        return np.asarray(lower + magnitudes)
    if np.isfinite(upper):
        return np.asarray(upper - magnitudes[::-1])
    return np.concatenate([-magnitudes[::-1], [0.0], magnitudes])


def _evaluate(function: Callable[[Any], Any], x: Any) -> np.ndarray:
    with np.errstate(all='ignore'):
        values = np.asarray(function(x), dtype=np.float64)
    values = np.broadcast_to(values, np.shape(x)).copy()
    values[np.isnan(values)] = -np.inf
    return values


def locate_mass(log_density_fn: Callable[[Any], Any], lower: float, upper: float) -> Anchor | None:
    """
    Find the mode of a density on ``(lower, upper)`` and the scale over which it falls off.

    Parameters
    ----------
    log_density_fn : callable
        The log-density, vectorised over its argument.
    lower, upper : float
        The interval to look in; either may be infinite.

    Returns
    -------
    Anchor or None
        None when the density is zero everywhere it was looked at.
    """
    grid = _candidate_points(lower, upper)
    values = _evaluate(log_density_fn, grid)
    best = int(np.argmax(values))
    if not np.isfinite(values[best]):
        return None
    mode, peak = float(grid[best]), float(values[best])
    # The mode lies between the grid points either side of the best one.
    left = float(grid[best - 1]) if best > 0 else (lower if np.isfinite(lower) else mode)
    right = float(grid[best + 1]) if best + 1 < len(grid) else (upper if np.isfinite(upper) else mode)
    if left < right:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = minimize_scalar(
                lambda x: -float(_evaluate(log_density_fn, x)),
                bounds=(left, right),
                method='bounded',
                options={'xatol': max(1e-14 * max(abs(left), abs(right)), 1e-300)},
            )
        if np.isfinite(result.fun) and -result.fun > peak:
            mode, peak = float(result.x), float(-result.fun)

    def drop(x: float) -> float:
        return peak - _SCALE_DROP - float(_evaluate(log_density_fn, x))

    widths = []
    for direction in (-1.0, 1.0):
        edge = lower if direction < 0 else upper
        limit = abs(edge - mode) if np.isfinite(edge) else math.inf
        if limit == 0.0:
            continue
        step = max(abs(mode), 1.0) * 1e-12
        # Shrink below the width of a very narrow peak, then grow past the drop.
        for _ in range(300):
            if drop(mode + direction * min(step, limit)) <= 0.0 or step < 1e-300:
                break
            step /= 16.0
        previous = 0.0
        while step < limit and drop(mode + direction * step) <= 0.0 and step < 1e300:
            previous, step = step, step * 4.0
        if step >= limit:
            # The density does not fall off before the support ends.
            widths.append(limit)
            continue
        try:
            width = brentq(lambda d, sign=direction: drop(mode + sign * d), previous, step, xtol=1e-12 * step)
        except ValueError:
            width = step
        widths.append(width)
    scale = max(widths, default=0.0)
    for edge, direction in ((lower, 1.0), (upper, -1.0)):
        if np.isfinite(edge) and abs(mode - edge) <= 1e-9 * max(1.0, abs(edge)):
            # The density peaks at the edge of its support, maybe without bound (Gamma(0.5, 1)), so its
            # fall-off says nothing about where its mass lies. That is where the mass per unit of
            # log-distance from the edge, ``d * pdf(edge + d)``, peaks.
            mode = edge
            scale = max(scale, _edge_scale(log_density_fn, edge, direction, abs(upper - lower)))
    if not (np.isfinite(scale) and scale > 0.0):
        return None
    return Anchor(mode, scale)


def _edge_scale(log_density_fn: Callable[[Any], Any], edge: float, direction: float, limit: float) -> float:
    """Return the distance from ``edge`` at which ``d * pdf(edge + direction * d)`` peaks."""
    distances = np.geomspace(1e-30 * max(1.0, abs(edge)), min(limit, 1e12 * max(1.0, abs(edge))), 241)
    points = edge + direction * distances
    # Distances too small to move off the edge would evaluate the density at its pole.
    distances, points = distances[points != edge], points[points != edge]
    with np.errstate(divide='ignore'):
        values = np.log(distances) + _evaluate(log_density_fn, points)
    # A pole at the other edge of the support is not a point to anchor on either.
    values[~np.isfinite(values)] = -np.inf
    best = int(np.argmax(values))
    return float(distances[best]) if np.isfinite(values[best]) else 0.0


class _Piece(NamedTuple):
    """A part of an interval, integrated over ``t`` in ``[lower, upper]`` with ``x = origin + step * t``."""

    lower: float
    upper: float
    origin: float
    step: float
    points: tuple[float, ...]


def _pieces(lower: float, upper: float, anchor: Anchor | None) -> list[_Piece]:
    """Split ``(lower, upper)`` into the core around the anchor and the tails on either side of it."""
    if anchor is None:
        return [_Piece(lower, upper, 0.0, 1.0, ())]
    location, scale = anchor
    core_lower = max(lower, location - _CORE_WIDTH * scale)
    core_upper = min(upper, location + _CORE_WIDTH * scale)
    pieces = []
    if core_lower < core_upper:
        points = (location,) if core_lower < location < core_upper else ()
        pieces.append(_Piece(core_lower, core_upper, 0.0, 1.0, points))
    # The tails, from the edge nearest the mass outwards, in units of the scale.
    for edge, end, direction in ((max(lower, core_upper), upper, 1.0), (min(upper, core_lower), lower, -1.0)):
        length = (end - edge) * direction
        if length <= 0.0:
            continue
        reach = length / scale
        tail_points = tuple(point for point in _TAIL_POINTS if point < reach) if np.isfinite(reach) else ()
        pieces.append(_Piece(0.0, reach, edge, direction * scale, tail_points))
    return pieces


def _no_nan(value: Any) -> float:
    """
    Return ``value`` as a float, or zero for NaN.

    Far in a tail, a factor that overflows to infinity, such as ``exp(x / 2)``, meets a density that
    underflowed to zero: their product is NaN, where the integrand holds no mass. A divergent
    integrand gives infinity instead, which is kept.
    """
    value = float(value)
    return 0.0 if math.isnan(value) else value


def anchored_quad(
    function: Callable[[float], float], lower: float, upper: float, anchor: Anchor | None
) -> tuple[float, float]:
    """
    Integrate ``function`` over ``(lower, upper)`` with ``quad``, split around ``anchor``.

    Returns
    -------
    value, error : float
        The integral and ``quad``'s estimate of its absolute error.
    """
    value = error = 0.0
    with warnings.catch_warnings():
        # Collected into the error estimate, which the caller reports once.
        warnings.simplefilter('ignore', IntegrationWarning)
        for piece in _pieces(lower, upper, anchor):
            origin, step = piece.origin, piece.step

            def integrand(t: float, origin: float = origin, step: float = step) -> float:
                return _no_nan(function(origin + step * t)) * abs(step)

            result, estimate = quad(integrand, piece.lower, piece.upper, limit=200, points=piece.points or None)
            value += result
            error += estimate
    return value, error


def anchored_nquad(
    function: Callable[..., float], bounds: Sequence[tuple[float, float]], anchors: Sequence[Anchor | None]
) -> tuple[float, float]:
    """
    Integrate ``function`` over a box with ``nquad``, each axis split around its anchor.

    ``function`` takes one value per axis, in the order of ``bounds``.
    """
    value = error = 0.0
    axes = [_pieces(lower, upper, anchor) for (lower, upper), anchor in zip(bounds, anchors, strict=True)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        for pieces in itertools.product(*axes):
            origins = [piece.origin for piece in pieces]
            steps = [piece.step for piece in pieces]
            jacobian = math.prod(abs(step) for step in steps)

            def integrand(
                *t: float, origins: list[float] = origins, steps: list[float] = steps, jacobian: float = jacobian
            ) -> float:
                return _no_nan(function(*(o + s * u for o, s, u in zip(origins, steps, t, strict=True)))) * jacobian

            ranges = [(piece.lower, piece.upper) for piece in pieces]
            options: list[Any] = [
                {'limit': 100, 'points': piece.points} if piece.points else {'limit': 100} for piece in pieces
            ]
            result, estimate = nquad(integrand, ranges, opts=options)
            value += result
            error += estimate
    return value, error


def warn_if_inaccurate(value: float, error: float) -> None:
    """Warn once when the error ``quad`` estimates is large relative to the integral."""
    if not np.isfinite(value) or error > max(_REPORTED_RELATIVE_ERROR * abs(value), _REPORTED_ABSOLUTE_ERROR):
        warnings.warn(
            f'The numerical integration of the expected maximum profit may be inaccurate: the estimated '
            f'absolute error is {error:.3g} on a value of {value:.6g}. Consider '
            "MaxProfit(integration_method='quasi-monte-carlo').",
            IntegrationWarning,
            stacklevel=2,
        )
