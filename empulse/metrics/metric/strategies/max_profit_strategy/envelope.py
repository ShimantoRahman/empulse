"""
Partition the stochastic variable's support into regions where one operating point is optimal.

For a fixed value ``x`` of the stochastic variable, the profit ``P(t, x)`` is affine in the rates
``(F_0, F_1)``, so the profit-maximising operating point is a vertex of the ROC convex hull. Each
vertex therefore contributes one candidate curve ``P_m(x)``, and the optimal threshold as a function
of ``x`` is the **upper envelope** of those ``M`` curves. The piecewise regions the EMP integrates
over are exactly the intervals on which a single curve is on top.

Computing the envelope needs no assumption about how the optimum moves along the hull. Earlier
versions solved the indifference equation symbolically, sorted every root branch globally and zipped
the sorted bounds against the vertex list; that pairing is only valid while the optimal vertex
traverses the hull monotonically in ``x``, which holds for a profit function linear in ``x`` and
fails from degree two upwards.

Two consequences fall out of the envelope formulation:

* Complex roots need no special handling. A conjugate pair means the two vertices never swap on the
  real line, so they contribute no breakpoint and the dominant one simply wins its whole interval.
* Roots are found numerically, per call, on coefficients that are already numbers. Polynomials of
  degree five and above -- which have no solution in radicals -- are no harder than quadratics.
"""

from collections.abc import Callable
from itertools import pairwise
from typing import Any, NamedTuple, Protocol

import numpy as np
from scipy.optimize import brentq

from ....._types import FloatNDArray, IntNDArray

__all__ = [
    'CallableEnvelope',
    'Partition',
    'PolynomialEnvelope',
    'ProfitEnvelope',
    'partition_support',
]

# Crossings closer together than this (relative to the span they live in) are treated as one.
_MERGE_RTOL = 1e-12

# Number of sub-intervals used to bracket sign changes of a general (non-polynomial) difference.
# Every bracket that changes sign is refined exactly, so the subdivision only decides how many
# crossings between one pair of vertices can be found, not how accurately.
_BRACKET_SUBDIVISIONS = 32

# A root counts as real when its imaginary part is this small next to its real part.
_IMAG_RTOL = 1e-9


class Partition(NamedTuple):
    """The support split into regions, each labelled with the hull vertex that is optimal on it."""

    bounds: list[float]
    """Region edges, ascending, of length ``len(vertex_indices) + 1``."""

    upper_bound: float
    """Upper end of the stochastic variable's support."""

    lower_bound: float
    """Lower end of the stochastic variable's support."""

    tprs: list[float]
    """True positive rate of the optimal vertex on each region."""

    fprs: list[float]
    """False positive rate of the optimal vertex on each region."""

    vertex_indices: IntNDArray
    """Index into the hull arrays of the optimal vertex on each region."""


class ProfitEnvelope(Protocol):
    """The family of candidate profit curves, one per ROC convex-hull vertex."""

    def crossings(self, lower_bound: float, upper_bound: float) -> FloatNDArray:
        """Return every value of the stochastic variable where two adjacent vertices swap rank."""
        ...

    def evaluate(self, x: float) -> FloatNDArray:
        """Return the profit of every hull vertex at ``x``."""
        ...


def _adjacent_pairs(n_vertices: int) -> tuple[IntNDArray, IntNDArray]:
    """Index the adjacent hull-vertex pairs.

    Only adjacent pairs need checking. Hull convexity makes the vertex slopes monotone along the
    hull, so the optimal vertex is a monotone function of the isocline slope; every rank change is
    therefore a swap between neighbours, even when the slope itself moves non-monotonically in the
    stochastic variable. This keeps the work linear in the number of vertices rather than quadratic.
    """
    index = np.arange(n_vertices)
    return index[:-1], index[1:]


def _linear_roots(differences: FloatNDArray) -> FloatNDArray:
    """Solve ``a_1 x + a_0 = 0`` for every hull pair at once; a flat difference yields no root."""
    constant = np.asarray(differences[:, 0], dtype=np.float64)
    slope = np.asarray(differences[:, 1], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(slope != 0.0, -constant / slope, np.nan)


def _quadratic_roots(differences: FloatNDArray) -> FloatNDArray:
    """Solve ``a_2 x^2 + a_1 x + a_0 = 0`` for every hull pair at once.

    Pairs whose quadratic term cancels are linear and are solved as such; a negative discriminant
    yields NaN, which is the complex-root case and contributes no boundary.
    """
    constant = np.asarray(differences[:, 0], dtype=np.float64)
    linear = np.asarray(differences[:, 1], dtype=np.float64)
    quadratic = np.asarray(differences[:, 2], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        discriminant = linear * linear - 4.0 * quadratic * constant
        root_of = np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
        lower = (-linear - root_of) / (2.0 * quadratic)
        upper = (-linear + root_of) / (2.0 * quadratic)
        degenerate = np.where(linear != 0.0, -constant / linear, np.nan)
    is_linear = quadratic == 0.0
    return np.concatenate([np.where(is_linear, degenerate, lower), np.where(is_linear, np.nan, upper)])


def _effective_degree(differences: FloatNDArray) -> IntNDArray:
    """Return each row's highest power with a non-zero coefficient, or -1 if the row is all zero."""
    occupied = differences != 0.0
    powers = np.arange(differences.shape[1])
    return np.where(occupied.any(axis=1), (occupied * powers).max(axis=1), -1)


def _companion_matrices(coefficients: FloatNDArray) -> FloatNDArray:
    """Stack one companion matrix per row, whose eigenvalues are that row's polynomial roots.

    The same construction :func:`numpy.polynomial.polynomial.polycompanion` uses, built for a whole
    batch of polynomials of equal degree so they can be solved in one call.
    """
    count, size = coefficients.shape[0], coefficients.shape[1] - 1
    matrices = np.zeros((count, size, size), dtype=np.float64)
    subdiagonal = np.arange(1, size)
    matrices[:, subdiagonal, subdiagonal - 1] = 1.0
    leading = np.asarray(coefficients[:, -1], dtype=np.float64)[:, None]
    matrices[:, :, -1] = -np.asarray(coefficients[:, :-1], dtype=np.float64) / leading
    return matrices


class PolynomialEnvelope:
    """Candidate curves given as polynomials in the stochastic variable, one per hull vertex."""

    def __init__(self, coefficients: FloatNDArray) -> None:
        """
        Build the envelope from per-vertex polynomial coefficients.

        Parameters
        ----------
        coefficients : ndarray of shape (n_vertices, degree + 1)
            Polynomial coefficients in ascending powers, so ``coefficients[m, k]`` multiplies
            ``x ** k`` in the profit of hull vertex ``m``.
        """
        self.coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))

    def crossings(self, lower_bound: float, upper_bound: float) -> FloatNDArray:
        """Return the real roots, inside the support, of every adjacent-pair difference.

        Degrees one and two are solved in closed form across all hull pairs at once. That is the
        whole job for every metric the package ships, and it avoids building one companion matrix
        per pair for a root that is a single division.
        """
        left, right = _adjacent_pairs(len(self.coefficients))
        differences = self.coefficients[left] - self.coefficients[right]

        # Powers that are zero for *every* pair lower the degree of the whole problem.
        occupied = np.flatnonzero(np.any(differences != 0.0, axis=0))
        if len(occupied) == 0:
            return np.empty(0, dtype=np.float64)
        differences = differences[:, : occupied[-1] + 1]

        degree = differences.shape[1] - 1
        if degree == 0:
            # Every pair differs by a non-zero constant, so no two curves ever cross.
            return np.empty(0, dtype=np.float64)
        if degree == 1:
            roots = _linear_roots(differences)
        elif degree == 2:
            roots = _quadratic_roots(differences)
        else:
            roots = self._roots_by_companion_matrix(differences)
        return roots[np.isfinite(roots) & (roots > lower_bound) & (roots < upper_bound)]

    @staticmethod
    def _roots_by_companion_matrix(differences: FloatNDArray) -> FloatNDArray:
        """Solve every pair's difference at once, for degrees with no closed-form solution.

        The pairs are grouped by their effective degree -- a hull segment along one axis cancels
        the leading term, which would make that pair's companion matrix singular -- and each group
        is solved in a single batched eigenvalue call rather than one call per pair.
        """
        highest = _effective_degree(differences)
        found: list[FloatNDArray] = []
        for degree in np.unique(highest):
            if degree < 1:
                # A constant difference never crosses zero, and an identically zero one means the
                # two vertices tie everywhere; neither contributes a boundary.
                continue
            block = differences[highest == degree][:, : degree + 1]
            roots = np.linalg.eigvals(_companion_matrices(block))
            real = np.where(np.abs(roots.imag) <= _IMAG_RTOL * np.maximum(1.0, np.abs(roots.real)), roots.real, np.nan)
            found.append(real.reshape(-1))
        return np.concatenate(found) if found else np.empty(0, dtype=np.float64)

    def evaluate(self, x: float) -> FloatNDArray:
        """Return the profit of every hull vertex at ``x``."""
        return np.asarray(np.polynomial.polynomial.polyval(x, self.coefficients.T), dtype=np.float64)


class CallableEnvelope:
    """Candidate curves given as an arbitrary function of the rates and the stochastic variable."""

    def __init__(
        self,
        profit_fn: Callable[..., FloatNDArray],
        true_positive_rates: FloatNDArray,
        false_positive_rates: FloatNDArray,
    ) -> None:
        """
        Build the envelope from a general profit function and the hull it is evaluated on.

        Parameters
        ----------
        profit_fn : callable
            Called as ``profit_fn(tpr, fpr, x)``; must accept array input for the rates.
        true_positive_rates, false_positive_rates : ndarray of shape (n_vertices,)
            The ROC convex hull.
        """
        self.profit_fn = profit_fn
        self.true_positive_rates = np.asarray(true_positive_rates, dtype=np.float64)
        self.false_positive_rates = np.asarray(false_positive_rates, dtype=np.float64)

    def crossings(self, lower_bound: float, upper_bound: float) -> FloatNDArray:
        """
        Bracket and refine every sign change of each adjacent-pair difference.

        A general function has no root formula, so the support is swept for sign changes and each
        one is refined with Brent's method. Unbounded ends are mapped through ``arctan`` so the
        sweep covers them in finitely many steps.
        """
        left, right = _adjacent_pairs(len(self.true_positive_rates))
        grid = self._sweep_grid(lower_bound, upper_bound)

        found: list[float] = []
        for i, j in zip(left, right, strict=True):

            def difference(x: FloatNDArray | float, i: int = int(i), j: int = int(j)) -> Any:
                return self.profit_fn(self.true_positive_rates[i], self.false_positive_rates[i], x) - self.profit_fn(
                    self.true_positive_rates[j], self.false_positive_rates[j], x
                )

            # The lambdified profit is a numpy expression, so the whole sweep costs one call.
            with np.errstate(all='ignore'):
                values = np.asarray(difference(grid), dtype=np.float64)
            finite = np.isfinite(values)
            for index in range(len(grid) - 1):
                if not (finite[index] and finite[index + 1]):
                    continue
                if values[index] == 0.0:
                    found.append(float(grid[index]))
                elif values[index] * values[index + 1] < 0.0:
                    found.append(float(brentq(difference, grid[index], grid[index + 1], xtol=1e-14)))
        return np.asarray(found, dtype=np.float64)

    @staticmethod
    def _sweep_grid(lower_bound: float, upper_bound: float) -> FloatNDArray:
        """Points spanning the support, spaced evenly in ``arctan`` so infinite ends are covered."""
        low = np.arctan(lower_bound) if np.isfinite(lower_bound) else -np.pi / 2
        high = np.arctan(upper_bound) if np.isfinite(upper_bound) else np.pi / 2
        # Step strictly inside the support: the endpoints are region edges already, and
        # tan(+-pi/2) is not representable.
        margin = (high - low) * 1e-9
        return np.asarray(np.tan(np.linspace(low + margin, high - margin, _BRACKET_SUBDIVISIONS + 1)))

    def evaluate(self, x: float) -> FloatNDArray:
        """Return the profit of every hull vertex at ``x``."""
        return np.asarray(self.profit_fn(self.true_positive_rates, self.false_positive_rates, x), dtype=np.float64)


def _probe(lower: float, upper: float) -> float:
    """
    Pick a point strictly inside ``(lower, upper)`` at which to rank the candidate curves.

    The interval contains no crossing by construction, so the ranking is the same everywhere in it
    and any interior point will do. Unbounded ends therefore only need *some* point beyond the
    finite edge rather than a distributionally meaningful one.
    """
    if np.isfinite(lower) and np.isfinite(upper):
        return (lower + upper) / 2.0
    if np.isfinite(lower):
        return lower + max(1.0, abs(lower))
    if np.isfinite(upper):
        return upper - max(1.0, abs(upper))
    return 0.0


def _dedupe(points: FloatNDArray, lower: float, upper: float) -> FloatNDArray:
    """Sort the candidate breakpoints and collapse ones that coincide to floating-point precision."""
    if len(points) == 0:
        return points
    points = np.sort(points)
    span = max(
        abs(lower) if np.isfinite(lower) else 0.0,
        abs(upper) if np.isfinite(upper) else 0.0,
        1.0,
    )
    keep = np.concatenate([[True], np.diff(points) > _MERGE_RTOL * span])
    return np.asarray(points[keep], dtype=np.float64)


def partition_support(
    envelope: ProfitEnvelope,
    true_positive_rates: FloatNDArray,
    false_positive_rates: FloatNDArray,
    lower_bound: float,
    upper_bound: float,
) -> Partition:
    """
    Split the support into regions and name the hull vertex that maximises profit on each.

    Parameters
    ----------
    envelope : ProfitEnvelope
        The candidate profit curves, one per ROC convex-hull vertex.
    true_positive_rates, false_positive_rates : ndarray of shape (n_vertices,)
        The ROC convex hull, as returned by :func:`~.common._convex_hull`.
    lower_bound, upper_bound : float
        The support of the stochastic variable; either may be infinite.

    Returns
    -------
    partition : Partition
        Region edges together with the optimal vertex on each region.
    """
    breakpoints = _dedupe(envelope.crossings(lower_bound, upper_bound), lower_bound, upper_bound)
    edges = np.concatenate([[lower_bound], breakpoints, [upper_bound]])

    indices = [int(np.argmax(envelope.evaluate(_probe(low, high)))) for low, high in pairwise(edges)]

    # Neighbouring regions that resolve to the same vertex are one region; the crossing between
    # them was either a tangency or a pair of roots too close together to separate.
    bounds = [float(edges[0])]
    kept: list[int] = []
    for position, index in enumerate(indices):
        if kept and index == kept[-1]:
            bounds[-1] = float(edges[position + 1])
            continue
        kept.append(index)
        bounds.append(float(edges[position + 1]))

    vertex_indices = np.asarray(kept, dtype=np.intp)
    return Partition(
        bounds=bounds,
        upper_bound=float(upper_bound),
        lower_bound=float(lower_bound),
        tprs=np.asarray(true_positive_rates, dtype=np.float64)[vertex_indices].tolist(),
        fprs=np.asarray(false_positive_rates, dtype=np.float64)[vertex_indices].tolist(),
        vertex_indices=vertex_indices,
    )
