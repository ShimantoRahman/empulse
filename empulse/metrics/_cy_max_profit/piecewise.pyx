# distutils: language = c++
"""
The expected maximum profit of a ROC convex hull, for a profit polynomial in one stochastic variable.

For each value ``x`` of the stochastic variable, the profit of hull vertex ``m`` is the polynomial
``P_m(x) = sum_k a_k(m) x**k``, whose coefficients are affine in the vertex's rates:
``a_k(m) = constant[k] + tpr_slope[k] * tpr[m] + fpr_slope[k] * fpr[m]``. The expected maximum
profit integrates the upper envelope of these curves against the density of ``x``: the support is
split into the regions on which a single vertex is on top, and on each region every power of ``x``
integrates to a closed-form partial moment of the distribution.

This is the compiled counterpart of ``partition_support`` with a ``PolynomialEnvelope`` followed by
a ``BaseMaxProfitScorePiecewise._integrate`` in ``strategies/max_profit_strategy``, and follows them
step by step, so that both give the same regions and the same score. It covers polynomials up to
degree two, whose crossings have closed-form roots; the Python code handles the rest.
"""

from libc.math cimport exp, expm1, isfinite, isnan, log, pow, sqrt
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from scipy.special.cython_special cimport betainc, chdtr, gamma, gammainc, ndtr

# Crossings closer together than this (relative to the span they live in) are treated as one.
cdef double _MERGE_RTOL = 1e-12
cdef double _NAN = float('nan')
cdef double _INF = float('inf')
cdef double _SQRT_2PI = sqrt(2.0 * 3.14159265358979323846)


cpdef enum Distribution:
    UNIFORM
    NORMAL
    GAMMA
    PARETO
    TRIANGULAR
    EXPONENTIAL
    CHI_SQUARED
    LOG_NORMAL
    BETA
    WEIBULL


# --- The regions -------------------------------------------------------------------------------


cdef inline double _probe(double lower, double upper) noexcept nogil:
    """A point strictly inside (lower, upper), where the candidate curves are ranked."""
    if isfinite(lower) and isfinite(upper):
        return (lower + upper) / 2.0
    if isfinite(lower):
        return lower + max(1.0, abs(lower))
    if isfinite(upper):
        return upper - max(1.0, abs(upper))
    return 0.0


cdef inline void _add_crossing(vector[double]& crossings, double x, double lower, double upper) noexcept nogil:
    if isfinite(x) and x > lower and x < upper:
        crossings.push_back(x)


cdef Py_ssize_t _best_vertex(
    const vector[double]& coefficients, Py_ssize_t n_vertices, Py_ssize_t n_powers, double x
) noexcept nogil:
    """The vertex with the highest profit at x, the first on a tie, or the first NaN, as np.argmax."""
    cdef Py_ssize_t m, k
    cdef Py_ssize_t best = 0
    cdef double best_profit = 0.0, profit
    for m in range(n_vertices):
        # Horner's rule, in the same order as numpy.polynomial.polynomial.polyval.
        profit = coefficients[m * n_powers + n_powers - 1]
        for k in range(n_powers - 2, -1, -1):
            profit = coefficients[m * n_powers + k] + profit * x
        if isnan(profit):
            return m
        if m == 0 or profit > best_profit:
            best = m
            best_profit = profit
    return best


cdef void _partition(
    const vector[double]& coefficients,
    Py_ssize_t n_vertices,
    Py_ssize_t n_powers,
    double lower,
    double upper,
    vector[double]& bounds,
    vector[Py_ssize_t]& vertices,
) noexcept nogil:
    """
    Split (lower, upper) into the regions on which a single vertex has the highest profit.

    Fills ``bounds`` with the region edges, ascending, and ``vertices`` with the optimal hull
    vertex on each region.

    Only adjacent vertices can swap rank, since the hull is convex, so the region edges are among
    the crossings of adjacent pairs of curves. The crossings are sorted, near-duplicates merged, and
    each region between them labelled with the vertex on top at a point inside it. Neighbouring
    regions with the same vertex are one region.
    """
    cdef Py_ssize_t m, k, i, best
    cdef int degree = -1
    cdef double constant, linear, quadratic, discriminant, root_of, span
    cdef vector[double] crossings, edges

    # The highest power at which any adjacent pair differs: powers above it cancel in every pair.
    for k in range(n_powers):
        for m in range(n_vertices - 1):
            if coefficients[m * n_powers + k] - coefficients[(m + 1) * n_powers + k] != 0.0:
                degree = k
                break

    if degree == 1:
        for m in range(n_vertices - 1):
            constant = coefficients[m * n_powers] - coefficients[(m + 1) * n_powers]
            linear = coefficients[m * n_powers + 1] - coefficients[(m + 1) * n_powers + 1]
            if linear != 0.0:
                _add_crossing(crossings, -constant / linear, lower, upper)
    elif degree == 2:
        for m in range(n_vertices - 1):
            constant = coefficients[m * n_powers] - coefficients[(m + 1) * n_powers]
            linear = coefficients[m * n_powers + 1] - coefficients[(m + 1) * n_powers + 1]
            quadratic = coefficients[m * n_powers + 2] - coefficients[(m + 1) * n_powers + 2]
            if quadratic == 0.0:
                if linear != 0.0:
                    _add_crossing(crossings, -constant / linear, lower, upper)
                continue
            discriminant = linear * linear - 4.0 * quadratic * constant
            if discriminant >= 0.0:  # a negative one means the two curves never cross
                root_of = sqrt(discriminant)
                _add_crossing(crossings, (-linear - root_of) / (2.0 * quadratic), lower, upper)
                _add_crossing(crossings, (-linear + root_of) / (2.0 * quadratic), lower, upper)

    sort(crossings.begin(), crossings.end())
    span = max(abs(lower) if isfinite(lower) else 0.0, abs(upper) if isfinite(upper) else 0.0, 1.0)
    edges.push_back(lower)
    for i in range(<Py_ssize_t>crossings.size()):
        if i == 0 or crossings[i] - crossings[i - 1] > _MERGE_RTOL * span:
            edges.push_back(crossings[i])
    edges.push_back(upper)

    bounds.push_back(edges[0])
    for i in range(<Py_ssize_t>edges.size() - 1):
        best = _best_vertex(coefficients, n_vertices, n_powers, _probe(edges[i], edges[i + 1]))
        if vertices.size() > 0 and best == vertices.back():
            bounds[bounds.size() - 1] = edges[i + 1]
        else:
            vertices.push_back(best)
            bounds.push_back(edges[i + 1])


# --- The distributions' CDFs, as scipy.stats computes them --------------------------------------


cdef inline double _standard_cdf_bounds(double y, double lower, double upper, double inside) noexcept nogil:
    """The CDF at y of a distribution supported on (lower, upper), given its value inside."""
    if isnan(y):
        return _NAN
    if y >= upper:
        return 1.0
    if y > lower:
        return inside
    return 0.0


cdef inline double _gamma_cdf(double x, double shape, double scale) noexcept nogil:
    if not (shape > 0.0 and scale > 0.0):
        return _NAN
    cdef double y = x / scale
    return _standard_cdf_bounds(y, 0.0, _INF, gammainc(shape, y) if y > 0.0 and y < _INF else 0.0)


cdef inline double _pareto_cdf(double x, double shape, double scale) noexcept nogil:
    if not (shape > 0.0 and scale > 0.0):
        return _NAN
    cdef double y = x / scale
    return _standard_cdf_bounds(y, 1.0, _INF, 1.0 - pow(y, -shape) if y > 1.0 and y < _INF else 0.0)


cdef inline double _exponential_cdf(double x, double scale) noexcept nogil:
    if not scale > 0.0:
        return _NAN
    cdef double y = x / scale
    return _standard_cdf_bounds(y, 0.0, _INF, -expm1(-y))


cdef inline double _chi_squared_cdf(double x, double df) noexcept nogil:
    if not df > 0.0:
        return _NAN
    return _standard_cdf_bounds(x, 0.0, _INF, chdtr(df, x) if x > 0.0 and x < _INF else 0.0)


cdef inline double _log_normal_cdf(double x, double sigma, double scale) noexcept nogil:
    if not (sigma > 0.0 and scale > 0.0):
        return _NAN
    cdef double y = x / scale
    return _standard_cdf_bounds(y, 0.0, _INF, ndtr(log(y) / sigma) if y > 0.0 and y < _INF else 0.0)


cdef inline double _weibull_cdf(double x, double shape, double scale) noexcept nogil:
    if not (shape > 0.0 and scale > 0.0):
        return _NAN
    cdef double y = x / scale
    return _standard_cdf_bounds(y, 0.0, _INF, -expm1(-pow(y, shape)) if y > 0.0 and y < _INF else 0.0)


cdef inline double _triangular_cdf(double x, double c, double loc, double scale) noexcept nogil:
    if not (c >= 0.0 and c <= 1.0 and scale > 0.0):
        return _NAN
    cdef double y = (x - loc) / scale
    cdef double inside = 0.0
    if y > 0.0 and y < 1.0:
        if c == 0.0:
            inside = 2.0 * y - y * y
        elif y < c:
            inside = y * y / c
        elif c != 1.0:
            inside = (y * y - 2.0 * y + c) / (c - 1.0)
        else:
            inside = y * y
    return _standard_cdf_bounds(y, 0.0, 1.0, inside)


# --- The integrals over the regions -----------------------------------------------------------


cdef inline bint _all_zero(const vector[double]& values) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t>values.size()):
        if values[i] != 0.0:
            return False
    return True


cdef double _integrate_uniform(
    const vector[vector[double]]& a, const vector[double]& bounds, double lower, double upper
) noexcept nogil:
    cdef double pdf = 1.0 / (upper - lower), total = 0.0, term
    cdef Py_ssize_t k, r
    for k in range(<Py_ssize_t>a.size()):
        if _all_zero(a[k]):
            continue
        term = 0.0
        for r in range(<Py_ssize_t>a[k].size()):
            term += (a[k][r] * pdf / (k + 1.0)) * (pow(bounds[r + 1], k + 1) - pow(bounds[r], k + 1))
        total += term
    return total


cdef inline double _normal_pdf(double z) noexcept nogil:
    return exp(-z * z / 2.0) / _SQRT_2PI


cdef inline double _pow_times_pdf(double x, int power, double pdf) noexcept nogil:
    """x**power * pdf, which is 0 where the pdf is (x infinite) rather than NaN."""
    return 0.0 if pdf == 0.0 else pow(x, power) * pdf


cdef double _integrate_normal(
    const vector[vector[double]]& a, const vector[double]& bounds, double mu, double sigma
) noexcept nogil:
    """Partial moments by the recurrence R_k = mu R_(k-1) + (k-1) sigma**2 R_(k-2) - sigma [x**(k-1) phi(z)]."""
    cdef Py_ssize_t n_regions = <Py_ssize_t>bounds.size() - 1, n_powers = <Py_ssize_t>a.size(), k, r
    cdef vector[double] pdf_at
    cdef vector[vector[double]] moments
    cdef double total = 0.0, term, z_lower, z_upper
    pdf_at.resize(n_regions + 1)
    moments.resize(n_powers)
    for k in range(n_powers):
        moments[k].resize(n_regions)
    for r in range(n_regions + 1):
        pdf_at[r] = _normal_pdf((bounds[r] - mu) / sigma)
    for r in range(n_regions):
        z_lower = (bounds[r] - mu) / sigma
        z_upper = (bounds[r + 1] - mu) / sigma
        moments[0][r] = ndtr(z_upper) - ndtr(z_lower)
        if n_powers > 1:
            moments[1][r] = mu * moments[0][r] - sigma * (pdf_at[r + 1] - pdf_at[r])
        for k in range(2, n_powers):
            moments[k][r] = (
                mu * moments[k - 1][r]
                + (k - 1.0) * (sigma * sigma) * moments[k - 2][r]
                - sigma * (
                    _pow_times_pdf(bounds[r + 1], k - 1, pdf_at[r + 1]) - _pow_times_pdf(bounds[r], k - 1, pdf_at[r])
                )
            )
    for k in range(n_powers):
        if _all_zero(a[k]):
            continue
        term = 0.0
        for r in range(n_regions):
            term += a[k][r] * moments[k][r]
        total += term
    return total


cdef double _triangular_antiderivative(double x, int k, double a, double b, double m) noexcept nogil:
    """The integral of x**k times the triangular density from a to x, for x within [a, b]."""
    cdef double at_mode = 0.0
    if x <= m:
        if m > a:
            return (2.0 / ((b - a) * (m - a))) * (
                pow(x, k + 2) / (k + 2.0) - (a * pow(x, k + 1)) / (k + 1.0) + pow(a, k + 2) / ((k + 1.0) * (k + 2.0))
            )
        return 0.0
    if b > m:
        if m > a:
            at_mode = (2.0 / ((b - a) * (m - a))) * (
                pow(m, k + 2) / (k + 2.0) - (a * pow(m, k + 1)) / (k + 1.0) + pow(a, k + 2) / ((k + 1.0) * (k + 2.0))
            )
        return at_mode + (2.0 / ((b - a) * (b - m))) * (
            b * (pow(x, k + 1) - pow(m, k + 1)) / (k + 1.0) - (pow(x, k + 2) - pow(m, k + 2)) / (k + 2.0)
        )
    return 0.0


cdef double _integrate_triangular(
    const vector[vector[double]]& coefficients, const vector[double]& bounds, double a, double b, double m
) noexcept nogil:
    cdef Py_ssize_t n_regions = <Py_ssize_t>bounds.size() - 1, k, r
    cdef vector[double] clipped
    cdef double total = 0.0, term, scale = b - a
    clipped.resize(n_regions + 1)
    for r in range(n_regions + 1):
        # np.clip: NaN stays NaN
        clipped[r] = bounds[r] if isnan(bounds[r]) else min(max(bounds[r], a), b)
    for k in range(<Py_ssize_t>coefficients.size()):
        if _all_zero(coefficients[k]):
            continue
        term = 0.0
        for r in range(n_regions):
            if k == 0:
                term += coefficients[k][r] * (
                    _triangular_cdf(clipped[r + 1], (m - a) / scale, a, scale)
                    - _triangular_cdf(clipped[r], (m - a) / scale, a, scale)
                )
            else:
                term += coefficients[k][r] * (
                    _triangular_antiderivative(clipped[r + 1], k, a, b, m)
                    - _triangular_antiderivative(clipped[r], k, a, b, m)
                )
        total += term
    return total


cdef double _size_biased_cdf(Distribution distribution, const double* p, int k, double x) noexcept nogil:
    """
    The CDF at x of the distribution of X weighted by X**k, for a strictly positive X.

    Its differences over the regions, times E[X**k], are the partial moments of X.
    """
    if distribution == GAMMA:
        return _gamma_cdf(x, p[0] + k, p[1])
    if distribution == PARETO:
        return _pareto_cdf(x, p[1] - k, p[0])
    if distribution == EXPONENTIAL:
        if k == 0:
            return _exponential_cdf(x, 1.0 / p[0])
        return _gamma_cdf(x, 1.0 + k, 1.0 / p[0])
    if distribution == CHI_SQUARED:
        return _chi_squared_cdf(x, p[0] + 2.0 * k)
    if distribution == LOG_NORMAL:
        return _log_normal_cdf(x, p[1], exp(p[0] + k * (p[1] * p[1])))
    if distribution == BETA:
        return betainc(p[0] + k, p[1], x)
    # WEIBULL
    if k == 0:
        return _weibull_cdf(x, p[1], p[0])
    return _gamma_cdf(pow(x / p[0], p[1]), 1.0 + k / p[1], 1.0)


cdef double _moment(Distribution distribution, const double* p, int k) noexcept nogil:
    """E[X**k]."""
    if distribution == EXPONENTIAL:
        return 1.0 if k == 0 else gamma(1.0 + k) * pow(1.0 / p[0], k)
    if distribution == WEIBULL:
        return 1.0 if k == 0 else pow(p[0], k) * gamma(1.0 + k / p[1])
    if k == 0:
        return 1.0
    if distribution == GAMMA:
        return pow(p[1], k) * (gamma(p[0] + k) / gamma(p[0]))
    if distribution == PARETO:
        return (p[1] * pow(p[0], k)) / (p[1] - k)
    if distribution == CHI_SQUARED:
        return pow(2.0, k) * gamma(k + p[0] / 2.0) / gamma(p[0] / 2.0)
    if distribution == LOG_NORMAL:
        return exp(k * p[0] + (k * k * (p[1] * p[1])) / 2.0)
    # BETA
    return (gamma(p[0] + k) * gamma(p[0] + p[1])) / (gamma(p[0]) * gamma(p[0] + p[1] + k))


cdef double _integrate_positive(
    Distribution distribution, const vector[vector[double]]& a, const vector[double]& bounds, const double* p
) noexcept nogil:
    cdef Py_ssize_t n_regions = <Py_ssize_t>bounds.size() - 1, k, r
    cdef vector[double] cdf_at
    cdef double total = 0.0, term, moment
    cdf_at.resize(n_regions + 1)
    for k in range(<Py_ssize_t>a.size()):
        moment = _moment(distribution, p, k)
        for r in range(n_regions + 1):
            cdf_at[r] = _size_biased_cdf(distribution, p, k, bounds[r])
        term = 0.0
        for r in range(n_regions):
            term += a[k][r] * moment * (cdf_at[r + 1] - cdf_at[r])
        total += term
    return total


# --- Entry point ------------------------------------------------------------------------------


cdef Py_ssize_t _n_parameters(Distribution distribution) noexcept nogil:
    """The number of parameters the distribution's sympy.stats constructor takes, or -1 if unknown."""
    if distribution == TRIANGULAR:
        return 3
    if distribution == EXPONENTIAL or distribution == CHI_SQUARED:
        return 1
    if (
        distribution == UNIFORM or distribution == NORMAL or distribution == GAMMA or distribution == PARETO
        or distribution == LOG_NORMAL or distribution == BETA or distribution == WEIBULL
    ):
        return 2
    return -1


def expected_max_profit(
    const double[::1] true_positive_rates,
    const double[::1] false_positive_rates,
    const double[::1] constant,
    const double[::1] tpr_slope,
    const double[::1] fpr_slope,
    double lower_bound,
    double upper_bound,
    Distribution distribution,
    const double[::1] distribution_parameters,
) -> float:
    """
    Return the expected maximum profit over a ROC convex hull.

    Parameters
    ----------
    true_positive_rates, false_positive_rates : 1D np.ndarray, shape=(n_vertices,)
        The ROC convex hull.

    constant, tpr_slope, fpr_slope : 1D np.ndarray, shape=(degree + 1,)
        The coefficient of each power of the stochastic variable in the profit, in ascending powers,
        as ``constant + tpr_slope * tpr + fpr_slope * fpr``. The degree is at most two.

    lower_bound, upper_bound : float
        The support of the stochastic variable; either may be infinite.

    distribution : Distribution
        The distribution of the stochastic variable.

    distribution_parameters : 1D np.ndarray
        The distribution's parameters, in the order its sympy.stats constructor takes them.

    Returns
    -------
    float
        The expected maximum profit.
    """
    cdef Py_ssize_t n_vertices = true_positive_rates.shape[0], n_powers = constant.shape[0], m, k, r
    if n_vertices == 0 or false_positive_rates.shape[0] != n_vertices:
        raise ValueError(
            'The hull needs at least one vertex, with as many false as true positive rates, got '
            f'{n_vertices} and {false_positive_rates.shape[0]}.'
        )
    if not 1 <= n_powers <= 3 or tpr_slope.shape[0] != n_powers or fpr_slope.shape[0] != n_powers:
        raise ValueError(
            'The profit needs one to three coefficients (a polynomial of degree at most two) of each '
            f'kind, got {n_powers}, {tpr_slope.shape[0]} and {fpr_slope.shape[0]}.'
        )
    cdef Py_ssize_t n_parameters = _n_parameters(distribution)
    if n_parameters < 0:
        raise ValueError(f'Unknown distribution {distribution}.')
    if distribution_parameters.shape[0] != n_parameters:
        raise ValueError(
            f'The distribution takes {n_parameters} parameters, got {distribution_parameters.shape[0]}.'
        )
    if distribution == PARETO:
        for k in range(n_powers):
            if distribution_parameters[1] <= k:
                raise ValueError(
                    f'The Pareto shape parameter (alpha={distribution_parameters[1]}) must be strictly '
                    f'greater than degree k={k} for the moment to exist.'
                )

    cdef const double* p = &distribution_parameters[0]
    cdef vector[double] coefficients  # coefficients[m * n_powers + k]: of x**k at vertex m
    cdef vector[double] bounds
    cdef vector[Py_ssize_t] vertices
    cdef vector[vector[double]] a  # a[k][r]: the coefficient of x**k on region r
    cdef double total

    with nogil:
        coefficients.resize(n_vertices * n_powers)
        for m in range(n_vertices):
            for k in range(n_powers):
                coefficients[m * n_powers + k] = (
                    constant[k] + tpr_slope[k] * true_positive_rates[m] + fpr_slope[k] * false_positive_rates[m]
                )
        _partition(coefficients, n_vertices, n_powers, lower_bound, upper_bound, bounds, vertices)
        a.resize(n_powers)
        for k in range(n_powers):
            a[k].resize(vertices.size())
            for r in range(<Py_ssize_t>vertices.size()):
                a[k][r] = coefficients[vertices[r] * n_powers + k]

        if distribution == UNIFORM:
            total = _integrate_uniform(a, bounds, lower_bound, upper_bound)
        elif distribution == NORMAL:
            total = _integrate_normal(a, bounds, p[0], p[1])
        elif distribution == TRIANGULAR:
            total = _integrate_triangular(a, bounds, p[0], p[1], p[2])
        else:
            total = _integrate_positive(distribution, a, bounds, p)
    return total
