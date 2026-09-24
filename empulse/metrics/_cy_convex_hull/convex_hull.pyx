# distutils: language = c++
"""
The ROC convex hull, the part of the convex hull of a ROC curve on or above its diagonal.

Everything between the inputs and the two output arrays runs in C++ without the GIL: sorting the
samples, accumulating the ROC curve and building its hull. Only large inputs call back into NumPy,
for its faster sort.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport isnan
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.vector cimport vector

cnp.import_array()

# Below this many samples, sorting them in C++ beats the overhead of calling NumPy's (faster) sort.
cdef Py_ssize_t _MAX_SAMPLES_SORTED_IN_CPP = 256


cdef struct Point:
    long long n_negative  # samples ranked at or above a threshold that are negative
    long long n_positive  # ... and positive


cdef struct Group:
    double score
    long long n_positive
    long long n_negative


cdef inline long long _cross(const Point& o, const Point& a, const Point& b) noexcept nogil:
    """
    Cross product of (o->a) and (o->b): positive for a left turn, zero when the points are collinear.

    Exact, since the points are integer counts: the counts of n samples are at most n, so the
    products stay below n**2 / 4 and fit a 64-bit integer for any n that fits in memory.
    """
    return (
        (a.n_negative - o.n_negative) * (b.n_positive - o.n_positive)
        - (a.n_positive - o.n_positive) * (b.n_negative - o.n_negative)
    )


cdef inline void _add_to_hull(vector[Point]& hull, long long n_negative, long long n_positive) noexcept nogil:
    """
    Extend the upper hull with the next point of the ROC curve.

    The ROC curve rises in both coordinates from (0, 0) to (1, 1), so its points come already
    sorted, and the part of its convex hull on or above the diagonal is the upper hull between
    those two corners. A monotone chain builds it in a single pass: walking the curve from left to
    right, a point is kept only while the hull turns right (clockwise) at it. Points on a straight
    line between two others add nothing and are dropped. The turns are computed exactly on the
    integer counts, so no point of the hull is lost to rounding, however close together the points
    of a large sample lie.
    """
    cdef Point point = Point(n_negative, n_positive)
    while hull.size() >= 2 and _cross(hull[hull.size() - 2], hull[hull.size() - 1], point) >= 0:
        hull.pop_back()
    hull.push_back(point)


cdef inline bint _ranks_higher(const Group& a, const Group& b) noexcept nogil:
    """Order by descending score, with NaN (which NumPy sorts last) ranked first."""
    return a.score > b.score or (isnan(a.score) and not isnan(b.score))


cdef void _add_groups_to_hull(vector[Group]& groups, vector[Point]& hull) noexcept nogil:
    """Rank the groups by score and add the ROC curve they give to the hull, one point per distinct score."""
    cpp_sort(groups.begin(), groups.end(), _ranks_higher)
    cdef long long n_negative = 0, n_positive = 0
    cdef size_t i
    for i in range(groups.size()):
        n_negative += groups[i].n_negative
        n_positive += groups[i].n_positive
        # Scores that tie form a single threshold: only the last of them is a point of the curve.
        if i + 1 == groups.size() or groups[i + 1].score != groups[i].score:
            _add_to_hull(hull, n_negative, n_positive)


cdef void _add_sorted_samples_to_hull(
    const int[:] y_true, const double[:] y_score, const cnp.intp_t[::1] ascending, vector[Point]& hull
) noexcept nogil:
    """Add the ROC curve of the samples to the hull, given the order that sorts their scores ascending."""
    cdef Py_ssize_t n_samples = ascending.shape[0], i, sample
    cdef long long n_positive = 0
    for i in range(n_samples - 1, -1, -1):
        sample = ascending[i]
        n_positive += y_true[sample]
        # Scores that tie form a single threshold: only the last of them is a point of the curve.
        if i == 0 or y_score[ascending[i - 1]] != y_score[sample]:
            _add_to_hull(hull, n_samples - i - n_positive, n_positive)


cdef tuple _rates(const vector[Point]& hull):
    """Turn the counts at the vertices of the hull into (TPR, FPR) arrays."""
    cdef long long n_negatives = hull.back().n_negative
    cdef long long n_positives = hull.back().n_positive
    cdef cnp.npy_intp n_vertices = hull.size()
    cdef cnp.ndarray tpr_array, fpr_array
    cdef double* tpr
    cdef double* fpr
    cdef Py_ssize_t i

    if n_negatives == 0 or n_positives == 0:
        # Without positives (or without negatives) the true (false) positive rate is undefined, but
        # it is also irrelevant, as it is weighted by a class prior of zero. The profit is then
        # linear in the one rate that matters, so it is highest either when no one or when everyone
        # is targeted: the hull is the diagonal between those two points, (0, 0) and (1, 1).
        n_vertices = 2

    tpr_array = cnp.PyArray_EMPTY(1, &n_vertices, cnp.NPY_FLOAT64, 0)
    fpr_array = cnp.PyArray_EMPTY(1, &n_vertices, cnp.NPY_FLOAT64, 0)
    tpr = <double*>cnp.PyArray_DATA(tpr_array)
    fpr = <double*>cnp.PyArray_DATA(fpr_array)
    if n_negatives == 0 or n_positives == 0:
        tpr[0] = fpr[0] = 0.0
        tpr[1] = fpr[1] = 1.0
    else:
        for i in range(n_vertices):
            tpr[i] = hull[i].n_positive / <double>n_positives
            fpr[i] = hull[i].n_negative / <double>n_negatives
    return tpr_array, fpr_array


def convex_hull(const int[:] y_true, const double[:] y_score) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the convex hull points of the ROC curve.

    Parameters
    ----------
    y_true : 1D np.ndarray, shape=(n_samples,)
        Binary target values.

    y_pred : 1D np.ndarray, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Convex Hull points of the ROC curve (TPR, FPR)
    """
    cdef Py_ssize_t n_samples = y_true.shape[0], i
    if n_samples != y_score.shape[0]:
        raise ValueError(
            f'y_true and y_score must have the same length, got {n_samples} and {y_score.shape[0]}.'
        )
    if n_samples == 0:
        raise ValueError('The ROC convex hull needs at least one sample.')

    cdef vector[Point] hull
    cdef vector[Group] samples
    cdef const cnp.intp_t[::1] ascending
    if n_samples <= _MAX_SAMPLES_SORTED_IN_CPP:
        with nogil:
            hull.push_back(Point(0, 0))  # targeting no one
            samples.resize(n_samples)
            for i in range(n_samples):
                samples[i] = Group(y_score[i], y_true[i], 1 - y_true[i])
            _add_groups_to_hull(samples, hull)
    else:
        # The order of tied scores does not matter, as each tie is a single point of the curve, so
        # the sort need not be stable.
        ascending = np.argsort(y_score)
        with nogil:
            hull.push_back(Point(0, 0))  # targeting no one
            _add_sorted_samples_to_hull(y_true, y_score, ascending, hull)
    return _rates(hull)


def convex_hull_from_counts(
    const double[:] y_score,
    const long long[:] n_positive,
    const long long[:] n_negative,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the convex hull points of the ROC curve of samples grouped by score.

    Gives the same points as :func:`convex_hull` on the samples themselves, in time that depends on
    the number of groups rather than the number of samples.

    Parameters
    ----------
    y_score : 1D np.ndarray, shape=(n_groups,)
        The score shared by the samples of each group. Scores need not be distinct.

    n_positive : 1D np.ndarray, shape=(n_groups,)
        The number of positive samples in each group.

    n_negative : 1D np.ndarray, shape=(n_groups,)
        The number of negative samples in each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Convex Hull points of the ROC curve (TPR, FPR)
    """
    cdef Py_ssize_t n_groups = y_score.shape[0], i
    if not n_groups == n_positive.shape[0] == n_negative.shape[0]:
        raise ValueError(
            'y_score, n_positive and n_negative must have the same length, got '
            f'{n_groups}, {n_positive.shape[0]} and {n_negative.shape[0]}.'
        )
    cdef bint has_negative_count = False
    cdef long long n_samples = 0
    cdef vector[Group] groups
    cdef vector[Point] hull
    with nogil:
        groups.resize(n_groups)
        for i in range(n_groups):
            groups[i] = Group(y_score[i], n_positive[i], n_negative[i])
            has_negative_count |= n_positive[i] < 0 or n_negative[i] < 0
            n_samples += n_positive[i] + n_negative[i]
    if has_negative_count:
        raise ValueError('The numbers of positive and negative samples cannot be negative.')
    if n_samples == 0:
        raise ValueError('The ROC convex hull needs at least one sample.')
    with nogil:
        hull.push_back(Point(0, 0))  # targeting no one
        _add_groups_to_hull(groups, hull)
    return _rates(hull)
