# distutils: language = c++

import cython
import numpy as np
from cython.cimports import numpy as cnp  # noqa: F401
from cython.cimports.libcpp.vector import vector  # noqa: F401

cdef struct Point:
    long long n_negative  # samples ranked at or above a threshold that are negative
    long long n_positive  # ... and positive


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


cdef tuple _compute_roc_curve(  # noqa: F401
        cnp.ndarray[cnp.int32_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_score  # noqa: F401
):
    """
    Compute the ROC curve of the samples, as the negatives and positives ranked at each distinct score.

    Returns cumulative counts: element i holds the numbers of negative and positive samples scored at
    or above the i-th highest distinct score.
    """
    # Sort by score descending while preserving stability on ties
    cdef cnp.ndarray[cnp.int32_t, ndim=1] desc_idx = np.argsort(y_score, kind="mergesort")[::-1].astype(np.int32)  # noqa: F401

    n_rows = cython.declare(cython.int, y_true.shape[0])
    cdef int i
    cdef double last_value = y_score[desc_idx[0]]
    threshold_idxs_vector: vector[cython.int]
    threshold_idxs_vector.reserve(n_rows)

    # Distinct thresholds (where score changes)
    for i in range(1, desc_idx.size):
        if y_score[desc_idx[i]] != last_value:
            threshold_idxs_vector.push_back(i - 1)
            last_value = y_score[desc_idx[i]]
    threshold_idxs_vector.push_back(n_rows - 1)

    cdef cnp.ndarray[cnp.int64_t, ndim=1] threshold_idxs = np.empty(threshold_idxs_vector.size(), dtype=np.int64)  # noqa: F401
    for i in range(threshold_idxs_vector.size()):
        threshold_idxs[i] = threshold_idxs_vector[i]

    # Accumulate the samples and positives at each threshold
    cdef cnp.ndarray[cnp.int64_t, ndim=1] n_ranked_positive = np.cumsum(y_true[desc_idx], dtype=np.int64)[threshold_idxs]  # noqa: F401
    cdef cnp.ndarray[cnp.int64_t, ndim=1] n_ranked_negative = threshold_idxs + 1 - n_ranked_positive  # noqa: F401
    return n_ranked_negative, n_ranked_positive


cdef tuple _compute_roc_curve_from_counts(  # noqa: F401
        cnp.ndarray[cnp.float64_t, ndim=1] y_score,  # noqa: F401
        cnp.ndarray[cnp.int64_t, ndim=1] n_positive,  # noqa: F401
        cnp.ndarray[cnp.int64_t, ndim=1] n_negative,  # noqa: F401
):
    """Compute the ROC curve of samples grouped by score, like :func:`_compute_roc_curve`."""
    cdef cnp.ndarray[cnp.int64_t, ndim=1] desc_idx = np.argsort(y_score, kind="mergesort")[::-1]  # noqa: F401
    cdef int n_groups = y_score.shape[0]
    n_ranked_negative_vector: vector[cython.longlong]
    n_ranked_positive_vector: vector[cython.longlong]
    n_ranked_negative_vector.reserve(n_groups)
    n_ranked_positive_vector.reserve(n_groups)

    # Scores that tie form a single threshold, like tied samples do.
    cdef long long ranked_negative = 0, ranked_positive = 0
    cdef int i, j
    for i in range(n_groups):
        j = desc_idx[i]
        ranked_negative += n_negative[j]
        ranked_positive += n_positive[j]
        if i == n_groups - 1 or y_score[desc_idx[i + 1]] != y_score[j]:
            n_ranked_negative_vector.push_back(ranked_negative)
            n_ranked_positive_vector.push_back(ranked_positive)

    cdef cnp.ndarray[cnp.int64_t, ndim=1] n_ranked_negative = np.empty(n_ranked_negative_vector.size(), dtype=np.int64)  # noqa: F401
    cdef cnp.ndarray[cnp.int64_t, ndim=1] n_ranked_positive = np.empty(n_ranked_negative_vector.size(), dtype=np.int64)  # noqa: F401
    for i in range(n_ranked_negative_vector.size()):
        n_ranked_negative[i] = n_ranked_negative_vector[i]
        n_ranked_positive[i] = n_ranked_positive_vector[i]
    return n_ranked_negative, n_ranked_positive


def convex_hull(cnp.ndarray[cnp.int32_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_score) -> tuple[np.ndarray, np.ndarray]:  # noqa: F401
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
    n_ranked_negative, n_ranked_positive = _compute_roc_curve(y_true, y_score)
    return _roc_convex_hull(n_ranked_negative, n_ranked_positive)


def convex_hull_from_counts(
    cnp.ndarray[cnp.float64_t, ndim=1] y_score,  # noqa: F401
    cnp.ndarray[cnp.int64_t, ndim=1] n_positive,  # noqa: F401
    cnp.ndarray[cnp.int64_t, ndim=1] n_negative,  # noqa: F401
) -> tuple[np.ndarray, np.ndarray]:  # noqa: F401
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
    n_ranked_negative, n_ranked_positive = _compute_roc_curve_from_counts(y_score, n_positive, n_negative)
    return _roc_convex_hull(n_ranked_negative, n_ranked_positive)


cdef tuple _roc_convex_hull(
    cnp.ndarray[cnp.int64_t, ndim=1] n_ranked_negative,  # noqa: F401
    cnp.ndarray[cnp.int64_t, ndim=1] n_ranked_positive,  # noqa: F401
):
    """
    Return the convex hull of the ROC curve points on or above the diagonal, as (TPR, FPR).

    The ROC curve rises in both coordinates from (0, 0) to (1, 1), so its points come already
    sorted, and the part of its convex hull on or above the diagonal is the upper hull between
    those two corners. A monotone chain builds it in a single pass: walking the curve from left to
    right, a point is kept only while the hull turns right (clockwise) at it. Points on a straight
    line between two others add nothing and are dropped. The turns are computed exactly on the
    integer counts, so no point of the hull is lost to rounding, however close together the points
    of a large sample lie.
    """
    cdef Py_ssize_t n_points = n_ranked_negative.shape[0]
    cdef double n_negatives = <double>n_ranked_negative[n_points - 1]
    cdef double n_positives = <double>n_ranked_positive[n_points - 1]

    hull: vector[Point]
    hull.reserve(n_points + 1)
    hull.push_back(Point(0, 0))  # predicting every sample negative

    cdef Point point
    cdef Py_ssize_t i
    for i in range(n_points):
        point = Point(n_ranked_negative[i], n_ranked_positive[i])
        while hull.size() >= 2 and _cross(hull[hull.size() - 2], hull[hull.size() - 1], point) >= 0:
            hull.pop_back()
        hull.push_back(point)

    cdef Py_ssize_t n_vertices = hull.size()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tpr = np.empty(n_vertices, dtype=np.float64)  # noqa: F401
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fpr = np.empty(n_vertices, dtype=np.float64)  # noqa: F401
    for i in range(n_vertices):
        tpr[i] = hull[i].n_positive / n_positives
        fpr[i] = hull[i].n_negative / n_negatives
    return tpr, fpr
