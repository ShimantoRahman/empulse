# distutils: language = c++

import cython
import numpy as np
from cython.cimports import numpy as cnp  # noqa: F401
from cython.cimports.libc.math import sqrt, atan2, pi  # noqa: F401
from cython.cimports.libcpp.vector import vector  # noqa: F401
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.pair cimport pair

cdef void argsort_float32(cnp.float32_t* arr, cnp.int64_t* indices, Py_ssize_t n) noexcept:
    """In-place argsort using C++ std::sort"""
    cdef Py_ssize_t i
    cdef vector[pair[float, Py_ssize_t]] pairs
    pairs.reserve(n)

    for i in range(n):
        pairs.push_back(pair[float, Py_ssize_t](arr[i], i))

    cpp_sort(pairs.begin(), pairs.end())

    for i in range(n):
        indices[i] = pairs[i].second

cdef void argsort_float32_desc(cnp.float32_t* arr, cnp.int32_t* indices, Py_ssize_t n) noexcept:
    """Descending argsort using C++ std::sort"""
    cdef Py_ssize_t i
    cdef vector[pair[float, int]] pairs
    pairs.reserve(n)

    for i in range(n):
        pairs.push_back(pair[float, int](-arr[i], i))  # Negate for descending

    cpp_sort(pairs.begin(), pairs.end())

    for i in range(n):
        indices[i] = pairs[i].second

cdef struct Point:
    float x
    float y

cdef inline int _orientation(const Point& p1, const Point& p2, const Point& p3) noexcept:
    """Cross product sign on vectors (p1->p2) and (p2->p3)"""
    cdef float d = (p3.y - p2.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p2.x)
    if d > 1.1920929e-07:
        return 1
    else:
        return -1

cdef inline void _push_point(vector[Point]& hull, float x, float y) noexcept:
    cdef Point p
    p.x = x
    p.y = y
    hull.push_back(p)

cdef cnp.float32_t[:, ::1] _compute_roc_curve(
    cnp.ndarray[cnp.int32_t, ndim=1] y_true,
    cnp.ndarray[cnp.float32_t, ndim=1] y_score
):
    cdef int n_rows = y_true.shape[0]

    # Custom descending argsort
    cdef cnp.ndarray[cnp.int32_t, ndim=1] desc_idx = np.empty(n_rows, dtype=np.int32)
    argsort_float32_desc(&y_score[0], &desc_idx[0], n_rows)

    # Find thresholds and compute cumsum in one pass
    cdef vector[int] threshold_idxs
    cdef vector[float] tpr_vec
    threshold_idxs.reserve(n_rows)
    tpr_vec.reserve(n_rows)

    cdef float last_value = y_score[desc_idx[0]]
    cdef float cumsum = 0.0
    cdef int i

    for i in range(n_rows):
        cumsum += y_true[desc_idx[i]]
        if i == n_rows - 1 or y_score[desc_idx[i + 1]] != last_value:
            threshold_idxs.push_back(i)
            tpr_vec.push_back(cumsum)
            if i < n_rows - 1:
                last_value = y_score[desc_idx[i + 1]]

    # Normalize
    cdef int n_samples = threshold_idxs.size()
    cdef float n_positives = tpr_vec[n_samples - 1]
    cdef float n_negatives = n_rows - n_positives

    # Pre-allocate with room for anchors
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.empty((n_samples + 2, 2), dtype=np.float32)

    # Anchor (0, 0)
    result[0, 0] = 0.0
    result[0, 1] = 0.0

    # Fill computed values
    cdef int offset = 1
    for i in range(n_samples):
        result[offset + i, 0] = (1.0 + threshold_idxs[i] - tpr_vec[i]) / n_negatives
        result[offset + i, 1] = tpr_vec[i] / n_positives

    # Check if we need anchor (1, 1)
    cdef int final_size = n_samples + 1
    if result[n_samples, 0] != 1.0 or result[n_samples, 1] != 1.0:
        result[n_samples + 1, 0] = 1.0
        result[n_samples + 1, 1] = 1.0
        final_size = n_samples + 2

    return result[:final_size, :]

cdef struct SortKey:
    float angle
    float dist2
    Py_ssize_t index

cdef inline bint compare_keys(const SortKey& a, const SortKey& b) noexcept:
    """Compare function for lexsort: primary by angle, secondary by dist2"""
    if a.angle < b.angle:
        return True
    if a.angle > b.angle:
        return False
    return a.dist2 < b.dist2

cdef cnp.float32_t[:, ::1] cy_convex_hull(cnp.ndarray[cnp.int32_t, ndim=1] y_true, cnp.ndarray[cnp.float32_t, ndim=1] y_score):
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
    cnp.float32_t[:, ::1]
        2D array with shape (n_points, 2) where column 0 is FPR and column 1 is TPR
    """
    cdef cnp.float32_t[:, ::1] roc_points = _compute_roc_curve(y_true, y_score)
    cdef int n_rows = roc_points.shape[0]
    cdef int i

    # Use C++ vector for sort keys
    cdef vector[SortKey] sort_keys
    sort_keys.reserve(n_rows)

    # Compute sort keys: primary - polar angle, secondary - distance^2
    cdef float p0x = roc_points[0, 0]
    cdef float p0y = roc_points[0, 1]
    cdef float dy, dx
    cdef SortKey key

    for i in range(n_rows):
        dy = roc_points[i, 1] - p0y
        dx = roc_points[i, 0] - p0x
        key.angle = atan2(dy, dx) if dy != 0.0 else -pi
        key.dist2 = dy * dy + dx * dx
        key.index = i
        sort_keys.push_back(key)

    # Sort using C++ std::sort with custom comparator
    cpp_sort(sort_keys.begin(), sort_keys.end(), compare_keys)

    cdef vector[Point] hull
    hull.reserve(n_rows)

    cdef Point p, p1, p2
    cdef float x, y

    # Graham scan using C++ vector as the stack
    for i in range(n_rows):
        x = roc_points[sort_keys[i].index, 0]
        y = roc_points[sort_keys[i].index, 1]
        while hull.size() >= 2:
            p2 = hull[hull.size() - 1]
            p1 = hull[hull.size() - 2]
            if _orientation(p1, p2, Point(x, y)) == 1:
                break
            hull.pop_back()
        _push_point(hull, x, y)

    # Filter points under the 45-degree line (x <= y)
    cdef Py_ssize_t m = hull.size()
    cdef Py_ssize_t k, cnt = 0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] xf = np.empty(m, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] yf = np.empty(m, dtype=np.float32)

    for k in range(m):
        p = hull[k]
        if p.x <= p.y:
            xf[cnt] = p.x
            yf[cnt] = p.y
            cnt += 1

    xf = xf[:cnt]
    yf = yf[:cnt]

    # Sort by x (ascending)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] idx
    if cnt > 1:
        idx = np.empty(cnt, dtype=np.int64)
        argsort_float32(&xf[0], &idx[0], cnt)
        xf = xf[idx]
        yf = yf[idx]

    # Create 2D result array
    cdef cnp.ndarray[cnp.float32_t, ndim=2] result = np.empty((cnt, 2), dtype=np.float32)
    for i in range(cnt):
        result[i, 0] = xf[i]
        result[i, 1] = yf[i]

    return result

cdef float max_profit_score(
    cnp.ndarray[cnp.int32_t, ndim=1] y_true,
    cnp.ndarray[cnp.float32_t, ndim=1] y_score,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
):
    cdef int i = 0
    cdef int n_samples = y_true.shape[0]
    cdef float pi0 = 0.0
    for i in range(n_samples):
        pi0 += y_true[i]
    pi0 /= n_samples
    cdef float pi1 = 1.0 - pi0

    # Change from ndarray to memory view
    cdef cnp.float32_t[:, ::1] ch = cy_convex_hull(y_true, y_score)
    cdef int hull_size = ch.shape[0]

    cdef float maximum_profit = -1e10
    cdef float profit = 0.0

    for i in range(hull_size):
        profit = (
            (tp_benefit + fn_cost) * pi0 * ch[i, 1]  # Note: swapped indices
            - (tn_benefit + fp_cost) * pi1 * ch[i, 0]
            + tn_benefit * pi1
            - fn_cost * pi0
        )
        if profit > maximum_profit:
            maximum_profit = profit

    return maximum_profit
