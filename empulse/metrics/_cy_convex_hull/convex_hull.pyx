# distutils: language = c++

import cython
import numpy as np
from cython.cimports import numpy as cnp  # noqa: F401
from cython.cimports.libc.math import sqrt, atan2, pi  # noqa: F401
from cython.cimports.libcpp.vector import vector  # noqa: F401

cdef struct Point:
    double x
    double y

cdef inline int _orientation(const Point& p1, const Point& p2, const Point& p3) nogil:
    """Cross product sign on vectors (p1->p2) and (p2->p3)"""
    cdef double d = (p3.y - p2.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p2.x)
    if d > 1.1920929e-07:
        return 1
    else:
        return -1

cdef inline void _push_point(vector[Point]& hull, double x, double y) noexcept nogil:
    cdef Point p
    p.x = x
    p.y = y
    hull.push_back(p)

cdef tuple[np.ndarray[np.float64], np.ndarray[np.float64]] _compute_roc_curve(  # noqa: F401
        cnp.ndarray[cnp.int32_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_score  # noqa: F401
):
    """Compute ROC curve points from true and predicted scores."""
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

    cdef cnp.ndarray[cnp.int32_t, ndim=1] threshold_idxs = np.empty(threshold_idxs_vector.size(), dtype=np.int32)  # noqa: F401
    for i in range(threshold_idxs_vector.size()):
        threshold_idxs[i] = threshold_idxs_vector[i]

    # Accumulate TP/FP at thresholds
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tpr = np.cumsum(y_true[desc_idx], dtype=np.float64)[threshold_idxs]  # noqa: F401
    n_samples: cython.int =  tpr.size
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fpr = np.array(threshold_idxs, dtype=np.float64)  # noqa: F401
    cdef double n_positives = tpr[n_samples - 1]
    cdef double n_negatives = n_rows - n_positives
    for i in range(n_samples):
        fpr[i] = (1.0 + fpr[i] - tpr[i]) / n_negatives
        tpr[i] = tpr[i] / n_positives

    # Anchor to (0,0) and (1,1) if not already present
    if fpr.size == 0 or fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.concatenate(([0.0], fpr))
        tpr = np.concatenate(([0.0], tpr))
    if fpr[n_samples - 1] != 1.0 or tpr[n_samples - 1] != 1.0:
        fpr = np.concatenate((fpr, [1.0]))
        tpr = np.concatenate((tpr, [1.0]))

    return (np.ascontiguousarray(fpr, dtype=np.float64),
            np.ascontiguousarray(tpr, dtype=np.float64))


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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fpr  # noqa: F401
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tpr  # noqa: F401
    fpr, tpr = _compute_roc_curve(y_true, y_score)

    n_rows = cython.declare(cython.int, fpr.shape[0])
    cdef int i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] angles = np.empty(n_rows, dtype=np.float64)  # noqa: F401
    cdef cnp.ndarray[cnp.float64_t, ndim=1] dist2 = np.empty(n_rows, dtype=np.float64)  # noqa: F401

    # Compute sort keys: primary - polar angle, secondary - distance^2
    cdef double p0x = fpr[0]
    cdef double p0y = tpr[0]
    cdef double dy, dx
    # Matching behavior of earlier polar_angle: if p.y == p0.y, put first via angle = -pi
    for i in range(n_rows):
        dy = tpr[i] - p0y
        dx = fpr[i] - p0x
        angles[i] = atan2(dy, dx) if dy != 0.0 else -pi
        dist2[i] = dy * dy + dx * dx

    cdef cnp.ndarray[cnp.int64_t, ndim=1] order = np.lexsort((dist2, angles))  # noqa: F401

    hull: vector[Point]
    hull.reserve(n_rows)

    cdef Point p, p1, p2
    cdef double x, y

    # Graham scan using C++ vector as the stack
    with nogil:
        for i in range(order.shape[0]):
            x = <double> fpr[order[i]]
            y = <double> tpr[order[i]]
            # skip exact duplicate of p0 that could appear first
            # (optional; harmless if included)
            while hull.size() >= 2:
                p2 = hull[hull.size() - 1]
                p1 = hull[hull.size() - 2]
                if _orientation(p1, p2, Point(x, y)) == 1:
                    break
                hull.pop_back()
            _push_point(hull, x, y)

    # Filter points under the 45-degree line (x <= y)
    # Collect into temporary arrays for final sort by x
    cdef Py_ssize_t m = hull.size()
    cdef Py_ssize_t k, cnt = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xf = np.empty(m, dtype=np.float64)  # noqa: F401
    cdef cnp.ndarray[cnp.float64_t, ndim=1] yf = np.empty(m, dtype=np.float64)  # noqa: F401
    for k in range(m):
        p = hull[k]
        if p.x <= p.y:
            xf[cnt] = p.x
            yf[cnt] = p.y
            cnt += 1

    xf = xf[:cnt]
    yf = yf[:cnt]

    # Sort by x (ascending). Use NumPy for the final ordering.
    if cnt > 1:
        idx = np.argsort(xf, kind="mergesort")
        xf = xf[idx]
        yf = yf[idx]

    return yf, xf