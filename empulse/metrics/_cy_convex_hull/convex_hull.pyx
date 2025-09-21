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
    elif d < -1.1920929e-07:
        return -1
    else:
        return 0

cdef inline void _push_point(vector[Point]& hull, double x, double y) noexcept nogil:
    cdef Point p
    p.x = x
    p.y = y
    hull.push_back(p)

cdef tuple _compute_roc_curve(cnp.ndarray[cnp.int32_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_score):    # noqa: F401
    """Compute ROC curve points from true and predicted scores."""
    # Ensure correct dtype and contiguity
    # y_true = np.ascontiguousarray(y_true, dtype=np.int32)
    # y_score = np.ascontiguousarray(y_score, dtype=np.float64)

    # Sort by score descending while preserving stability on ties
    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    # Distinct thresholds (where score changes)
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate TP/FP at thresholds
    tps = np.cumsum(y_true, dtype=np.float64)[threshold_idxs]
    fps = 1.0 + threshold_idxs.astype(np.float64) - tps

    n_samples: cython.int =  fps.size
    fpr = fps / fps[n_samples - 1]
    tpr = tps / tps[n_samples - 1]

    # Anchor to (0,0) and (1,1) if not already present
    if fpr.size == 0 or fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.concatenate(([0.0], fpr))
        tpr = np.concatenate(([0.0], tpr))
    if fpr[n_samples - 1] != 1.0 or tpr[n_samples - 1] != 1.0:
        fpr = np.concatenate((fpr, [1.0]))
        tpr = np.concatenate((tpr, [1.0]))

    # Make contiguous float64 outputs
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

    n_rows = cython.declare(cython.int, y_true.shape[0])
    p0_idx = np.lexsort((fpr, tpr))[0]
    p0x: cython.double = fpr[p0_idx]
    p0y: cython.double = tpr[p0_idx]

    # Compute sort keys: primary - polar angle, secondary - distance^2
    dy = tpr - p0y
    dx = fpr - p0x
    # Matching behavior of earlier polar_angle: if p.y == p0.y, put first via angle = -pi
    angles = np.where(dy == 0.0, -pi, np.arctan2(dy, dx))
    dist2 = dy * dy + dx * dx

    cdef cnp.ndarray[cnp.int64_t, ndim=1] order = np.lexsort((dist2, angles))  # noqa: F401

    hull: vector[Point]
    hull.reserve(fpr.shape[0])

    cdef Py_ssize_t i
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

    return np.array(yf), np.array(xf)