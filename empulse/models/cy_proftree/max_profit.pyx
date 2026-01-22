import numpy as np
cimport numpy as cnp

from ...metrics._cy_convex_hull.convex_hull cimport convex_hull


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

    cdef cnp.ndarray[cnp.float64_t, ndim=1] f0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] f1
    f0, f1 = convex_hull(y_true, y_score)
    cdef int hull_size = f0.shape[0]

    cdef float maximum_profit = -np.inf
    cdef float profit = 0.0

    for i in range(hull_size):
        profit = (
            (tp_benefit + fn_cost) * pi0 * f0[i]
            - (tn_benefit + fp_cost) * pi1 * f1[i]
            + tn_benefit * pi1
            - fn_cost * pi0
        )
        if profit > maximum_profit:
            maximum_profit = profit

    return maximum_profit
