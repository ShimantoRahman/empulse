from cython.cimports import numpy as cnp  # noqa: F401
from cython.cimports.libc.math import sqrt, atan2, pi  # noqa: F401
from cython.cimports.libcpp.vector import vector  # noqa: F401

cdef float max_profit_score(
    cnp.ndarray[cnp.int32_t, ndim=1] y_true,
    cnp.ndarray[cnp.float32_t, ndim=1] y_score,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
)
