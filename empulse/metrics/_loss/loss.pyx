import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, fabs


ScoreType = cython.fused_type(cython.float[:], cython.double[:])
GradientType = cython.fused_type(cython.float[:], cython.double[:])


cdef inline double expit(double x) noexcept nogil:
    return 1.0 / (1.0 + exp(-x))


cdef inline double sign(double x) noexcept nogil:
    if x == 0.0:
        return 0.0
    elif x > 0.0:
        return 1.0
    else:
        return -1.0


# The elastic-net penalty arrives already scaled, as the two precomputed weights `l1_weight` and
# `l2_weight`. `ElasticNetPenalty` derives them once per fit from `C`, `l1_ratio` and the
# objective's scale, which keeps the scaling policy in Python (where it is easy to change without a
# rebuild) while the arithmetic stays here, where it costs nothing. Passing them as weights rather
# than as `C`/`l1_ratio` also removes the per-call branching on `l1_ratio == 0.0 / == 1.0`.
# Pass 0.0 for both to get the unpenalized data term, which is what the split-variable solver wants.


@cython.cdivision(True)  # turn off check zero division error (n_rows is always positive)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_logit_loss_gradient(
        const double[:] weights,
        const double[:, :] features,
        const double[:, :] grad_const,
        const double[:] loss_const1,
        const double[:] loss_const2,
        l1_weight: cython.double = 0.0,
        l2_weight: cython.double = 0.0,
        start_coef: cython.int = 1,
):
    i: cython.int
    j: cython.int
    n_rows: cython.int = <int>features.shape[0]
    n_cols: cython.int = <int>features.shape[1]

    loss: cython.double = 0.0
    cdef cnp.ndarray[double, ndim=1] gradient = np.zeros(n_cols, dtype=np.float64)
    cdef double[:] gradient_view = gradient
    cdef cnp.ndarray[double, ndim=1] y_score = np.zeros(n_rows, dtype=np.float64)
    cdef double[:] y_score_view = y_score

    for i in range(n_rows):
        for j in range(n_cols):
            y_score_view[i] += weights[j] * features[i, j]
        y_score_view[i] = expit(y_score_view[i])

    # compute gradient
    for i in range(n_rows):
        s: cython.double = y_score_view[i] * (1 - y_score_view[i])
        for j in range(n_cols):
            gradient_view[j] += s * grad_const[i, j]
    for j in range(n_cols):
        gradient_view[j] = gradient_view[j] / n_rows

    # compute loss
    for i in range(n_rows):
        loss += y_score[i] * loss_const1[i] + (1 - y_score[i]) * loss_const2[i]
    loss = loss / n_rows

    # apply the elastic-net penalty (coefficients only; the intercept is never penalized)
    if l1_weight != 0.0 or l2_weight != 0.0:
        for j in range(start_coef, n_cols):
            w: cython.double = weights[j]
            loss += l1_weight * fabs(w) + 0.5 * l2_weight * w * w
            gradient_view[j] += l1_weight * sign(w) + l2_weight * w

    return loss, gradient


@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_logit_loss(
        const double[:] weights,
        const double[:, :] features,
        const double[:] loss_const1,
        const double[:] loss_const2,
        l1_weight: cython.double = 0.0,
        l2_weight: cython.double = 0.0,
        start_coef: cython.int = 1,
):
    i: cython.int
    j: cython.int
    n_rows: cython.int = <int>features.shape[0]
    n_cols: cython.int = <int>features.shape[1]

    loss: cython.double = 0.0
    cdef cnp.ndarray[double, ndim=1] y_score = np.zeros(n_rows, dtype=np.float64)
    cdef double[:] y_score_view = y_score

    for i in range(n_rows):
        for j in range(n_cols):
            y_score_view[i] += weights[j] * features[i, j]
        y_score_view[i] = expit(y_score_view[i])

    # compute loss
    for i in range(n_rows):
        loss += y_score[i] * loss_const1[i] + (1 - y_score[i]) * loss_const2[i]
    loss = loss / n_rows

    # apply the elastic-net penalty (coefficients only; the intercept is never penalized)
    if l1_weight != 0.0 or l2_weight != 0.0:
        for j in range(start_coef, n_cols):
            w: cython.double = weights[j]
            loss += l1_weight * fabs(w) + 0.5 * l2_weight * w * w

    return loss


@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_logit_gradient(
        const double[:] weights,
        const double[:, :] features,
        const double[:, :] grad_const,
        l1_weight: cython.double = 0.0,
        l2_weight: cython.double = 0.0,
        start_coef: cython.int = 1,
):
    i: cython.int
    j: cython.int
    n_rows: cython.int = <int>features.shape[0]
    n_cols: cython.int = <int>features.shape[1]

    cdef cnp.ndarray[double, ndim=1] gradient = np.zeros(n_cols, dtype=np.float64)
    cdef double[:] gradient_view = gradient
    cdef cnp.ndarray[double, ndim=1] y_score = np.zeros(n_rows, dtype=np.float64)
    cdef double[:] y_score_view = y_score

    for i in range(n_rows):
        for j in range(n_cols):
            y_score_view[i] += weights[j] * features[i, j]
        y_score_view[i] = expit(y_score_view[i])

    # compute gradient
    for i in range(n_rows):
        s: cython.double = y_score_view[i] * (1 - y_score_view[i])
        for j in range(n_cols):
            gradient_view[j] += s * grad_const[i, j]
    for j in range(n_cols):
        gradient_view[j] = gradient_view[j] / n_rows

    # apply the elastic-net penalty (coefficients only; the intercept is never penalized)
    if l1_weight != 0.0 or l2_weight != 0.0:
        for j in range(start_coef, n_cols):
            w: cython.double = weights[j]
            gradient_view[j] += l1_weight * sign(w) + l2_weight * w

    return gradient


@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_boost_grad_hess(y_true, y_score: ScoreType, grad_const: GradientType):
    i: cython.int
    j: cython.int
    n_rows: cython.int = <int>grad_const.shape[0]

    cdef cnp.ndarray[double, ndim=1] gradient = np.zeros(n_rows, dtype=np.float64)
    cdef double[:] gradient_view = gradient
    cdef cnp.ndarray[double, ndim=1] hessian = np.zeros(n_rows, dtype=np.float64)
    cdef double[:] hessian_view = hessian

    # compute gradient
    for i in range(n_rows):
        y_proba: cython.double = expit(y_score[i])
        s: cython.double = y_proba * (1 - y_proba)
        gradient_view[i] = s * grad_const[i]
        hessian_view[i] = fabs((1 - 2 * y_proba) * gradient_view[i])

    return gradient, hessian
