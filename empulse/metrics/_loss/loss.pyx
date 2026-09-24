import cython
import numpy as np
from libc.math cimport exp, fabs
from scipy.linalg.cython_blas cimport dgemv


ScoreType = cython.fused_type(cython.float[:], cython.double[:])
GradientType = cython.fused_type(cython.float[:], cython.double[:])

# The logistic function of each margin m is computed from E = exp(-m), as
#
#     p = expit(m) = 1 / (1 + E),    1 - p = E * p,    expit'(m) = (E * p) * p.
#
# Neither the complement nor the derivative is formed by subtracting from 1, so both keep their
# precision where the probability is close to 0 or 1. The exponent is clamped to [-700, 700], so E
# neither overflows nor underflows; that changes only probabilities below 1e-304.
#
# The exponentials are taken by numpy for the whole vector at once: its exp is vectorized, and
# costs a fraction of calling the C library's exp for every sample. The matrix-vector products go
# to BLAS. Everything else is a single pass over the samples, without the GIL, and without a
# branch on the sign of the margin: that sign is as good as random from one sample to the next,
# and a branch on it would be mispredicted about half the time, costing more than the arithmetic.
_exp = np.exp
# Below this many samples, calling numpy costs more than it saves, so the C library's exp is used.
cdef Py_ssize_t _VECTORIZED_EXP_MIN_ROWS = 128


# The elastic-net penalty arrives already scaled, as the two precomputed weights `l1_weight` and
# `l2_weight`. `ElasticNetPenalty` derives them once per fit from `C`, `l1_ratio` and the
# objective's scale, which keeps the scaling policy in Python (where it is easy to change without a
# rebuild) while the arithmetic stays here, where it costs nothing. Passing them as weights rather
# than as `C`/`l1_ratio` also removes the per-call branching on `l1_ratio == 0.0 / == 1.0`.
# Pass 0.0 for both to get the unpenalized data term, which is what the split-variable solver wants.


cdef inline double _clamped_exponent(double negative_margin) noexcept nogil:
    """Clamp the exponent to [-700, 700], letting NaN through."""
    if negative_margin > 700.0:
        return 700.0
    if negative_margin < -700.0:
        return -700.0
    return negative_margin


cdef inline double sign(double x) noexcept nogil:
    if x == 0.0:
        return 0.0
    elif x > 0.0:
        return 1.0
    else:
        return -1.0


cdef void _matrix_vector(
    bint transpose, const double[:, ::1] matrix, const double* vector, double scale, double* out
) noexcept nogil:
    """Set `out` to `scale * matrix @ vector`, or to `scale * matrix.T @ vector` if `transpose`.

    `out` must be zeroed beforehand when the matrix is empty, since BLAS then leaves it untouched.
    """
    cdef int n_rows = <int>matrix.shape[0]
    cdef int n_cols = <int>matrix.shape[1]
    cdef int one = 1
    cdef double zero = 0.0
    # BLAS reads matrices column by column, so it sees this row-major matrix as its transpose.
    cdef char blas_transpose = b'N' if transpose else b'T'
    if n_rows == 0 or n_cols == 0:
        return
    dgemv(
        &blas_transpose, &n_cols, &n_rows, &scale, <double*>&matrix[0, 0], &n_cols,
        <double*>vector, &one, &zero, out, &one,
    )


cdef object _margins_and_exponentials(const double[::1] weights, const double[:, ::1] features):
    """Return `features @ weights` and `exp(-features @ weights)`, see `_clamped_exponent`."""
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = features.shape[0]
    if weights.shape[0] != features.shape[1]:
        raise ValueError(
            f'weights has {weights.shape[0]} entries, but features has {features.shape[1]} columns.'
        )
    margins = np.zeros(n_rows, dtype=np.float64)
    exponentials = np.empty(n_rows, dtype=np.float64)
    cdef double[::1] margins_view = margins
    cdef double[::1] exponentials_view = exponentials
    with nogil:
        _matrix_vector(False, features, &weights[0] if weights.shape[0] else NULL, 1.0, &margins_view[0] if n_rows else NULL)
        if n_rows < _VECTORIZED_EXP_MIN_ROWS:
            for i in range(n_rows):
                exponentials_view[i] = exp(_clamped_exponent(-margins_view[i]))
        else:
            for i in range(n_rows):
                exponentials_view[i] = _clamped_exponent(-margins_view[i])
    if n_rows >= _VECTORIZED_EXP_MIN_ROWS:
        _exp(exponentials, out=exponentials)
    return margins, exponentials


cdef void _check_rows(Py_ssize_t n_rows, const double[::1] loss_const1, const double[::1] loss_const2) except *:
    if loss_const1.shape[0] != n_rows or loss_const2.shape[0] != n_rows:
        raise ValueError(
            f'features has {n_rows} rows, but loss_const1 and loss_const2 have '
            f'{loss_const1.shape[0]} and {loss_const2.shape[0]} entries.'
        )


cdef void _check_gradient_const(const double[:, ::1] features, const double[:, ::1] grad_const) except *:
    if grad_const.shape[0] != features.shape[0] or grad_const.shape[1] != features.shape[1]:
        raise ValueError(
            f'features has shape ({features.shape[0]}, {features.shape[1]}), but grad_const has shape '
            f'({grad_const.shape[0]}, {grad_const.shape[1]}).'
        )


cdef double _data_loss_and_slopes(
    const double[::1] loss_const1,
    const double[::1] loss_const2,
    const double[::1] exponentials,
    double[::1] slopes,
    bint want_slopes,
) noexcept nogil:
    """Return the mean data loss. If `want_slopes`, also store expit' of each margin in `slopes`."""
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = exponentials.shape[0]
    cdef double loss = 0.0
    cdef double exponential, probability, complement
    for i in range(n_rows):
        exponential = exponentials[i]
        probability = 1.0 / (1.0 + exponential)
        complement = exponential * probability
        loss += probability * loss_const1[i] + complement * loss_const2[i]
        if want_slopes:
            slopes[i] = complement * probability
    return loss / n_rows


cdef double _penalty(const double[::1] weights, double l1_weight, double l2_weight, Py_ssize_t start_coef) noexcept nogil:
    """Return the elastic-net penalty (coefficients only; the intercept is never penalized)."""
    cdef Py_ssize_t j
    cdef double w
    cdef double penalty = 0.0
    if l1_weight == 0.0 and l2_weight == 0.0:
        return 0.0
    for j in range(start_coef, weights.shape[0]):
        w = weights[j]
        penalty += l1_weight * fabs(w) + 0.5 * l2_weight * w * w
    return penalty


cdef void _add_penalty_gradient(
    const double[::1] weights, double[::1] gradient, double l1_weight, double l2_weight, Py_ssize_t start_coef
) noexcept nogil:
    """Add the gradient of the elastic-net penalty to `gradient`."""
    cdef Py_ssize_t j
    cdef double w
    if l1_weight == 0.0 and l2_weight == 0.0:
        return
    for j in range(start_coef, weights.shape[0]):
        w = weights[j]
        gradient[j] += l1_weight * sign(w) + l2_weight * w


def cy_logit_loss_gradient(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[:, ::1] grad_const,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
):
    cdef Py_ssize_t n_rows = features.shape[0]
    cdef Py_ssize_t n_cols = features.shape[1]
    cdef double loss
    _check_rows(n_rows, loss_const1, loss_const2)
    _check_gradient_const(features, grad_const)
    margins, exponentials = _margins_and_exponentials(weights, features)
    gradient = np.zeros(n_cols, dtype=np.float64)
    cdef double[::1] slopes = margins
    cdef const double[::1] exponentials_view = exponentials
    cdef double[::1] gradient_view = gradient
    with nogil:
        loss = _data_loss_and_slopes(loss_const1, loss_const2, exponentials_view, slopes, True)
        if n_cols:
            _matrix_vector(True, grad_const, &slopes[0] if n_rows else NULL, 1.0 / n_rows, &gradient_view[0])
        loss += _penalty(weights, l1_weight, l2_weight, start_coef)
        _add_penalty_gradient(weights, gradient_view, l1_weight, l2_weight, start_coef)
    return loss, gradient


def cy_logit_loss(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
):
    cdef Py_ssize_t n_rows = features.shape[0]
    cdef double loss
    _check_rows(n_rows, loss_const1, loss_const2)
    margins, exponentials = _margins_and_exponentials(weights, features)
    cdef double[::1] margins_view = margins
    cdef const double[::1] exponentials_view = exponentials
    with nogil:
        loss = _data_loss_and_slopes(loss_const1, loss_const2, exponentials_view, margins_view, False)
        loss += _penalty(weights, l1_weight, l2_weight, start_coef)
    return loss


def cy_logit_gradient(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[:, ::1] grad_const,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = features.shape[0]
    cdef Py_ssize_t n_cols = features.shape[1]
    cdef double exponential, probability
    _check_gradient_const(features, grad_const)
    margins, exponentials = _margins_and_exponentials(weights, features)
    gradient = np.zeros(n_cols, dtype=np.float64)
    cdef double[::1] slopes = margins
    cdef const double[::1] exponentials_view = exponentials
    cdef double[::1] gradient_view = gradient
    with nogil:
        for i in range(n_rows):
            exponential = exponentials_view[i]
            probability = 1.0 / (1.0 + exponential)
            slopes[i] = exponential * probability * probability
        if n_cols:
            _matrix_vector(True, grad_const, &slopes[0] if n_rows else NULL, 1.0 / n_rows, &gradient_view[0])
        _add_penalty_gradient(weights, gradient_view, l1_weight, l2_weight, start_coef)
    return gradient


def cy_boost_grad_hess(y_true, ScoreType y_score, GradientType grad_const):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = grad_const.shape[0]
    cdef double exponential, probability, complement, gradient_i
    if y_score.shape[0] != n_rows:
        raise ValueError(f'y_score has {y_score.shape[0]} entries, but grad_const has {n_rows}.')

    gradient = np.empty(n_rows, dtype=np.float64)
    hessian = np.empty(n_rows, dtype=np.float64)
    cdef double[::1] gradient_view = gradient
    cdef double[::1] hessian_view = hessian
    # The exponentials are gathered in the hessian's buffer, which is filled only afterwards.
    with nogil:
        for i in range(n_rows):
            hessian_view[i] = _clamped_exponent(-<double>y_score[i])
    _exp(hessian, out=hessian)
    with nogil:
        for i in range(n_rows):
            exponential = hessian_view[i]
            probability = 1.0 / (1.0 + exponential)
            complement = exponential * probability
            gradient_i = complement * probability * grad_const[i]
            gradient_view[i] = gradient_i
            hessian_view[i] = fabs((complement - probability) * gradient_i)

    return gradient, hessian
