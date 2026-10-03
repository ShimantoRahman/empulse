import cython
import numpy as np
from cython.parallel cimport prange
from libc.float cimport DBL_EPSILON
from libc.math cimport exp, fabs, log
from scipy.linalg.cython_blas cimport dgemv


ScoreType = cython.fused_type(cython.float[:], cython.double[:])
GradientType = cython.fused_type(cython.float[:], cython.double[:])

# The logistic function of each margin m is computed from E = exp(-m), as
#
#     p = expit(m) = 1 / (1 + E),    1 - p = E * p,    expit'(m) = (E * p) * p.
#
# Neither the complement nor the derivative is formed by subtracting from 1, so both keep their
# precision where the probability is close to 0 or 1. The exponent is clamped to [-700, 700], so E
# neither overflows nor underflows; that changes only probabilities below 1e-304. No branch depends
# on the sign of the margin: that sign is effectively random from one sample to the next, so a branch
# would be mispredicted about half the time and cost more than the arithmetic.
#
# The boosting kernels take their exponentials with numpy for the whole vector at once.
_exp = np.exp
# Below this many values, calling numpy costs more than it saves, so the C library's exp is used.
cdef Py_ssize_t _VECTORIZED_MIN_SIZE = 128


# The elastic-net penalty arrives as the two precomputed weights `l1_weight` and `l2_weight`.
# `ElasticNetPenalty` derives them once per fit from `C`, `l1_ratio` and the objective's scale,
# which keeps the scaling policy in Python, where it changes without a rebuild. Weights also avoid
# a per-call branch on `l1_ratio == 0.0` and `l1_ratio == 1.0`. Pass 0.0 for both to get the
# unpenalized data term, which the split-variable solver uses.


cdef inline double _clamped_exponent(double negative_margin) noexcept nogil:
    """Clamp the exponent to [-700, 700], letting NaN through."""
    if negative_margin > 700.0:
        return 700.0
    if negative_margin < -700.0:
        return -700.0
    return negative_margin


cdef inline double _clipped_probability(double probability) noexcept nogil:
    """Clip a probability to [eps, 1 - eps] before taking its log, letting NaN through."""
    if probability < DBL_EPSILON:
        return DBL_EPSILON
    if probability > 1.0 - DBL_EPSILON:
        return 1.0 - DBL_EPSILON
    return probability


cdef inline double sign(double x) noexcept nogil:
    if x == 0.0:
        return 0.0
    elif x > 0.0:
        return 1.0
    else:
        return -1.0


cdef void _exp_in_place(object array, double[::1] values) except *:
    """Replace each of `values`, which views `array`, by its exponential."""
    cdef Py_ssize_t i
    if values.shape[0] < _VECTORIZED_MIN_SIZE:
        with nogil:
            for i in range(values.shape[0]):
                values[i] = exp(values[i])
    else:
        _exp(array, out=array)


cdef object _negative_exponentials(const double[::1] margins):
    """Return `exp(-margins)`, see `_clamped_exponent`."""
    cdef Py_ssize_t i
    exponentials = np.empty(margins.shape[0], dtype=np.float64)
    cdef double[::1] exponentials_view = exponentials
    with nogil:
        for i in range(margins.shape[0]):
            exponentials_view[i] = _clamped_exponent(-margins[i])
    _exp_in_place(exponentials, exponentials_view)
    return exponentials


cdef void _check_rows(
    Py_ssize_t n_rows,
    const double[::1] loss_const1,
    const double[::1] loss_const2,
    str description='features has {} rows',
) except *:
    if loss_const1.shape[0] != n_rows or loss_const2.shape[0] != n_rows:
        raise ValueError(
            f'{description.format(n_rows)}, but loss_const1 and loss_const2 have '
            f'{loss_const1.shape[0]} and {loss_const2.shape[0]} entries.'
        )


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


# The logit kernels make a single pass over the rows of `features`. The rows are taken in blocks
# small enough to stay in cache between the two matrix-vector products that read them, one for the
# margins and one for the gradient, so each call reads `features` from memory once rather than twice.
# The data term of a row, and its derivative with respect to the row's margin, are computed in
# between from the two loss constants of the row:
#
#     cost:      c1 * p + c2 * (1 - p),                    derivative (c1 - c2) * expit'(m)
#     log cost:  c1 * log(p) + c2 * log(1 - p),            derivative c1 * (1 - p) - c2 * p
#
# with both probabilities of the log cost clipped to [eps, 1 - eps] first, and its derivative taken
# from the unclipped ones.
#
# Consecutive blocks are grouped into at most `_MAX_CHUNKS` chunks, which are spread over
# `n_threads` threads. Each chunk sums its loss and gradient on its own, and the chunk sums are added
# up in order afterwards. The chunks depend only on the shape of `features`, so the result is the
# same, to the last bit, whatever the number of threads.

cdef enum _DataTerm:
    _COST
    _LOG_COST


cdef Py_ssize_t _MAX_CHUNKS = 64


cdef struct _Pass:
    _DataTerm data_term
    const double* weights
    const double* features
    const double* loss_const1
    const double* loss_const2
    Py_ssize_t n_rows
    Py_ssize_t n_cols
    Py_ssize_t block_rows
    Py_ssize_t n_blocks
    Py_ssize_t n_chunks
    bint want_loss
    bint want_gradient


cdef inline Py_ssize_t _block_rows(Py_ssize_t n_cols) noexcept nogil:
    """Return the rows per block: at most 256, and at most about 1 MiB of features."""
    if n_cols <= 512:
        return 256
    return max(16, 131072 // n_cols)


cdef inline void _matrix_vector(
    bint transpose, const double* matrix, int n_rows, int n_cols, const double* vector, double beta, double* out
) noexcept nogil:
    """Set `out` to `matrix @ vector + beta * out`, or to `matrix.T @ vector + beta * out` if `transpose`.

    `matrix` is row-major, with at least one row and one column.
    """
    cdef int one = 1
    cdef double alpha = 1.0
    # BLAS reads matrices column by column, so it sees this row-major matrix as its transpose.
    cdef char blas_transpose = b'N' if transpose else b'T'
    dgemv(
        &blas_transpose, &n_cols, &n_rows, &alpha, <double*>matrix, &n_cols,
        <double*>vector, &one, &beta, out, &one,
    )


cdef double _block(const _Pass* p, Py_ssize_t start, int n_rows, double* buffer, double* gradient) noexcept nogil:
    """Return the summed loss of `n_rows` rows from `start` on, and add their gradient to `gradient`.

    `buffer` holds `3 * block_rows` values. The first `n_rows` hold the margins, then the
    exponentials, then the derivative of each row's loss with respect to its margin. The log cost
    keeps the clipped probabilities, and then their logarithms, in the other two thirds.
    """
    cdef int i
    cdef int n_cols = <int>p.n_cols
    cdef const double* features = p.features + start * p.n_cols
    cdef const double* loss_const1 = p.loss_const1 + start
    cdef const double* loss_const2 = p.loss_const2 + start
    cdef double* log_probabilities = buffer + p.block_rows
    cdef double* log_complements = buffer + 2 * p.block_rows
    cdef double loss = 0.0
    cdef double exponential, probability, complement
    # Zeroed even though BLAS overwrites it, since some BLAS libraries scale `out` by beta = 0, and
    # 0 * NaN left over from the previous block would then be NaN.
    for i in range(n_rows):
        buffer[i] = 0.0
    if n_cols:
        _matrix_vector(False, features, n_rows, n_cols, p.weights, 0.0, buffer)
    # exp and log each get a loop of their own. Called from the loop below, which keeps many more
    # values live, they cost up to 70% more per value with MSVC.
    for i in range(n_rows):
        buffer[i] = _clamped_exponent(-buffer[i])
    for i in range(n_rows):
        buffer[i] = exp(buffer[i])
    if p.data_term == _COST:
        for i in range(n_rows):
            exponential = buffer[i]
            probability = 1.0 / (1.0 + exponential)
            complement = exponential * probability
            if p.want_loss:
                loss += probability * loss_const1[i] + complement * loss_const2[i]
            buffer[i] = complement * probability * (loss_const1[i] - loss_const2[i])
    else:
        for i in range(n_rows):
            exponential = buffer[i]
            probability = 1.0 / (1.0 + exponential)
            complement = exponential * probability
            buffer[i] = loss_const1[i] * complement - loss_const2[i] * probability
            log_probabilities[i] = _clipped_probability(probability)
            log_complements[i] = _clipped_probability(complement)
        if p.want_loss:
            for i in range(n_rows):
                log_probabilities[i] = log(log_probabilities[i])
            for i in range(n_rows):
                log_complements[i] = log(log_complements[i])
            for i in range(n_rows):
                loss += loss_const1[i] * log_probabilities[i] + loss_const2[i] * log_complements[i]
    if p.want_gradient and n_cols:
        _matrix_vector(True, features, n_rows, n_cols, buffer, 1.0, gradient)
    return loss


cdef double _chunk(const _Pass* p, Py_ssize_t chunk, double* buffer, double* gradient) noexcept nogil:
    """Return the summed loss of one chunk's rows, and add their gradient to `gradient`."""
    cdef Py_ssize_t start = (chunk * p.n_blocks // p.n_chunks) * p.block_rows
    cdef Py_ssize_t end = min(((chunk + 1) * p.n_blocks // p.n_chunks) * p.block_rows, p.n_rows)
    cdef Py_ssize_t stop
    cdef double loss = 0.0
    while start < end:
        stop = min(start + p.block_rows, end)
        loss += _block(p, start, <int>(stop - start), buffer, gradient)
        start = stop
    return loss


cdef double _data_term(
    _DataTerm data_term,
    const double[::1] weights,
    const double[:, ::1] features,
    const double[::1] loss_const1,
    const double[::1] loss_const2,
    double[::1] gradient,
    bint want_loss,
    bint want_gradient,
    int n_threads,
) except *:
    """Return the mean data loss. If `want_gradient`, also store its gradient in `gradient`."""
    cdef _Pass p
    cdef Py_ssize_t chunk, j
    cdef double loss = 0.0
    p.n_rows = features.shape[0]
    p.n_cols = features.shape[1]
    if weights.shape[0] != p.n_cols:
        raise ValueError(f'weights has {weights.shape[0]} entries, but features has {p.n_cols} columns.')
    _check_rows(p.n_rows, loss_const1, loss_const2)
    p.data_term = data_term
    p.weights = &weights[0] if p.n_cols else NULL
    p.features = &features[0, 0] if p.n_rows else NULL
    p.loss_const1 = &loss_const1[0] if p.n_rows else NULL
    p.loss_const2 = &loss_const2[0] if p.n_rows else NULL
    p.block_rows = _block_rows(p.n_cols)
    p.n_blocks = (p.n_rows + p.block_rows - 1) // p.block_rows
    p.n_chunks = min(p.n_blocks, _MAX_CHUNKS)
    p.want_loss = want_loss
    p.want_gradient = want_gradient
    n_threads = max(1, n_threads)

    buffers = np.empty((p.n_chunks, 3 * p.block_rows), dtype=np.float64)
    partial_losses = np.empty(p.n_chunks, dtype=np.float64)
    partial_gradients = np.zeros((p.n_chunks, p.n_cols if want_gradient else 0), dtype=np.float64)
    cdef double[:, ::1] buffers_view = buffers
    cdef double[::1] partial_losses_view = partial_losses
    cdef double[:, ::1] partial_gradients_view = partial_gradients
    with nogil:
        for chunk in prange(
            p.n_chunks, schedule='dynamic', num_threads=n_threads, use_threads_if=n_threads > 1 and p.n_chunks > 1
        ):
            partial_losses_view[chunk] = _chunk(
                &p, chunk, &buffers_view[chunk, 0], &partial_gradients_view[chunk, 0]
            )
        for chunk in range(p.n_chunks):
            loss += partial_losses_view[chunk]
        if want_gradient and p.n_rows:
            for chunk in range(p.n_chunks):
                for j in range(p.n_cols):
                    gradient[j] += partial_gradients_view[chunk, j]
            for j in range(p.n_cols):
                gradient[j] /= p.n_rows
    return loss / p.n_rows


def cy_logit_loss_gradient(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
        int n_threads = 1,
):
    cdef double loss
    gradient = np.zeros(features.shape[1], dtype=np.float64)
    cdef double[::1] gradient_view = gradient
    loss = _data_term(_COST, weights, features, loss_const1, loss_const2, gradient_view, True, True, n_threads)
    with nogil:
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
        int n_threads = 1,
):
    cdef double loss = _data_term(_COST, weights, features, loss_const1, loss_const2, None, True, False, n_threads)
    with nogil:
        loss += _penalty(weights, l1_weight, l2_weight, start_coef)
    return loss


def cy_logit_gradient(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
        int n_threads = 1,
):
    gradient = np.zeros(features.shape[1], dtype=np.float64)
    cdef double[::1] gradient_view = gradient
    _data_term(_COST, weights, features, loss_const1, loss_const2, gradient_view, False, True, n_threads)
    with nogil:
        _add_penalty_gradient(weights, gradient_view, l1_weight, l2_weight, start_coef)
    return gradient


def cy_boost_grad_hess(y_true, ScoreType y_score, GradientType grad_const):
    # The hessian is |grad_const| * p * (1 - p), not the exact second derivative, which carries an
    # extra |1 - 2p|: that vanishes where boosting starts (p near 1/2), so a booster's minimum child
    # weight stopped trees from splitting until the predictions had moved away from 1/2.
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
    _exp_in_place(hessian, hessian_view)
    with nogil:
        for i in range(n_rows):
            exponential = hessian_view[i]
            probability = 1.0 / (1.0 + exponential)
            complement = exponential * probability
            gradient_i = complement * probability * grad_const[i]
            gradient_view[i] = gradient_i
            hessian_view[i] = fabs(gradient_i)

    return gradient, hessian


# The second derivative of the log cost with respect to the margin, used by the boosting kernel
# below, is (c1 + c2) * p * (1 - p).


def cy_log_cost_loss_gradient(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
        int n_threads = 1,
):
    cdef double loss
    gradient = np.zeros(features.shape[1], dtype=np.float64)
    cdef double[::1] gradient_view = gradient
    loss = _data_term(_LOG_COST, weights, features, loss_const1, loss_const2, gradient_view, True, True, n_threads)
    with nogil:
        loss += _penalty(weights, l1_weight, l2_weight, start_coef)
        _add_penalty_gradient(weights, gradient_view, l1_weight, l2_weight, start_coef)
    return loss, gradient


def cy_log_cost_loss(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
        int n_threads = 1,
):
    cdef double loss = _data_term(_LOG_COST, weights, features, loss_const1, loss_const2, None, True, False, n_threads)
    with nogil:
        loss += _penalty(weights, l1_weight, l2_weight, start_coef)
    return loss


def cy_log_cost_gradient(
        const double[::1] weights,
        const double[:, ::1] features,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
        double l1_weight = 0.0,
        double l2_weight = 0.0,
        Py_ssize_t start_coef = 1,
        int n_threads = 1,
):
    gradient = np.zeros(features.shape[1], dtype=np.float64)
    cdef double[::1] gradient_view = gradient
    _data_term(_LOG_COST, weights, features, loss_const1, loss_const2, gradient_view, False, True, n_threads)
    with nogil:
        _add_penalty_gradient(weights, gradient_view, l1_weight, l2_weight, start_coef)
    return gradient


def cy_log_cost_boost_grad_hess(
        const double[::1] y_score,
        const double[::1] loss_const1,
        const double[::1] loss_const2,
):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = y_score.shape[0]
    cdef double exponential, probability, complement
    _check_rows(n_rows, loss_const1, loss_const2, 'y_score has {} entries')
    gradient = np.empty(n_rows, dtype=np.float64)
    hessian = _negative_exponentials(y_score)
    cdef double[::1] gradient_view = gradient
    cdef double[::1] hessian_view = hessian
    with nogil:
        # The exponentials are replaced by the hessian as they are read.
        for i in range(n_rows):
            exponential = hessian_view[i]
            probability = 1.0 / (1.0 + exponential)
            complement = exponential * probability
            gradient_view[i] = loss_const1[i] * complement - loss_const2[i] * probability
            hessian_view[i] = fabs(loss_const1[i] + loss_const2[i]) * complement * probability
    return gradient, hessian
