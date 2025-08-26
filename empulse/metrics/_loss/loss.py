# mypy: ignore-errors
import cython
import numpy as np
from cython.cimports import numpy as cnp  # noqa: F401
from cython.cimports.libc.math import exp, fabs
from numpy.typing import NDArray

ScoreType = cython.fused_type(cython.float[:], cython.double[:])
GradientType = cython.fused_type(cython.float[:], cython.double[:])


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
@cython.cdivision(True)
def expit(x: cython.double) -> cython.double:
    return 1.0 / (1.0 + exp(-x))


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def sign(x: cython.double) -> cython.double:
    is_positive: cython.bint = x > 0
    if is_positive:
        return 1.0
    else:
        return -1.0


@cython.cdivision(True)  # turn off check zero division error (n_rows is always positive)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_logit_loss_gradient(
    weights: cython.double[:],
    features: cython.double[:, :],
    grad_const: cython.double[:, :],
    loss_const1: cython.double[:],
    loss_const2: cython.double[:],
    C: cython.double = 1.0,
    l1_ratio: cython.double = 0.0,
    fit_intercept: cython.bint = True,
    soft_threshold: cython.bint = False,
) -> tuple[float, NDArray[np.float64]]:
    i: cython.int
    j: cython.int
    n_rows = cython.declare(cython.int, features.shape[0])
    n_cols = cython.declare(cython.int, features.shape[1])

    start_coef: cython.int = 1 if fit_intercept else 0  # index of where the coefficients start (not intercept)

    loss: cython.double = 0.0
    gradient = np.zeros(n_cols, dtype=np.float64)
    gradient_view = cython.declare(cython.double[:], gradient)
    y_score = np.zeros(n_rows, dtype=np.float64)
    y_score_view: cython.double[:] = y_score

    for i in range(n_rows):
        for j in range(n_cols):
            y_score_view[i] += weights[j] * features[i, j]
        y_score_view[i] = expit(y_score_view[i])

    if soft_threshold:
        weights = weights.copy()
        for j in range(start_coef, n_cols):
            absolute_val: cython.double = fabs(weights[j])
            if (absolute_val - C) > 0:
                weights[j] = sign(weights[j]) * (absolute_val - C)
            elif (absolute_val - C) < 0:
                weights[j] = 0.0

    # compute gradient
    for i in range(n_rows):
        s: cython.double = y_score_view[i] * (1 - y_score_view[i])
        for j in range(n_cols):
            gradient_view[j] += s * grad_const[i, j]
    # apply regularization
    if l1_ratio == 0.0:  # L2 regularization penalty
        for j in range(start_coef, n_cols):
            gradient_view[j] = gradient_view[j] / n_rows
            gradient_view[j] += weights[j]
    elif l1_ratio == 1.0:  # L1 regularization penalty
        for j in range(start_coef, n_cols):
            gradient_view[j] = gradient_view[j] / n_rows
            gradient_view[j] += fabs(weights[j])
    else:
        for j in range(start_coef, n_cols):  # elastic net regularization penalty
            gradient_view[j] = gradient_view[j] / n_rows
            gradient_view[j] += (1 - l1_ratio) * weights[j] + l1_ratio * fabs(weights[j])
    if fit_intercept:
        gradient_view[0] = gradient_view[0] / n_rows  # intercept term is not regularized

    # compute loss
    for i in range(n_rows):
        loss += y_score_view[i] * loss_const1[i] + (1 - y_score_view[i]) * loss_const2[i]
    loss = loss / n_rows
    # apply regularization
    if l1_ratio == 0.0:  # L2 regularization penalty
        for j in range(start_coef, n_cols):
            loss += 0.5 * weights[j] ** 2 / C
    elif l1_ratio == 1.0:  # L1 regularization penalty
        for j in range(start_coef, n_cols):
            loss += fabs(weights[j]) / C
    else:
        for j in range(start_coef, n_cols):  # elastic net regularization penalty
            loss += (1 - l1_ratio) * 0.5 * weights[j] ** 2 / C + l1_ratio * fabs(weights[j]) / C
    return loss, gradient


@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_boost_grad_hess(
    y_true: cython.int[:], y_score: ScoreType, grad_const: GradientType
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    i: cython.int
    n_rows = cython.declare(cython.int, grad_const.shape[0])

    gradient = np.zeros(n_rows, dtype=np.float64)
    gradient_view = cython.declare(cython.double[:], gradient)
    hessian = np.zeros(n_rows, dtype=np.float64)
    hessian_view = cython.declare(cython.double[:], hessian)

    for i in range(n_rows):
        y_proba: cython.double = expit(y_score[i])
        s: cython.double = y_proba * (1 - y_proba)
        gradient_view[i] = s * grad_const[i]
        hessian_view[i] = fabs((1 - 2 * y_proba) * gradient_view[i])

    return gradient, hessian
