import numbers
from typing import TypeVar

import numpy as np
from numpy.typing import NBitBase

from ..._types import FloatArrayLike, FloatNDArray
from .._validation import _check_fraction, _check_gt_one, _check_positive, _check_shape, _check_y_pred, _check_y_true

T = TypeVar('T', bound=NBitBase)


def _validate_input(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, clv: float | FloatArrayLike, d: float, f: float
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    if not isinstance(clv, numbers.Real):
        clv = np.asarray(clv)
        mean_clv = np.mean(clv)
        _check_positive(float(mean_clv), 'clv')
    else:
        _check_positive(clv, 'clv')
    _check_positive(d, 'incentive_cost')
    _check_positive(f, 'contact_cost')
    if isinstance(clv, numbers.Real) and clv <= d:
        raise ValueError(f'clv should be greater than d, got a value of {clv} for clv and for {d} instead.')
    if isinstance(clv, np.ndarray) and np.mean(clv) <= d:
        raise ValueError(f'mean clv should be greater than d, got a value of {clv} for mean clv and for {d} instead.')

    return y_true, y_pred, clv


def _validate_input_emp(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    alpha: float,
    beta: float,
    clv: float | FloatArrayLike,
    d: float,
    f: float,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    _check_gt_one(alpha, 'alpha')
    _check_gt_one(beta, 'beta')
    return _validate_input(y_true, y_pred, clv, d, f)


def _validate_input_mp(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, gamma: float, clv: float | FloatArrayLike, d: float, f: float
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    _check_fraction(gamma, 'gamma')
    return _validate_input(y_true, y_pred, clv, d, f)


def _validate_input_mpc(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    clv: FloatArrayLike,
    accept_rate: float,
    incentive_fraction: float,
    contact_cost: float,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    clv = np.asarray(clv)
    _check_fraction(accept_rate, 'accept_rate')
    _check_fraction(incentive_fraction, 'incentive_fraction')
    _check_positive(contact_cost, 'contact_cost')

    return y_true, y_pred, clv


def _validate_input_empb(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    clv: FloatArrayLike,
    alpha: float,
    beta: float,
    incentive_fraction: float,
    contact_cost: float,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    clv = np.asarray(clv)
    _check_positive(alpha, 'alpha')
    _check_positive(beta, 'beta')
    _check_fraction(incentive_fraction, 'incentive_fraction')
    _check_positive(contact_cost, 'contact_cost')

    return y_true, y_pred, clv


def _validate_input_cost_loss_churn(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    clv: FloatArrayLike,
    accept_rate: float,
    incentive_fraction: float | FloatArrayLike,
    contact_cost: float,
) -> tuple[
    FloatNDArray,
    FloatNDArray,
    FloatNDArray,
    FloatNDArray | float,
]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    clv = np.asarray(clv)
    _check_fraction(accept_rate, 'accept_rate')
    if isinstance(incentive_fraction, float | int):
        _check_fraction(incentive_fraction, 'incentive_fraction')
    else:
        incentive_fraction = np.asarray(incentive_fraction)
    _check_positive(contact_cost, 'contact_cost')

    return y_true, y_pred, clv, incentive_fraction
