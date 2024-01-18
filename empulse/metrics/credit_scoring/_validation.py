import numpy as np
from numpy.typing import ArrayLike

from .._validation import _check_shape, _check_positive, _check_fraction, _check_y_true, _check_y_pred


def _validate_input(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        roi: float
) -> tuple[np.ndarray, np.ndarray]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    _check_positive(roi, 'roi')
    return y_true, y_pred


def _validate_input_mp(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        default_prob: float,
        roi: float
) -> tuple[np.ndarray, np.ndarray]:
    _check_fraction(default_prob, 'default_prob')
    return _validate_input(y_true, y_pred, roi)


def _validate_input_emp(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        p_0: float,
        p_1: float,
        roi: float
) -> tuple[np.ndarray, np.ndarray]:
    _check_fraction(p_0, 'p_0')
    _check_fraction(p_1, 'p_1')
    return _validate_input(y_true, y_pred, roi)
