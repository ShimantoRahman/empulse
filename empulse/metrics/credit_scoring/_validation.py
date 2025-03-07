from ..._types import FloatArrayLike, FloatNDArray
from .._validation import _check_fraction, _check_positive, _check_shape, _check_y_pred, _check_y_true


def _validate_input(y_true: FloatArrayLike, y_pred: FloatArrayLike, roi: float) -> tuple[FloatNDArray, FloatNDArray]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    _check_positive(roi, 'roi')
    return y_true, y_pred


def _validate_input_mp(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, default_prob: float, roi: float
) -> tuple[FloatNDArray, FloatNDArray]:
    _check_fraction(default_prob, 'default_prob')
    return _validate_input(y_true, y_pred, roi)


def _validate_input_emp(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, p_0: float, p_1: float, roi: float
) -> tuple[FloatNDArray, FloatNDArray]:
    _check_fraction(p_0, 'p_0')
    _check_fraction(p_1, 'p_1')
    return _validate_input(y_true, y_pred, roi)
