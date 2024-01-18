import numpy as np
from numpy.typing import ArrayLike

from .._validation import _check_shape, _check_positive, _check_fraction, _check_y_true, _check_y_pred


def _validate_input(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        contact_cost: float,
        sales_cost: float,
        direct_selling: float,
        commission: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate input for all acquisition parameters."""
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    _check_positive(contact_cost, 'contact_cost')
    _check_positive(sales_cost, 'sales_cost')
    _check_fraction(direct_selling, 'direct_selling')
    _check_fraction(commission, 'commission')

    return y_true, y_pred
