import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import column_or_1d
from ._validation import _check_shape, _check_binary, _check_fraction, _check_variance


def _validate_input(y_true: ArrayLike, y_pred: ArrayLike, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    y_true = column_or_1d(np.asarray(y_true))
    y_pred = column_or_1d(np.asarray(y_pred))
    _check_binary(y_true)
    _check_variance(y_true)
    _check_shape(y_true, y_pred)
    _check_fraction(fraction, 'fraction')

    return y_true, y_pred


def lift_score(y_true: ArrayLike, y_pred: ArrayLike, fraction: float = 0.1) -> float:
    """
    Compute the lift score for the top fraction of predictions.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    fraction : float, optional, default: 0.1
        Fraction of data to consider. Must be between 0 and 1.

    Returns
    -------
    lift_score : float
        Lift score for the top fraction of the data.
    """
    y_true, y_pred = _validate_input(y_true, y_pred, fraction)

    # Sort the predictions in descending order
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_labels = y_true[sorted_indices]
    top_fraction = int(round(len(sorted_labels) * fraction, 0))

    n_positives_top_fraction = np.sum(sorted_labels[:top_fraction])
    prop_positives_top_fraction = n_positives_top_fraction / top_fraction

    return prop_positives_top_fraction / np.mean(y_true)
