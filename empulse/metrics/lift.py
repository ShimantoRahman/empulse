import numpy as np
from numpy.typing import ArrayLike

from ._validation import _check_shape, _check_binary, _check_fraction, _check_variance, _check_y_true, _check_y_pred


def _validate_input(y_true: ArrayLike, y_score: ArrayLike, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    y_true = _check_y_true(y_true)
    y_score = _check_y_pred(y_score)
    _check_shape(y_true, y_score)
    _check_binary(y_true)
    _check_variance(y_true)
    _check_shape(y_true, y_score)
    _check_fraction(fraction, 'fraction')

    return y_true, y_score


def lift_score(y_true: ArrayLike, y_score: ArrayLike, *, fraction: float = 0.1, check_input: bool = True) -> float:
    """
    Compute the lift score for the top fraction of predictions.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    fraction : float, optional, default: 0.1
        Fraction of data to consider. Must be between 0 and 1.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    lift_score : float
        Lift score for the top fraction of the data.
    """
    if check_input:
        y_true, y_score = _validate_input(y_true, y_score, fraction)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Sort the predictions in descending order
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_indices]
    top_fraction = int(round(len(sorted_labels) * fraction, 0))

    n_positives_top_fraction = np.sum(sorted_labels[:top_fraction])
    prop_positives_top_fraction = n_positives_top_fraction / top_fraction

    return prop_positives_top_fraction / np.mean(y_true)
