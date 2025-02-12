import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import column_or_1d


def _check_y_true(y_true: ArrayLike) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_true = column_or_1d(y_true)
    _check_numeric(y_true)
    _check_nan(y_true)
    _check_inf(y_true)
    _check_binary(y_true)
    _check_variance(y_true)
    return y_true


def _check_y_pred(y_pred: ArrayLike) -> np.ndarray:
    y_pred = np.asarray(y_pred)
    y_pred = column_or_1d(y_pred)
    _check_nan(y_pred)
    _check_inf(y_pred)
    return y_pred


def _check_nan(array: NDArray) -> None:
    if np.isnan(array).any():
        raise ValueError(f'The array should not contain NaN values, got {np.sum(np.isnan(array))} instead.')


def _check_inf(array: NDArray) -> None:
    if np.isinf(array).any():
        raise ValueError(f'The array should not contain Inf values, got {np.sum(np.isinf(array))} instead.')


def _check_shape(y_true: NDArray, y_pred: NDArray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f'The shapes of the true label and predictions should match, '
            f'got shape y_true={y_true.shape} and shape y_pred={y_pred.shape} instead.'
        )


def _check_consistent_length(*arrays: NDArray) -> None:
    """
    Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            'Found input variables with inconsistent numbers of samples: %r' % [int(length) for length in lengths]
        )


def _check_numeric(y_true: NDArray) -> None:
    if not np.issubdtype(y_true.dtype, np.number):
        raise TypeError(f'The true labels should be numeric, got dtype {y_true.dtype} instead.')


def _check_binary(y_true: NDArray) -> None:
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(f'The true labels should be binary, got unique values {np.unique(y_true)} instead.')


def _check_variance(array: NDArray) -> None:
    unique_values = np.unique(array)
    if len(unique_values) < 2:
        raise ValueError(
            f'The array should have at least two different values, got unique values {unique_values} instead.'
        )


def _check_positive(var: float | int, var_name: str) -> None:
    if var < 0:
        raise ValueError(f'{var_name} should be positive, got a value of {var} instead.')


def _check_gt_one(var: float | int, var_name: str) -> None:
    if var < 0:
        raise ValueError(f'{var_name} should be greater than 1, got a value of {var} instead.')


def _check_fraction(var: float, var_name: str) -> None:
    if not (0 <= var <= 1):
        raise ValueError(f'{var_name} should lay between 0 and 1, got a value of {var} instead.')
