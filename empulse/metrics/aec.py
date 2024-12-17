from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import _check_consistent_length, _check_y_true, _check_y_pred


def _compute_expected_cost(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        tp_costs: Union[ArrayLike, float] = 0.0,
        tn_costs: Union[ArrayLike, float] = 0.0,
        fn_costs: Union[ArrayLike, float] = 0.0,
        fp_costs: Union[ArrayLike, float] = 0.0,
        check_input: bool = True,
) -> NDArray:
    """
    Compute expected cost for binary classification.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        True labels.

    y_pred : 1D array-like, shape=(n_samples,)
        Predicted probabilities.

    tp_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for true positive predictions.

    tn_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for true negative predictions.

    fn_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for false negative predictions.

    fp_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for false positive predictions.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.

    Returns
    -------
    expected_costs : 1D numpy.ndarray, shape=(n_samples,)
        Average expected costs.
    """
    if check_input:
        y_true = _check_y_true(y_true)
        y_pred = _check_y_pred(y_pred)
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

    if not isinstance(tp_costs, (int, float)):
        tp_costs = np.asarray(tp_costs)
    if not isinstance(tn_costs, (int, float)):
        tn_costs = np.asarray(tn_costs)
    if not isinstance(fn_costs, (int, float)):
        fn_costs = np.asarray(fn_costs)
    if not isinstance(fp_costs, (int, float)):
        fp_costs = np.asarray(fp_costs)

    if check_input:
        _check_consistent_length(
            *(array for array in
              (y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs) if isinstance(array, np.ndarray))
        )

    return y_true * (y_pred * tp_costs + (1 - y_pred) * fn_costs) \
        + (1 - y_true) * (y_pred * fp_costs + (1 - y_pred) * tn_costs)


def aec_loss(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_costs: Union[ArrayLike, float] = 0.0,
        tn_costs: Union[ArrayLike, float] = 0.0,
        fn_costs: Union[ArrayLike, float] = 0.0,
        fp_costs: Union[ArrayLike, float] = 0.0,
        check_input: bool = True
) -> float:
    """
    Compute average expected cost for binary classification.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        True labels.

    y_pred : 1D array-like, shape=(n_samples,)
        Predicted probabilities.

    tp_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for true positive predictions.

    tn_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for true negative predictions.

    fn_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for false negative predictions.

    fp_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for false positive predictions.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    average_expected_cost : float
        Average expected cost.
    """
    aec = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, check_input=check_input)
    return aec.mean()


def log_aec_loss(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_costs: Union[ArrayLike, float] = 0.0,
        tn_costs: Union[ArrayLike, float] = 0.0,
        fn_costs: Union[ArrayLike, float] = 0.0,
        fp_costs: Union[ArrayLike, float] = 0.0,
        check_input: bool = True,
) -> float:
    """
    Compute log average expected cost for binary classification.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Predicted probabilities.

    tp_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for true positive predictions.

    tn_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for true negative predictions.

    fn_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for false negative predictions.

    fp_costs : float or 1D array-like, shape=(n_samples,), optional
        Cost(s) for false positive predictions.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    log_average_expected_cost : float
        Log average expected cost.
    """
    aec = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, check_input=check_input)
    epsilon = np.finfo(aec.dtype).eps  # avoid division by zero
    return np.log(aec + epsilon).mean()
