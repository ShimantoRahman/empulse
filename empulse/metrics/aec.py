from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import _check_consistent_length, _check_y_true, _check_y_pred


def _compute_expected_cost(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        tp_cost: Union[ArrayLike, float] = 0.0,
        tn_cost: Union[ArrayLike, float] = 0.0,
        fn_cost: Union[ArrayLike, float] = 0.0,
        fp_cost: Union[ArrayLike, float] = 0.0,
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

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.

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

    if not isinstance(tp_cost, (int, float)):
        tp_cost = np.asarray(tp_cost)
    if not isinstance(tn_cost, (int, float)):
        tn_cost = np.asarray(tn_cost)
    if not isinstance(fn_cost, (int, float)):
        fn_cost = np.asarray(fn_cost)
    if not isinstance(fp_cost, (int, float)):
        fp_cost = np.asarray(fp_cost)

    if check_input:
        _check_consistent_length(
            *(array for array in
              (y_true, y_pred, tp_cost, tn_cost, fn_cost, fp_cost) if isinstance(array, np.ndarray))
        )

    return y_true * (y_pred * tp_cost + (1 - y_pred) * fn_cost) \
        + (1 - y_true) * (y_pred * fp_cost + (1 - y_pred) * tn_cost)


def aec_loss(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_cost: Union[ArrayLike, float] = 0.0,
        tn_cost: Union[ArrayLike, float] = 0.0,
        fn_cost: Union[ArrayLike, float] = 0.0,
        fp_cost: Union[ArrayLike, float] = 0.0,
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

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    average_expected_cost : float
        Average expected cost.
    """
    aec = _compute_expected_cost(y_true, y_pred, tp_cost, tn_cost, fn_cost, fp_cost, check_input=check_input)
    return aec.mean()


def log_aec_loss(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_cost: Union[ArrayLike, float] = 0.0,
        tn_cost: Union[ArrayLike, float] = 0.0,
        fn_cost: Union[ArrayLike, float] = 0.0,
        fp_cost: Union[ArrayLike, float] = 0.0,
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

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    log_average_expected_cost : float
        Log average expected cost.
    """
    aec = _compute_expected_cost(y_true, y_pred, tp_cost, tn_cost, fn_cost, fp_cost, check_input=check_input)
    epsilon = np.finfo(aec.dtype).eps  # avoid division by zero
    return np.log(aec + epsilon).mean()
