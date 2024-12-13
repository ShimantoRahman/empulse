from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import _check_consistent_length, _check_y_true, _check_y_pred


def _compute_expected_cost(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        tp_costs: Optional[ArrayLike] = None,
        tn_costs: Optional[ArrayLike] = None,
        fn_costs: Optional[ArrayLike] = None,
        fp_costs: Optional[ArrayLike] = None,
        validation: bool = True,
) -> NDArray:
    """
    Compute expected cost for binary classification.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        True labels.
    y_pred : 1D array-like, shape=(n_samples,)
        Predicted probabilities.
    tp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true positive predictions.
    tn_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true negative predictions.
    fn_costs : 1D array-like, shape=(n_samples,), optional
        Costs for false negative predictions.
    fp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for false positive predictions.
    validation : bool, default=True
        Perform input validation. Turning off improves performance

    Returns
    -------
    expected_costs : 1D numpy.ndarray, shape=(n_samples,)
        expected costs.
    """
    if validation:
        y_true = _check_y_true(y_true)
        y_pred = _check_y_pred(y_pred)

        if all(costs is None for costs in (tp_costs, tn_costs, fn_costs, fp_costs)):
            raise ValueError("At least one of tp_costs, tn_costs, fn_costs, fp_costs must be provided.")
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

    if tp_costs is None:
        tp_costs = 0.0
    else:
        tp_costs = np.asarray(tp_costs)
    if tn_costs is None:
        tn_costs = 0.0
    else:
        tn_costs = np.asarray(tn_costs)
    if fn_costs is None:
        fn_costs = 0.0
    else:
        fn_costs = np.asarray(fn_costs)
    if fp_costs is None:
        fp_costs = 0.0
    else:
        fp_costs = np.asarray(fp_costs)

    if validation:
        _check_consistent_length(
            *(array for array in
              (y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs) if isinstance(array, np.ndarray))
        )

    return y_true * (y_pred * tp_costs + (1 - y_pred) * fn_costs) \
        + (1 - y_true) * (y_pred * fp_costs + (1 - y_pred) * tn_costs)


def aec_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_costs: Optional[ArrayLike] = None,
        tn_costs: Optional[ArrayLike] = None,
        fn_costs: Optional[ArrayLike] = None,
        fp_costs: Optional[ArrayLike] = None,
        validation: bool = True
) -> float:
    """
    Compute average expected cost for binary classification.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        True labels.
    y_pred : 1D array-like, shape=(n_samples,)
        Predicted probabilities.
    tp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true positive predictions.
    tn_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true negative predictions.
    fn_costs : 1D array-like, shape=(n_samples,), optional
        Costs for false negative predictions.
    fp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for false positive predictions.
    validation : bool, default=True
        Perform input validation. Turning off improves performance.

    Returns
    -------
    average_expected_cost : float
        Average expected cost.
    """
    aec = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, validation=validation)
    return aec.mean()


def log_aec_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_costs: Optional[ArrayLike] = None,
        tn_costs: Optional[ArrayLike] = None,
        fn_costs: Optional[ArrayLike] = None,
        fp_costs: Optional[ArrayLike] = None,
        validation: bool = True,
) -> float:
    """
    Compute log average expected cost for binary classification.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        True labels.
    y_pred : 1D array-like, shape=(n_samples,)
        Predicted probabilities.
    tp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true positive predictions.
    tn_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true negative predictions.
    fn_costs : 1D array-like, shape=(n_samples,), optional
        Costs for false negative predictions.
    fp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for false positive predictions.
    validation : bool, default=True
        Perform input validation. Turning off improves performance.

    Returns
    -------
    log_average_expected_cost : float
        Log average expected cost.
    """
    aec = _compute_expected_cost(y_true, y_pred, tp_costs, tn_costs, fn_costs, fp_costs, validation=validation)
    epsilon = np.finfo(aec.dtype).eps  # avoid division by zero
    return np.log(aec + epsilon).mean()
