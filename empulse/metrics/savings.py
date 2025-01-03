import numbers
from functools import partial, update_wrapper
from typing import Union, Callable, Literal

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from ._validation import _check_y_true, _check_y_pred, _check_consistent_length


def _validate_input(y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, check_input):
    if check_input:
        y_true = _check_y_true(y_true)
        y_pred = _check_y_pred(y_pred)
        arrays = [y_true, y_pred]
        if not isinstance(tp_cost, numbers.Number):
            tp_cost = np.asarray(tp_cost)
            arrays.append(tp_cost)
        if not isinstance(fp_cost, numbers.Number):
            fp_cost = np.asarray(fp_cost)
            arrays.append(fp_cost)
        if not isinstance(tn_cost, numbers.Number):
            tn_cost = np.asarray(tn_cost)
            arrays.append(tn_cost)
        if not isinstance(fn_cost, numbers.Number):
            fn_cost = np.asarray(fn_cost)
            arrays.append(fn_cost)
        _check_consistent_length(*arrays)
        if len(arrays) == 2 and all(cost == 0.0 for cost in (tp_cost, fp_cost, fn_cost, tn_cost)):
            raise ValueError("All costs are zero. At least one cost must be non-zero.")
        return y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if not isinstance(tp_cost, numbers.Number):
            tp_cost = np.asarray(tp_cost)
        if not isinstance(fp_cost, numbers.Number):
            fp_cost = np.asarray(fp_cost)
        if not isinstance(tn_cost, numbers.Number):
            tn_cost = np.asarray(tn_cost)
        if not isinstance(fn_cost, numbers.Number):
            fn_cost = np.asarray(fn_cost)
        return y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost


def _compute_expected_cost(
        y_true: NDArray,
        y_pred: NDArray,
        tp_cost: Union[NDArray, float] = 0.0,
        tn_cost: Union[NDArray, float] = 0.0,
        fn_cost: Union[NDArray, float] = 0.0,
        fp_cost: Union[NDArray, float] = 0.0,
) -> NDArray:
    return y_true * (y_pred * tp_cost + (1 - y_pred) * fn_cost) \
        + (1 - y_true) * (y_pred * fp_cost + (1 - y_pred) * tn_cost)


def _compute_log_expected_cost(
        y_true: NDArray,
        y_pred: NDArray,
        tp_cost: Union[NDArray, float] = 0.0,
        tn_cost: Union[NDArray, float] = 0.0,
        fn_cost: Union[NDArray, float] = 0.0,
        fp_cost: Union[NDArray, float] = 0.0,
) -> NDArray:
    epsilon = np.finfo(y_pred.dtype).eps
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    inverse_y_pred = 1 - y_pred
    log_y_pred = np.log(y_pred)
    log_inv_y_pred = np.log(inverse_y_pred)
    return y_true * (log_y_pred * tp_cost + log_inv_y_pred * fn_cost) \
        + (1 - y_true) * (log_y_pred * fp_cost + log_inv_y_pred * tn_cost)


def cost_loss(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_cost: Union[float, ArrayLike] = 0.0,
        fp_cost: Union[float, ArrayLike] = 0.0,
        tn_cost: Union[float, ArrayLike] = 0.0,
        fn_cost: Union[float, ArrayLike] = 0.0,
        normalize: bool = False,
        check_input: bool = True,
) -> float:
    """
    Cost of a classifier.

    The cost of a classifier is the sum of the costs of each instance.
    This allows you to give attribute specific costs (or benefits in case of negative costs)
    to each type of classification.
    For example, in a credit card fraud detection problem,
    the cost of a false negative (not detecting a fraudulent transaction) is higher than
    the cost of a false positive (flagging a non-fraudulent transaction as fraudulent).

    .. seealso::

        :func:`~empulse.metrics.expected_cost_loss` : Expected cost of a classifier.

        :func:`~empulse.metrics.savings_score` : Cost savings of a classifier compared to using a naive algorithm.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Predicted labels or calibrated probabilities.
        If the predictions are calibrated probabilities,
        the optimal decision threshold is calculated for each instance as [3]_:

        .. math:: t^*_i = \\frac{C_i(1|0) - C_i(0|0)}{C_i(1|0) - C_i(0|0) + C_i(0|1) - C_i(1|1)}

        .. note:: The optimal decision threshold is only accurate when the probabilities are well-calibrated.
                  See `scikit-learn's user guide <https://scikit-learn.org/stable/modules/calibration.html>`_
                  for more information.

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

    normalize : bool, default=False
        Normalize the cost by the number of samples.
        If ``True``, return the average cost.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    cost_loss : float
        Cost of a classifier.

    Notes
    -----
    The cost of each instance :math:`C_i` is calculated as [3]_:

    .. math:: C_i = y_i \\cdot (\\hat y_i \\cdot C_i(1|1) + (1 - \\hat y_i) \\cdot C_i(0|1)) + (1 - \\hat y_i) \\cdot (\\hat y_i \\cdot C_i(1|0) + (1 - \\hat y_i) \\cdot C_i(0|0))

    where

        - :math:`y_i` is the true label,
        - :math:`\\hat y_i` is the predicted label,
        - :math:`C_i(1|1)` is the cost of a true positive ``tp_cost``,
        - :math:`C_i(0|1)` is the cost of a false positive ``fp_cost``,
        - :math:`C_i(1|0)` is the cost of a false negative ``fn_cost``, and
        - :math:`C_i(0|0)` is the cost of a true negative ``tn_cost``.

    Code modified from `costcla.metrics.cost_loss`.

    References
    ----------
    .. [1] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    .. [2] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           "Improving Credit Card Fraud Detection with Calibrated Probabilities",
           in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    .. [3] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.

    Examples
    --------
    >>> import numpy as np
    >>> from empulse.metrics import cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> fp_cost = np.array([4, 1, 2, 2])
    >>> fn_cost = np.array([1, 3, 3, 1])
    >>> cost_loss(y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost)
    3.0
    """
    y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    # If the prediction is not binary, we need to find the optimal threshold
    if not np.all((y_pred == 0) | (y_pred == 1)):
        denominator = fp_cost - tn_cost + fn_cost - tp_cost
        denominator = np.clip(denominator, np.finfo(float).eps, denominator)  # Avoid division by zero
        optimal_thresholds = (fp_cost - tn_cost) / denominator
        y_pred = (y_pred > optimal_thresholds).astype(int)

    cost = _compute_expected_cost(y_true, y_pred, tp_cost, tn_cost, fn_cost, fp_cost)

    if normalize:
        return cost.mean()
    return np.sum(cost)


def expected_cost_loss(
        y_true: ArrayLike,
        y_proba: ArrayLike,
        *,
        tp_cost: Union[float, ArrayLike] = 0.0,
        fp_cost: Union[float, ArrayLike] = 0.0,
        tn_cost: Union[float, ArrayLike] = 0.0,
        fn_cost: Union[float, ArrayLike] = 0.0,
        normalize: bool = False,
        check_input: bool = True,
) -> float:
    """
    Expected cost of a classifier.

    The expected cost of a classifier is the sum of the expected costs of each instance.
    This allows you to give attribute specific costs (or benefits in case of negative costs)
    to each type of classification.
    For example, in a credit card fraud detection problem,
    the cost of a false negative (not detecting a fraudulent transaction) is higher than
    the cost of a false positive (flagging a non-fraudulent transaction as fraudulent).

    .. seealso::

        :func:`~empulse.metrics.cost_loss` : Cost of a classifier.

        :func:`~empulse.metrics.expected_savings_score` : Expected savings of a classifier
        compared to using a naive algorithm.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_proba : 1D array-like, shape=(n_samples,)
        Target probabilities, should lie between 0 and 1.

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

    normalize : bool, default=False
        Normalize the cost by the number of samples.
        If ``True``, return the average expected cost [3]_.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    cost_loss : float
        Cost of a classifier.

    Notes
    -----
    The expected cost of each instance :math:`\\mathbb{E}[C_i]` is calculated as [3]_:

    .. math:: \\mathbb{E}[C_i] = y_i \\cdot (s_i \\cdot C_i(1|1) + (1 - s_i) \\cdot C_i(0|1)) + (1 - s_i) \\cdot (s_i \\cdot C_i(1|0) + (1 - s_i) \\cdot C_i(0|0))

    where

    - :math:`y_i` is the true label,
    - :math:`s_i` is the predicted probability,
    - :math:`C_i(1|1)` is the cost of a true positive ``tp_cost``,
    - :math:`C_i(0|1)` is the cost of a false positive ``fp_cost``,
    - :math:`C_i(1|0)` is the cost of a false negative ``fn_cost``, and
    - :math:`C_i(0|0)` is the cost of a true negative ``tn_cost``.

    Code modified from `costcla.metrics.cost_loss`.

    References
    ----------
    .. [1] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    .. [2] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           "Improving Credit Card Fraud Detection with Calibrated Probabilities",
           in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    .. [3] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.

    Examples
    --------
    >>> import numpy as np
    >>> from empulse.metrics import expected_cost_loss
    >>> y_proba = [0.2, 0.9, 0.1, 0.2]
    >>> y_true = [0, 1, 1, 0]
    >>> fp_cost = np.array([4, 1, 2, 2])
    >>> fn_cost = np.array([1, 3, 3, 1])
    >>> expected_cost_loss(y_true, y_proba, fp_cost=fp_cost, fn_cost=fn_cost)
    4.2
    """
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    cost = _compute_expected_cost(y_true, y_proba, tp_cost, tn_cost, fn_cost, fp_cost)

    if normalize:
        return cost.mean()
    return np.sum(cost)


def expected_log_cost_loss(
        y_true: ArrayLike,
        y_proba: ArrayLike,
        *,
        tp_cost: Union[ArrayLike, float] = 0.0,
        tn_cost: Union[ArrayLike, float] = 0.0,
        fn_cost: Union[ArrayLike, float] = 0.0,
        fp_cost: Union[ArrayLike, float] = 0.0,
        normalize: bool = False,
        check_input: bool = True,
) -> float:
    """
    Expected log cost of a classifier.

    The expected log cost of a classifier is the sum of the expected log costs of each instance.
    This allows you to give attribute specific costs (or benefits in case of negative costs)
    to each type of classification.
    For example, in a credit card fraud detection problem,
    the cost of a false negative (not detecting a fraudulent transaction) is higher than
    the cost of a false positive (flagging a non-fraudulent transaction as fraudulent).

    .. seealso::

        :func:`~empulse.metrics.expected_cost_loss` : Expected cost of a classifier.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_proba : 1D array-like, shape=(n_samples,)
        Target probabilities, should lie between 0 and 1.

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

    normalize : bool, default=False
        Normalize the cost by the number of samples.
        If ``True``, return the log average expected cost.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    log_expected_cost : float
        Log expected cost.

    Notes
    -----
    The expected log cost of each instance :math:`\\mathbb{E}[C^l_i]` is calculated as:

    .. math:: \\mathbb{E}[C^l_i] = y_i \\cdot (\\log(s_i) \\cdot C_i(1|1) + \\log(1 - s_i) \\cdot C_i(0|1)) + (1 - s_i) \\cdot (\\log(s_i) \\cdot C_i(1|0) + \\log(1 - s_i) \\cdot C_i(0|0))

    where

    - :math:`y_i` is the true label,
    - :math:`s_i` is the predicted probability,
    - :math:`C_i(1|1)` is the cost of a true positive ``tp_cost``,
    - :math:`C_i(0|1)` is the cost of a false positive ``fp_cost``,
    - :math:`C_i(1|0)` is the cost of a false negative ``fn_cost``, and
    - :math:`C_i(0|0)` is the cost of a true negative ``tn_cost``.

    When ``tp_cost`` and ``tn_cost`` equal -1, and `fp_cost`` and ``tn_cost`` equal 0,
    the expected log cost is equivalent to the log loss :func:`sklearn:sklearn.metrics.log_loss`.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.metrics import expected_log_cost_loss
        y_proba = [0.1, 0.9, 0.8, 0.2]
        y_true = [0, 1, 1, 0]
        fp_cost = np.array([4, 1, 2, 2])
        fn_cost = np.array([1, 3, 3, 1])
        expected_log_cost_loss(y_true, y_proba, fp_cost=fp_cost, fn_cost=fn_cost)

    """
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )
    cost = _compute_log_expected_cost(y_true, y_proba, tp_cost, tn_cost, fn_cost, fp_cost)
    if normalize:
        return cost.mean()
    return cost.sum()


def savings_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        tp_cost: Union[float, ArrayLike] = 0.0,
        fp_cost: Union[float, ArrayLike] = 0.0,
        tn_cost: Union[float, ArrayLike] = 0.0,
        fn_cost: Union[float, ArrayLike] = 0.0,
        check_input: bool = True,
) -> float:
    """
    Cost savings of a classifier compared to using a naive algorithm.

    The cost savings of a classifiers is the cost the classifier saved over a naive classification model
    (predicting all ones or zeros whichever is better).
    With 1 being the perfect model, 0 being not better than the naive model.

    Modified from `costcla.metrics.savings_score`.

    .. seealso::

        :func:`~empulse.metrics.expected_savings_score` : Expected savings of a classifier
        compared to using a naive algorithm.

        :func:`~empulse.metrics.cost_loss` : Cost of a classifier.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Predicted labels or calibrated probabilities.
        If the predictions are calibrated probabilities,
        the optimal decision threshold is calculated for each instance as [2]_:

        .. math:: t^*_i = \\frac{C_i(1|0) - C_i(0|0)}{C_i(1|0) - C_i(0|0) + C_i(0|1) - C_i(1|1)}

        .. note:: The optimal decision threshold is only accurate when the probabilities are well-calibrated.
                  See `scikit-learn's user guide <https://scikit-learn.org/stable/modules/calibration.html>`_
                  for more information.

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

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    Returns
    -------
    score : float
        Cost savings of a classifier compared to using a naive algorithm.

    Notes
    -----
    The cost of each instance :math:`C_i` is calculated as [1]_:

    .. math:: C_i(s_i) = y_i \\cdot (\\hat y_i \\cdot C_i(1|1) + (1 - \\hat y_i) \\cdot C_i(0|1)) + (1 - \\hat y_i) \\cdot (\\hat y_i \\cdot C_i(1|0) + (1 - \\hat y_i) \\cdot C_i(0|0))

    The savings over a naive model is calculated as:

    .. math::  \\text{Savings} = 1 - \\frac{\\sum_{i=1}^N C_i(s_i)}{\\min(\\sum_{i=1}^N C_i(0), \\sum_{i=1}^N C_i(1))}

    where

        - :math:`y_i` is the true label,
        - :math:`\\hat y_i` is the predicted label,
        - :math:`C_i(1|1)` is the cost of a true positive ``tp_cost``,
        - :math:`C_i(0|1)` is the cost of a false positive ``fp_cost``,
        - :math:`C_i(1|0)` is the cost of a false negative ``fn_cost``, and
        - :math:`C_i(0|0)` is the cost of a true negative ``tn_cost``.
        - :math:`N` is the number of samples.

    Code modified from `costcla.metrics.cost_loss`.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.metrics import savings_score
        y_pred = [0, 1, 0, 0]
        y_true = [0, 1, 1, 0]
        fp_cost = np.array([4, 1, 2, 2])
        fn_cost = np.array([1, 3, 3, 1])
        savings_score(y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost)

    References
    ----------
    .. [1] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           "Improving Credit Card Fraud Detection with Calibrated Probabilities",
           in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    .. [2] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """

    y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_pred, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    # Calculate the cost of naive prediction
    cost_base = min(
        cost_loss(y_true, np.zeros_like(y_true), tp_cost=tp_cost, fp_cost=fp_cost,
                  tn_cost=tn_cost, fn_cost=fn_cost, check_input=False),
        cost_loss(y_true, np.ones_like(y_true), tp_cost=tp_cost, fp_cost=fp_cost,
                  tn_cost=tn_cost, fn_cost=fn_cost, check_input=False)
    )

    cost = cost_loss(y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost,
                     tn_cost=tn_cost, fn_cost=fn_cost, check_input=False)
    return 1.0 - cost / cost_base


def expected_savings_score(
        y_true: ArrayLike,
        y_proba: ArrayLike,
        *,
        tp_cost: Union[float, ArrayLike] = 0.0,
        fp_cost: Union[float, ArrayLike] = 0.0,
        tn_cost: Union[float, ArrayLike] = 0.0,
        fn_cost: Union[float, ArrayLike] = 0.0,
        check_input: bool = True,
) -> float:
    """
    Expected savings of a classifier compared to using a naive algorithm.

    The expected cost savings of a classifiers is the expected cost the classifier saved
    over a naive classification model (predicting all ones or zeros whichever is better).
    With 1 being the perfect model, 0 being not better than the naive model.

    .. seealso::

        :func:`~empulse.metrics.savings_score` : Cost savings of a classifier compared to using a naive algorithm.

        :func:`~empulse.metrics.expected_cost_loss` : Expected cost of a classifier.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_proba : 1D array-like, shape=(n_samples,)
        Target probabilities, should lie between 0 and 1.

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
    score : float
        Expected savings of a classifier compared to using a naive algorithm.

        Notes
    -----
    The expected cost of each instance :math:`\\mathbb{E}[C_i]` is calculated as [1]_:

    .. math:: \\mathbb{E}[C_i(s_i)] = y_i \\cdot (s_i \\cdot C_i(1|1) + (1 - s_i) \\cdot C_i(0|1)) + (1 - s_i) \\cdot (s_i \\cdot C_i(1|0) + (1 - s_i) \\cdot C_i(0|0))

    The expected savings over a naive model is calculated as:

    .. math::  \\text{Expected Savings} = 1 - \\frac{\\sum_{i=1}^N \\mathbb{E}[C_i(s_i)]}{\\min(\\sum_{i=1}^N C_i(0), \\sum_{i=1}^N C_i(1))}

    where

        - :math:`y_i` is the true label,
        - :math:`\\hat y_i` is the predicted label,
        - :math:`C_i(1|1)` is the cost of a true positive ``tp_cost``,
        - :math:`C_i(0|1)` is the cost of a false positive ``fp_cost``,
        - :math:`C_i(1|0)` is the cost of a false negative ``fn_cost``, and
        - :math:`C_i(0|0)` is the cost of a true negative ``tn_cost``.
        - :math:`N` is the number of samples.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.metrics import expected_savings_score
        y_pred = [0.4, 0.8, 0.75, 0.1]
        y_true = [0, 1, 1, 0]
        fp_cost = np.array([4, 1, 2, 2])
        fn_cost = np.array([1, 3, 3, 1])
        expected_savings_score(y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost)

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    # Calculate the cost of naive prediction
    cost_base = min(
        cost_loss(y_true, np.zeros_like(y_true), tp_cost=tp_cost, fp_cost=fp_cost,
                  tn_cost=tn_cost, fn_cost=fn_cost, check_input=False),
        cost_loss(y_true, np.ones_like(y_true), tp_cost=tp_cost, fp_cost=fp_cost,
                  tn_cost=tn_cost, fn_cost=fn_cost, check_input=False)
    )

    cost = expected_cost_loss(y_true, y_proba, tp_cost=tp_cost, fp_cost=fp_cost,
                              tn_cost=tn_cost, fn_cost=fn_cost, check_input=False)
    return 1.0 - cost / cost_base


def make_objective_aec(
        model: Literal['xgboost', 'lightgbm', 'catboost', 'cslogit', 'csboost'],
        *,
        tp_cost: Union[ArrayLike, float] = 0.0,
        tn_cost: Union[ArrayLike, float] = 0.0,
        fn_cost: Union[ArrayLike, float] = 0.0,
        fp_cost: Union[ArrayLike, float] = 0.0,
) -> Callable[[np.ndarray, Union[xgb.DMatrix, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """
    Create an objective function for the Average Expected Cost (AEC) measure.

    The objective function presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    Parameters
    ----------
    model : {'xgboost', 'lightgbm', 'catboost', 'cslogit', 'csboost'}
        The model for which the objective function is created.

        - 'xgboost' : :class:`xgboost:xgboost.XGBClassifier`
        - 'lightgbm' : :class:`lightgbm:lightgbm.LGBMClassifier`  (not implemented yet)
        - 'catboost' : :class:`catboost:catboost.CatBoostClassifier`  (not implemented yet)
        - 'cslogit' : :class:`~empulse.models.CSLogitClassifier`
        - 'csboost' : :class:`~empulse.models.CSBoostClassifier`

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
        If array-like, then it is the cost of each false negative classification.

    Returns
    -------
    objective : Callable
        A custom objective function for the specified model.


    Examples
    --------
    .. code-block::  python

        import xgboost as xgb
        from empulse.metrics import make_objective_aec
        from sklearn.datasets import make_classification

        X, y = make_classification()
        objective = make_objective_aec('xgboost', fp_cost=1, fn_cost=1)

        clf = xgb.XGBClassifier(objective=objective, n_estimators=100, max_depth=3)
        clf.fit(X, y)
        y_pred = clf.predict(X)

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """

    if model == 'xgboost':
        objective = partial(_objective_xgboost, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
        update_wrapper(objective, _objective_xgboost)
    elif model == 'lightgbm':
        def objective(y_pred, train_data):
            return _objective_lightgbm(
                y_pred,
                train_data,
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost
            )
    elif model == 'catboost':
        objective = partial(_objective_catboost, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
        update_wrapper(objective, _objective_catboost)
    elif model == 'cslogit':
        objective = partial(_objective_cslogit, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
        update_wrapper(objective, _objective_cslogit)
    elif model == 'csboost':
        objective = partial(_objective_xgboost, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
        update_wrapper(objective, _objective_xgboost)
    else:
        raise ValueError(f"Expected model to be one of 'xgboost', 'lightgbm', 'catboost', 'cslogit', 'csboost', "
                         f"got {model} instead.")

    return objective


def _objective_cslogit(
        features: np.ndarray,
        weights: np.ndarray,
        y_true: np.ndarray,
        tp_cost: float = 0.0,
        tn_cost: float = 0.0,
        fn_cost: float = 0.0,
        fp_cost: float = 0.0,
) -> tuple[float, np.ndarray]:
    y_pred = expit(np.dot(weights, features.T))

    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=1)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=1)

    average_expected_cost = expected_cost_loss(
        y_true,
        y_pred,
        tp_cost=tp_cost,
        tn_cost=tn_cost,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        normalize=True,
        check_input=False
    )
    gradient = np.sum(
        features * y_pred * (1 - y_pred) * (
                y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)
        )
        , axis=0) / len(y_true)
    return average_expected_cost, gradient


def _objective_xgboost(
        y_pred: np.ndarray,
        dtrain: Union[xgb.DMatrix, np.ndarray],
        tp_cost: float = 0.0,
        tn_cost: float = 0.0,
        fn_cost: float = 0.0,
        fp_cost: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an objective function for the AEC measure.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    dtrain : xgb.DMatrix or np.ndarray
        Training data.

    Returns
    -------
    gradient  : np.ndarray
        Gradient of the objective function.

    hessian : np.ndarray
        Hessian of the objective function.
    """

    if isinstance(dtrain, np.ndarray):
        y_true = dtrain
    elif isinstance(dtrain, xgb.DMatrix):
        y_true = dtrain.get_label()
    else:
        raise TypeError(f"Expected dtrain to be of type np.ndarray or xgb.DMatrix, got {type(dtrain)} instead.")

    y_pred = expit(y_pred)
    cost = y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)
    gradient = y_pred * (1 - y_pred) * cost
    hessian = np.abs((1 - 2 * y_pred) * gradient)
    return gradient, hessian


def _objective_lightgbm(
        y_pred,
        train_data,
        tp_cost=0.0,
        tn_cost=0.0,
        fn_cost=0.0,
        fp_cost=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(train_data, 'get_label'):
        y_true = train_data.get_label()
    elif isinstance(train_data, np.ndarray):
        y_true = train_data
    else:
        raise TypeError(f"Invalid train data type, got {type(train_data)}.")

    y_pred = expit(y_pred)
    cost = y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)
    gradient = y_pred * (1 - y_pred) * cost
    hessian = np.abs((1 - 2 * y_pred) * gradient)
    return gradient, hessian


def _objective_catboost(
        y_pred,
        train_data,
        tp_cost=0.0,
        tn_cost=0.0,
        fn_cost=0.0,
        fp_cost=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("Objective function for CatBoost is not implemented yet.")
