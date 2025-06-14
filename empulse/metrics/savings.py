import numbers
from collections.abc import Callable, Sequence
from functools import partial, update_wrapper
from typing import Any, Literal, overload

import numpy as np
from scipy.special import expit

from .._types import FloatArrayLike, FloatNDArray
from ._validation import _check_consistent_length, _check_y_pred, _check_y_true


def _validate_input(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    tp_cost: float | FloatArrayLike,
    fp_cost: float | FloatArrayLike,
    tn_cost: float | FloatArrayLike,
    fn_cost: float | FloatArrayLike,
    check_input: bool,
) -> tuple[
    FloatNDArray,
    FloatNDArray,
    float | FloatNDArray,
    float | FloatNDArray,
    float | FloatNDArray,
    float | FloatNDArray,
]:
    if check_input:
        y_true = _check_y_true(y_true)
        y_proba = _check_y_pred(y_proba)
        arrays = [y_true, y_proba]
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
            raise ValueError('All costs are zero. At least one cost must be non-zero.')
        return y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost
    else:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        if not isinstance(tp_cost, numbers.Number):
            tp_cost = np.asarray(tp_cost)
        if not isinstance(fp_cost, numbers.Number):
            fp_cost = np.asarray(fp_cost)
        if not isinstance(tn_cost, numbers.Number):
            tn_cost = np.asarray(tn_cost)
        if not isinstance(fn_cost, numbers.Number):
            fn_cost = np.asarray(fn_cost)
        return y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost


def _compute_expected_cost(
    y_true: FloatNDArray,
    y_pred: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> FloatNDArray:
    return y_true * (y_pred * tp_cost + (1 - y_pred) * fn_cost) + (1 - y_true) * (
        y_pred * fp_cost + (1 - y_pred) * tn_cost
    )


def _compute_log_expected_cost(
    y_true: FloatNDArray,
    y_pred: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> FloatNDArray:
    epsilon: np.floating[Any] = np.finfo(y_pred.dtype).eps  # type: ignore[arg-type]
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    inverse_y_pred = 1 - y_pred
    log_y_pred = np.log(y_pred)
    log_inv_y_pred: FloatNDArray = np.log(inverse_y_pred)
    return y_true * (log_y_pred * tp_cost + log_inv_y_pred * fn_cost) + (1 - y_true) * (
        log_y_pred * fp_cost + log_inv_y_pred * tn_cost
    )


def cost_loss(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    *,
    tp_cost: float | FloatArrayLike = 0.0,
    fp_cost: float | FloatArrayLike = 0.0,
    tn_cost: float | FloatArrayLike = 0.0,
    fn_cost: float | FloatArrayLike = 0.0,
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

        :func:`~empulse.metrics.savings_score` : Cost savings of a classifier compared to using a baseline.

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

    .. math::

        C_i = y_i \\cdot (\\hat y_i \\cdot C_i(1|1) + (1 - \\hat y_i) \\cdot C_i(0|1)) + \
        (1 - \\hat y_i) \\cdot (\\hat y_i \\cdot C_i(1|0) + (1 - \\hat y_i) \\cdot C_i(0|0))

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
        return float(np.mean(cost))
    return float(np.sum(cost))


def expected_cost_loss(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    tp_cost: float | FloatArrayLike = 0.0,
    fp_cost: float | FloatArrayLike = 0.0,
    tn_cost: float | FloatArrayLike = 0.0,
    fn_cost: float | FloatArrayLike = 0.0,
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
        compared to using a baseline.

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

    .. math::

        \\mathbb{E}[C_i] = y_i \\cdot (s_i \\cdot C_i(1|1) + (1 - s_i) \\cdot C_i(0|1)) + \
        (1 - y_i) \\cdot (s_i \\cdot C_i(1|0) + (1 - s_i) \\cdot C_i(0|0))

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
    """  # noqa: D401
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    cost = _compute_expected_cost(y_true, y_proba, tp_cost, tn_cost, fn_cost, fp_cost)

    if normalize:
        return float(np.mean(cost))
    return float(np.sum(cost))


def expected_log_cost_loss(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    tp_cost: FloatArrayLike | float = 0.0,
    tn_cost: FloatArrayLike | float = 0.0,
    fn_cost: FloatArrayLike | float = 0.0,
    fp_cost: FloatArrayLike | float = 0.0,
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

    .. math::

        \\mathbb{E}[C^l_i] = y_i \\cdot (\\log(s_i) \\cdot C_i(1|1) + \\log(1 - s_i) \\cdot C_i(0|1)) + \
        (1 - y_i) \\cdot (\\log(s_i) \\cdot C_i(1|0) + \\log(1 - s_i) \\cdot C_i(0|0))

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

    """  # noqa: D401
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )
    cost = _compute_log_expected_cost(y_true, y_proba, tp_cost, tn_cost, fn_cost, fp_cost)
    if normalize:
        return float(np.mean(cost))
    return float(np.sum(cost))


def savings_score(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    *,
    baseline: FloatArrayLike | Literal['zero_one'] = 'zero_one',
    tp_cost: float | FloatArrayLike = 0.0,
    fp_cost: float | FloatArrayLike = 0.0,
    tn_cost: float | FloatArrayLike = 0.0,
    fn_cost: float | FloatArrayLike = 0.0,
    check_input: bool = True,
) -> float:
    """
    Cost savings of a classifier compared to using a baseline.

    The cost savings of a classifiers is the cost the classifier saved over a baseline classification model.
    By default, a naive algorithm is used (predicting all ones or zeros whichever is better).
    With 1 being the perfect model, 0 being as good as the baseline model,
    and values smaller than 0 being worse than the baseline model.

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

    baseline : 'zero_one' or 1D array-like, shape=(n_samples,), default='zero_one'
        Predicted labels or calibrated probabilities of the baseline model.
        If ``'zero_one'``, the baseline model is a naive model that predicts all zeros or all ones
        depending on which is better.

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
        Cost savings of a classifier compared to using a baseline.

    Notes
    -----
    The cost of each instance :math:`C_i` is calculated as [1]_:

    .. math::

        C_i(s_i) = \
        y_i \\cdot (\\hat y_i \\cdot C_i(1|1) + (1 - \\hat y_i) \\cdot C_i(0|1)) + \
        (1 - \\hat y_i) \\cdot (\\hat y_i \\cdot C_i(1|0) + (1 - \\hat y_i) \\cdot C_i(0|0))

    The savings over a naive model is calculated as:

    .. math::  \\text{Savings} = 1 - \\frac{\\sum_{i=1}^N C_i(s_i)}{\\min(\\sum_{i=1}^N C_i(0), \\sum_{i=1}^N C_i(1))}

    The savings over a baseline model is calculated as:

    .. math::  \\text{Savings} = 1 - \\frac{\\sum_{i=1}^N C_i(s_i)}{\\sum_{i=1}^N C_i(s_i^*)}

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

    if baseline == 'zero_one':
        # Calculate the cost of naive prediction
        cost_base = min(
            cost_loss(
                y_true,
                np.zeros_like(y_true),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
            cost_loss(
                y_true,
                np.ones_like(y_true),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
        )
    else:
        baseline = np.asarray(baseline)
        cost_base = cost_loss(
            y_true,
            baseline,
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )

    cost = cost_loss(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=False
    )
    if cost_base == 0.0:
        cost_base = float(np.finfo(float).eps)
    return 1.0 - cost / cost_base


def expected_savings_score(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    baseline: Literal['zero_one', 'prior'] | FloatArrayLike = 'zero_one',
    tp_cost: float | FloatArrayLike = 0.0,
    fp_cost: float | FloatArrayLike = 0.0,
    tn_cost: float | FloatArrayLike = 0.0,
    fn_cost: float | FloatArrayLike = 0.0,
    check_input: bool = True,
) -> float:
    """
    Expected savings of a classifier compared to a baseline.

    The expected cost savings of a classifiers is the expected cost the classifier saved
    over a baseline classification model.
    By default, a naive model is used (predicting all ones or zeros whichever is better).
    With 1 being the perfect model, 0 being as good as the baseline model,
    and values smaller than 0 being worse than the baseline model.

    .. seealso::

        :func:`~empulse.metrics.savings_score` : Cost savings of a classifier compared to a baseline.

        :func:`~empulse.metrics.expected_cost_loss` : Expected cost of a classifier.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_proba : 1D array-like, shape=(n_samples,)
        Target probabilities, should lie between 0 and 1.

    baseline : {'zero_one', 'prior'} or 1D array-like, shape=(n_samples,), default='zero_one'
        
        - If ``'zero_one'``, the baseline model is a naive model that predicts all zeros or all ones
          depending on which is better.
        - If ``'prior'``, the baseline model is a model that predicts the prior probability of 
          the majority or minority class depending on which is better.
        - If array-like, target probabilities of the baseline model.
        

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
        Expected savings of a classifier compared to a baseline.

    Notes
    -----
    The expected cost of each instance :math:`\\mathbb{E}[C_i]` is calculated as [1]_:

    .. math::

        \\mathbb{E}[C_i(s_i)] = \
        y_i \\cdot (s_i \\cdot C_i(1|1) + (1 - s_i) \\cdot C_i(0|1)) + (1 - s_i) \\cdot \
        (s_i \\cdot C_i(1|0) + (1 - s_i) \\cdot C_i(0|0))

    The expected savings over a naive model is calculated as:

    .. math::

        \\text{Expected Savings} = \
        1 - \\frac{\\sum_{i=1}^N \\mathbb{E}[C_i(s_i)]}{\\min(\\sum_{i=1}^N C_i(0), \\sum_{i=1}^N C_i(1))}

    The expected savings over a baseline model is calculated as:

    .. math::

        \\text{Expected Savings} = \
        1 - \\frac{\\sum_{i=1}^N \\mathbb{E}[C_i(s_i)]}{\\sum_{i=1}^N \\mathbb{E}[C_i(s_i^*)]}

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
    """  # noqa: D401
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    if baseline == 'zero_one':
        # Calculate the cost of naive prediction
        cost_base = min(
            cost_loss(
                y_true,
                np.zeros_like(y_true),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
            cost_loss(
                y_true,
                np.ones_like(y_true),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
        )
    elif baseline == 'prior':
        prior_pos = np.mean(y_true)
        prior_neg = 1 - prior_pos
        cost_base = min(
            cost_loss(
                y_true,
                np.full_like(y_true, prior_pos),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
            cost_loss(
                y_true,
                np.full_like(y_true, prior_neg),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
        )
    else:
        baseline = np.asarray(baseline)
        cost_base = expected_cost_loss(
            y_true,
            baseline,
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )

    # avoid division by zero
    if cost_base == 0.0:
        cost_base = float(np.finfo(float).eps)

    cost = expected_cost_loss(
        y_true, y_proba, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=False
    )
    return 1.0 - cost / cost_base


@overload
def make_objective_aec(
    model: Literal['catboost'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> tuple['AECObjective', 'AECMetric']: ...


@overload
def make_objective_aec(
    model: Literal['xgboost', 'lightgbm'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]: ...


@overload
def make_objective_aec(
    model: Literal['cslogit'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> Callable[[FloatNDArray, FloatNDArray, FloatNDArray], tuple[float, FloatNDArray]]: ...


def make_objective_aec(
    model: Literal['xgboost', 'lightgbm', 'catboost', 'cslogit'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> (
    Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]
    | Callable[[FloatNDArray, FloatNDArray, FloatNDArray], tuple[float, FloatNDArray]]
    | tuple['AECObjective', 'AECMetric']
):
    """
    Create an objective function for the Average Expected Cost (AEC) measure.

    The objective function presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    Read more in the :ref:`User Guide <cost_functions>`.

    Parameters
    ----------
    model : {'xgboost', 'lightgbm', 'catboost', 'cslogit'}
        The model for which the objective function is created.

        - 'xgboost' : :class:`xgboost:xgboost.XGBClassifier`
        - 'lightgbm' : :class:`lightgbm:lightgbm.LGBMClassifier`
        - 'catboost' : :class:`catboost.CatBoostClassifier`
        - 'cslogit' : :class:`~empulse.models.CSLogitClassifier`

    tp_cost : float or np.array, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.

    fp_cost : float or np.array, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.

    tn_cost : float or np.array, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.

    fn_cost : float or np.array, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.

    Returns
    -------
    objective : Callable
        A custom objective function for the specified model.


    Examples
    --------

    .. code-block::  python

        from xgboost import XGBClassifier
        from empulse.metrics import make_objective_aec

        objective = make_objective_aec('xgboost', fp_cost=1, fn_cost=1)
        clf = XGBClassifier(objective=objective, n_estimators=100, max_depth=3)

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """
    if model == 'xgboost':
        objective: Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]] = partial(
            _objective_boost, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
        )
        update_wrapper(objective, _objective_boost)
    elif model == 'lightgbm':

        def objective(y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
            """
            Create an objective function for the AEC measure.

            Parameters
            ----------
            y_true : np.ndarray
                Ground truth labels
            y_score : np.ndarray
                Predicted labels

            Returns
            -------
            gradient  : np.ndarray
                Gradient of the objective function.

            hessian : np.ndarray
                Hessian of the objective function.
            """
            return _objective_boost(y_true, y_score, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)

    elif model == 'catboost':
        return (
            AECObjective(tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost),
            AECMetric(tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost),
        )
    elif model == 'cslogit':
        objective = partial(_objective_cslogit, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)  # type: ignore[assignment]
        update_wrapper(objective, _objective_cslogit)
    else:
        raise ValueError(
            f"Expected model to be one of 'xgboost', 'lightgbm', 'catboost' or 'cslogit', got {model} instead."
        )

    return objective


def _objective_cslogit(
    features: FloatNDArray,
    weights: FloatNDArray,
    y_true: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> tuple[float, FloatNDArray]:
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
        check_input=False,
    )
    gradient = np.mean(
        features * y_pred * (1 - y_pred) * (y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)), axis=0
    )
    return average_expected_cost, gradient


def _objective_boost(
    y_true: FloatNDArray,
    y_score: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> tuple[FloatNDArray, FloatNDArray]:
    """
    Create an objective function for the AEC measure.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0 or 1).
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    gradient  : np.ndarray
        Gradient of the objective function.

    hessian : np.ndarray
        Hessian of the objective function.
    """
    y_proba = expit(y_score)
    cost = y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)
    gradient = y_proba * (1 - y_proba) * cost
    hessian = np.abs((1 - 2 * y_proba) * gradient)
    return gradient, hessian


class AECObjective:
    """AEC objective for catboost."""

    def __init__(
        self,
        tp_cost: FloatNDArray | float = 0.0,
        tn_cost: FloatNDArray | float = 0.0,
        fn_cost: FloatNDArray | float = 0.0,
        fp_cost: FloatNDArray | float = 0.0,
    ):
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def calc_ders_range(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: FloatNDArray
    ) -> list[tuple[float, float]]:
        """
        Compute first and second derivative of the loss function with respect to the predicted value for each object.

        Parameters
        ----------
        predictions : indexed container of floats
            Current predictions for each object.

        targets : indexed container of floats
            Target values you provided with the dataset.

        weights : float, optional (default=None)
            Instance weight.

        Returns
        -------
            der1 : list-like object of float
            der2 : list-like object of float

        """
        weights = weights.astype(int)
        # Use weights as a proxy to index the costs
        tp_cost = self.tp_cost[weights] if isinstance(self.tp_cost, np.ndarray) else self.tp_cost
        tn_cost = self.tn_cost[weights] if isinstance(self.tn_cost, np.ndarray) else self.tn_cost
        fn_cost = self.fn_cost[weights] if isinstance(self.fn_cost, np.ndarray) else self.fn_cost
        fp_cost = self.fp_cost[weights] if isinstance(self.fp_cost, np.ndarray) else self.fp_cost

        y_proba = expit(predictions)
        cost = targets * (tp_cost - fn_cost) + (1 - targets) * (fp_cost - tn_cost)
        gradient = y_proba * (1 - y_proba) * cost
        hessian = np.abs((1 - 2 * y_proba) * gradient)
        # convert from two arrays to one list of tuples
        return list(zip(-gradient, -hessian, strict=False))


class AECMetric:
    """AEC metric for catboost."""

    def __init__(
        self,
        tp_cost: FloatNDArray | float = 0.0,
        tn_cost: FloatNDArray | float = 0.0,
        fn_cost: FloatNDArray | float = 0.0,
        fp_cost: FloatNDArray | float = 0.0,
    ) -> None:
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def is_max_optimal(self) -> bool:
        """Return whether great values of metric are better."""
        return False

    def evaluate(
        self, predictions: Sequence[float], targets: Sequence[float], weights: FloatNDArray
    ) -> tuple[float, float]:
        """
        Evaluate metric value.

        Parameters
        ----------
        approxes : list of indexed containers (containers with only __len__ and __getitem__ defined) of float
            Vectors of approx labels.

        targets : one dimensional indexed container of float
            Vectors of true labels.

        weights : one dimensional indexed container of float, optional (default=None)
            Weight for each instance.

        Returns
        -------
            weighted error : float
            total weight : float

        """
        weights = weights.astype(int)
        # Use weights as a proxy to index the costs
        tp_cost = self.tp_cost[weights] if isinstance(self.tp_cost, np.ndarray) else self.tp_cost
        tn_cost = self.tn_cost[weights] if isinstance(self.tn_cost, np.ndarray) else self.tn_cost
        fn_cost = self.fn_cost[weights] if isinstance(self.fn_cost, np.ndarray) else self.fn_cost
        fp_cost = self.fp_cost[weights] if isinstance(self.fp_cost, np.ndarray) else self.fp_cost

        y_proba = expit(predictions)
        return expected_cost_loss(
            targets,
            y_proba,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            normalize=True,
            check_input=False,
        ), 1

    def get_final_error(self, error: float, weight: float) -> float:
        """
        Return final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        metric value : float

        """
        return error
