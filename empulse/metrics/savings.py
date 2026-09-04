import numbers
from typing import Literal

import numpy as np

from .._types import FloatArrayLike, FloatNDArray
from ._validation import _check_consistent_length, _check_y_pred, _check_y_true
from .metric.prebuilt_metrics import make_generic_cost_metric, make_generic_log_cost_metric, make_generic_savings_metric


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
        - :math:`C_i(1|0)` is the cost of a false positive ``fp_cost``,
        - :math:`C_i(0|1)` is the cost of a false negative ``fn_cost``, and
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


expected_cost_loss = make_generic_cost_metric()
#: Expected cost of a classifier.
#:
#: A generic :class:`~empulse.metrics.Metric` built from the :class:`~empulse.metrics.Cost`
#: strategy on a plain cost matrix, accepting class- or instance-dependent ``tp_cost``,
#: ``tn_cost``, ``fp_cost``, and ``fn_cost`` parameters. Call as
#: ``expected_cost_loss(y_true, y_proba, tp_cost=..., tn_cost=..., fp_cost=..., fn_cost=...)``.
#:
#: .. seealso::
#:
#:     :func:`~empulse.metrics.cost_loss` : Cost of a classifier using hard (thresholded) labels.
#:
#:     :func:`~empulse.metrics.expected_savings_score` : Expected savings of a classifier
#:     compared to using a baseline.
#:
#: .. note::
#:    This replaces the previous native ``expected_cost_loss`` function, which by default
#:    returned the *summed* cost (with an optional ``normalize=True`` argument to switch to the
#:    mean). This metric always returns the *mean* cost per instance.
#:
#: Examples
#: --------
#: .. code-block:: python
#:
#:     import numpy as np
#:     from empulse.metrics import expected_cost_loss
#:
#:     y_proba = [0.2, 0.9, 0.1, 0.2]
#:     y_true = [0, 1, 1, 0]
#:     fp_cost = np.array([4, 1, 2, 2])
#:     fn_cost = np.array([1, 3, 3, 1])
#:     expected_cost_loss(y_true, y_proba, fp_cost=fp_cost, fn_cost=fn_cost)


expected_log_cost_loss = make_generic_log_cost_metric()
#: Expected log cost of a classifier.
#:
#: A generic :class:`~empulse.metrics.Metric` built from the :class:`~empulse.metrics.LogCost`
#: strategy on a plain cost matrix, accepting class- or instance-dependent ``tp_cost``,
#: ``tn_cost``, ``fp_cost``, and ``fn_cost`` parameters. Call as
#: ``expected_log_cost_loss(y_true, y_proba, tp_cost=..., tn_cost=..., fp_cost=..., fn_cost=...)``.
#: When ``tp_cost`` and ``tn_cost`` equal -1, and ``fp_cost`` and ``fn_cost`` equal 0, the expected
#: log cost is equivalent to the log loss :func:`sklearn:sklearn.metrics.log_loss`.
#:
#: .. seealso::
#:
#:     :func:`~empulse.metrics.expected_cost_loss` : Expected cost of a classifier.
#:
#: .. note::
#:    This replaces the previous native ``expected_log_cost_loss`` function, which by default
#:    returned the *summed* log cost (with an optional ``normalize=True`` argument to switch to
#:    the mean). This metric always returns the *mean* log cost per instance.
#:
#: Examples
#: --------
#: .. code-block:: python
#:
#:     import numpy as np
#:     from empulse.metrics import expected_log_cost_loss
#:
#:     y_proba = [0.1, 0.9, 0.8, 0.2]
#:     y_true = [0, 1, 1, 0]
#:     fp_cost = np.array([4, 1, 2, 2])
#:     fn_cost = np.array([1, 3, 3, 1])
#:     expected_log_cost_loss(y_true, y_proba, fp_cost=fp_cost, fn_cost=fn_cost)


def savings_score(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    *,
    baseline: FloatArrayLike | Literal['zero_one', 'one', 'zero'] = 'zero_one',
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

        - If ``'zero_one'``, the baseline model is a naive model that predicts all zeros or all ones
          depending on which is better.
        - If ``'one'``, the baseline model is a model that predicts all ones.
        - If ``'zero'``, the baseline model is a model that predicts all zeros.

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
        - :math:`C_i(1|0)` is the cost of a false positive ``fp_cost``,
        - :math:`C_i(0|1)` is the cost of a false negative ``fn_cost``, and
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

    if not isinstance(baseline, str):
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
    elif baseline == 'zero_one':
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
    elif baseline == 'one':
        cost_base = cost_loss(
            y_true,
            np.ones_like(y_true),
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )
    elif baseline == 'zero':
        cost_base = cost_loss(
            y_true,
            np.zeros_like(y_true),
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )
    else:
        raise ValueError("Invalid baseline. Must be 'zero_one', 'zero', 'one', or an array-like.")

    cost = cost_loss(
        y_true, y_pred, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=False
    )
    if cost_base == 0.0:
        cost_base = float(np.finfo(float).eps)
    return 1.0 - cost / cost_base


expected_savings_score = make_generic_savings_metric()
#: Expected savings of a classifier compared to a baseline.
#:
#: A generic :class:`~empulse.metrics.Metric` built from the :class:`~empulse.metrics.Savings`
#: strategy on a plain cost matrix, accepting class- or instance-dependent ``tp_cost``,
#: ``tn_cost``, ``fp_cost``, and ``fn_cost`` parameters, plus a ``baseline`` argument. Call as
#: ``expected_savings_score(y_true, y_proba, tp_cost=..., tn_cost=..., fp_cost=..., fn_cost=...,
#: baseline='zero_one')``.
#:
#: ``baseline`` accepts:
#:
#: - ``'zero_one'`` (default): a naive model that predicts all zeros or all ones, whichever is
#:   better.
#: - ``'one'``: a model that predicts all ones.
#: - ``'zero'``: a model that predicts all zeros.
#: - ``'prior'``: a model that predicts the prior probability of the majority or minority class,
#:   whichever is better.
#: - array-like: target probabilities of a baseline model.
#:
#: With 1 being the perfect model, 0 being as good as the baseline model, and values smaller than
#: 0 being worse than the baseline model.
#:
#: .. seealso::
#:
#:     :func:`~empulse.metrics.savings_score` : Cost savings of a classifier compared to a
#:     baseline, using hard (thresholded) labels.
#:
#:     :func:`~empulse.metrics.expected_cost_loss` : Expected cost of a classifier.
#:
#: Examples
#: --------
#: .. code-block:: python
#:
#:     import numpy as np
#:     from empulse.metrics import expected_savings_score
#:
#:     y_pred = [0.4, 0.8, 0.75, 0.1]
#:     y_true = [0, 1, 1, 0]
#:     fp_cost = np.array([4, 1, 2, 2])
#:     fn_cost = np.array([1, 3, 3, 1])
#:     expected_savings_score(y_true, y_pred, fp_cost=fp_cost, fn_cost=fn_cost)
