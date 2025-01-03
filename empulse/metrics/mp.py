import numpy as np
from numpy.typing import ArrayLike

from ._convex_hull import _compute_convex_hull


def max_profit_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        tp_benefit: float = 0.0,
        tn_benefit: float = 0.0,
        fn_cost: float = 0.0,
        fp_cost: float = 0.0,
) -> float:
    """
    :func:`~empulse.metrics.max_profit()` but only returning the MP score.

    .. seealso::
        :func:`~empulse.metrics.max_profit` : To also return the threshold at which the maximum profit is achieved.

        :func:`~empulse.metrics.emp` : For a stochastic version of the maximum profit.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    tp_benefit : float, default=0.0
        Benefit attributed to true positive predictions.

    tn_benefit : float, default=0.0
        Benefit attributed to true negative predictions.

    fn_cost : float, default=0.0
        Cost attributed to false negative predictions.

    fp_cost : float, default=0.0
        Cost attributed to false positive predictions.

    Returns
    -------
    mp : float
        Maximum Profit.

    Notes
    -----
    The MP is defined as [1]_:

    .. math::  P(T;b_0, c_0, b_1, c_1)

    The MP requires that the positive class is encoded as 0, and negative class as 1.
    However, this implementation assumes the standard notation ('positive': 1, 'negative': 0).

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289

    Examples
    --------

    Reimplement MPC:

    >>> from empulse.metrics import max_profit_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>>
    >>> clv = 200
    >>> d = 10
    >>> f = 1
    >>> gamma = 0.3
    >>> tp_benefit = clv * (gamma * (1 - (d / clv) - (f / clv)))
    >>> fp_cost = d + f
    >>>
    >>> max_profit_score(y_true, y_score, tp_benefit=tp_benefit, fp_cost=fp_cost)
    24.22...
    """
    return max_profit(y_true, y_score, tp_benefit=tp_benefit, tn_benefit=tn_benefit, fn_cost=fn_cost, fp_cost=fp_cost)[0]


def max_profit(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        tp_benefit: float = 0.0,
        tn_benefit: float = 0.0,
        fn_cost: float = 0.0,
        fp_cost: float = 0.0,
) -> tuple[float, float]:
    """
    Maximum Profit Measure (MP).

    .. seealso::
        :func:`~empulse.metrics.max_profit_score` : To only return the maximum profit score.

        :func:`~empulse.metrics.emp` : For a stochastic version of the maximum profit.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    tp_benefit : float, default=0.0
        Benefit attributed to true positive predictions.

    tn_benefit : float, default=0.0
        Benefit attributed to true negative predictions.

    fn_cost : float, default=0.0
        Cost attributed to false negative predictions.

    fp_cost : float, default=0.0
        Cost attributed to false positive predictions.

    Returns
    -------
    mp : float
        Maximum Profit

    threshold : float
        Threshold at which the maximum profit is achieved

    Notes
    -----
    The MP is defined as [1]_:

    .. math::  P(T;b_0, c_0, b_1, c_1)

    The MP requires that the positive class is encoded as 0, and negative class as 1.
    However, this implementation assumes the standard notation ('positive': 1, 'negative': 0).

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289

    Examples
    --------

    Reimplement MPC:

    >>> from empulse.metrics import max_profit
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>>
    >>> clv = 200
    >>> d = 10
    >>> f = 1
    >>> gamma = 0.3
    >>> tp_benefit = clv * (gamma * (1 - (d / clv)) - (f / clv))
    >>> fp_cost = d + f
    >>>
    >>> max_profit(y_true, y_score, tp_benefit=tp_benefit, fp_cost=fp_cost)
    (23.87..., 0.875)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0

    f0, f1 = _compute_convex_hull(y_true, y_score)

    profits = (tp_benefit + fn_cost) * pi0 * f0 - \
              (tn_benefit + fp_cost) * pi1 * f1 + \
              tn_benefit * pi1 - fn_cost * pi0
    best_index = np.argmax(profits)
    maximum_profit = float(profits[best_index])
    customer_threshold = float(f0[best_index] * pi0 + f1[best_index] * pi1)
    return maximum_profit, customer_threshold
