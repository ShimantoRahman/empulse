from functools import lru_cache
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_mp
from ..common import _compute_profits


def mpc_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        accept_rate: float = 0.3,
        clv: float = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
) -> float:
    """
    Convenience function around `mpc()` only returning MPC score

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    accept_rate : float, default=0.3
        Probability that a churner accepts the retention offer (gamma > 0).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If clv is a float: constant customer lifetime value per retained customer (clv > d).
        If clv is an array: invidivualized customer lifetime value of each customer when retained (mean(clv) > d).

    incentive_cost : float, default=10
        Constant cost of the retention offer offered to potention churners (d > 0).

    contact_cost : float, default=1
        Constant cost of contacting a customer (f > 0).

    Returns
    -------
    mpc : float
        Maximum Profit Measure for Customer Churn.

    Notes
    -----
    The MPC is defined as [1]_:

    .. math::  CLV (\gamma (1 - \delta) - \phi) \pi_0 F_0(T) - CLV (\delta + \phi) \pi_1 F_1(T)

    The MPC requires that the churn class is encoded as 0, and it is NOT interchangeable (see [3]_ p37).
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    An equivalent R implementation is available in [2]_.

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289
    .. [2] Bravo, C. and Vanden Broucke, S. and Verbraken, T. (2019).
        EMP: Expected Maximum Profit Classification Performance Measure.
        R package version 2.0.5. Available Online:
        http://cran.r-project.org/web/packages/EMP/index.html
    .. [3] Verbraken, T. (2013). Business-Oriented Data Analytics:
        Theory and Case Studies. Ph.D. dissertation, Dept. LIRIS, KU Leuven,
        Leuven, Belgium, 2013.

    Examples
    --------
    >>> from empulse.metrics import mpc_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpc_score(y_true, y_pred)
    23.874999999999996

    Using scorer:

    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_score, StratifiedKFold
    >>> from sklearn.metrics import make_scorer
    >>> from empulse.metrics import mpa_score
    >>>
    >>> X, y = make_classification(random_state=42)
    >>> model = LogisticRegression()
    >>> cv = StratifiedKFold(n_splits=5, random_state=42)
    >>> scorer = make_scorer(
    >>>     mpc_score,
    >>>     needs_proba=True,
    >>>     clv=300,
    >>>     incentive_cost=15,
    >>> )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    42.08999999999999
    """
    return \
    mpc(y_true, y_pred, clv=clv, incentive_cost=incentive_cost, contact_cost=contact_cost, accept_rate=accept_rate)[0]


def mpc(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        accept_rate: float = 0.3,
        clv: float = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
) -> tuple[float, float]:
    """
    Maximum Profit Measure for Customer Churn (MPC)
    ===============================================

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    accept_rate : float, default=0.3
        Probability that a churner accepts the retention offer (gamma > 0).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If clv is a float: constant customer lifetime value per retained customer (clv > d).
        If clv is an array: invidivualized customer lifetime value of each customer when retained (mean(clv) > d).

    incentive_cost : float, default=10
        Constant cost of retention offer offered to potential churners (d > 0).

    contact_cost : float, default=1
        Constant cost of contacting a customer (f > 0).

    Returns
    -------
    (empc, threshold) : tuple[float, float]
        Maximum Profit Measure for Customer Churn and
        the threshold η at which the maximum profit is achieved.

    Notes
    -----
    The MPC is defined as [1]_:

    .. math::  CLV (\gamma (1 - \delta) - \phi) \pi_0 F_0(T) - CLV (\delta + \phi) \pi_1 F_1(T)

    The MPC requires that the churn class is encoded as 0, and it is NOT interchangeable (see [3]_ p37).
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    An equivalent R implementation is available in [2]_.

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289
    .. [2] Bravo, C. and Vanden Broucke, S. and Verbraken, T. (2019).
        EMP: Expected Maximum Profit Classification Performance Measure.
        R package version 2.0.5. Available Online:
        http://cran.r-project.org/web/packages/EMP/index.html
    .. [3] Verbraken, T. (2013). Business-Oriented Data Analytics:
        Theory and Case Studies. Ph.D. dissertation, Dept. LIRIS, KU Leuven,
        Leuven, Belgium, 2013.

    Examples
    --------
    >>> from empulse.metrics import mpc
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpc(y_true, y_pred)
    (23.874999999999996, 0.875)
    """
    profits, customer_thresholds = compute_profit_churn(y_true, y_pred, clv, incentive_cost, contact_cost, accept_rate)
    max_profit_index = np.argmax(profits)

    return profits[max_profit_index], customer_thresholds[max_profit_index]


def compute_profit_churn(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        clv: float = 200,
        d: float = 10,
        f: float = 1,
        gamma: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred, clv = _validate_input_mp(y_true, y_pred, gamma, clv, d, f)
    cost_benefits = _compute_cost_benefits(gamma, clv, d, f)
    return _compute_profits(y_true, y_pred, cost_benefits)


@lru_cache(maxsize=1)
def _compute_cost_benefits(gamma: float, clv: float, d: float, f: float) -> np.ndarray:
    delta = d / clv
    phi = f / clv

    true_positive_benefit = clv * (gamma * (1 - delta) - phi)
    false_positive_cost = -1 * clv * (delta + phi)
    return np.array([true_positive_benefit, false_positive_cost])