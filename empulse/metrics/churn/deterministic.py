from functools import lru_cache
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_mp
from ..common import _compute_profits


def mpc_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        accept_rate: float = 0.3,
        clv: float = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
        check_input: bool = True,
) -> float:
    """
    :func:`~empulse.metrics.mpc()` but only returning the MPC score.

    MPC presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer.
    For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.mpc_score` : to also return the fraction of the customer base
        that should be targeted to maximize profit.

        :func:`~empulse.metrics.empc_score` : for a stochastic version of this metric.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    accept_rate : float, default=0.3
        Probability of a customer accepting the retention offer (``0 < accept_rate < 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: average customer lifetime value of retained customers (``clv > incentive_cost``).
        If ``array``: customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

        .. note::
            Passing a CLV array is equivalent to passing a float with the average CLV of that array.

    incentive_cost : float, default=10
        Cost of incentive offered to a customer (``incentive_cost > 0``).

    contact_cost : float, default=1
        Cost of contacting a customer (``contact_cost > 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    mpc : float
        Maximum Profit Measure for Customer Churn.

    Notes
    -----
    The MPC is defined as [1]_:

    .. math::  CLV (\\gamma (1 - \\delta) - \\phi) \\pi_0 F_0(T) - CLV (\\delta + \\phi) \\pi_1 F_1(T)

    The MPC requires that the churn class is encoded as 0, and it is NOT interchangeable (see [2]_ p37).
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    An equivalent R implementation is available in [3]_.

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
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpc_score(y_true, y_score)
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
    >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> scorer = make_scorer(
    ...     mpc_score,
    ...     response_method='predict_proba',
    ...     clv=300,
    ...     incentive_cost=15,
    ... )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    42.08999999999999
    """
    return mpc(
        y_true,
        y_score,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
        check_input=check_input,
    )[0]


def mpc(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        accept_rate: float = 0.3,
        clv: Union[ArrayLike, float] = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
        check_input: bool = True,
) -> tuple[float, float]:
    """
    Maximum Profit Measure for Customer Churn (MPC).

    MPC presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer.
    For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.mpc_score` : to only return the MPC score.

        :func:`~empulse.metrics.empc` : for a stochastic version of this metric.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    accept_rate : float, default=0.3
        Probability of a customer accepting the retention offer (``0 < accept_rate < 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: average customer lifetime value of retained customers (``clv > incentive_cost``).
        If ``array``: customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

        .. note::
            Passing a CLV array is equivalent to passing a float with the average CLV of that array.

    incentive_cost : float, default=10
        Cost of incentive offered to a customer (``incentive_cost > 0``).

    contact_cost : float, default=1
        Cost of contacting a customer (``contact_cost > 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empc : float
        Maximum Profit Measure for Customer Churn

    threshold : float
        Fraction of the customer base that should be targeted to maximize profit

    Notes
    -----
    The MPC is defined as [1]_:

    .. math::  CLV (\\gamma (1 - \\delta) - \\phi) \\pi_0 F_0(T) - CLV (\\delta + \\phi) \\pi_1 F_1(T)

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
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpc(y_true, y_score)
    (23.874999999999996, 0.875)
    """
    profits, customer_thresholds = compute_profit_churn(
        y_true,
        y_score,
        clv,
        incentive_cost,
        contact_cost,
        accept_rate,
        check_input,
    )
    max_profit_index = np.argmax(profits)

    return profits[max_profit_index], customer_thresholds[max_profit_index]


def compute_profit_churn(
        y_true: ArrayLike,
        y_score: ArrayLike,
        clv: Union[ArrayLike, float] = 200,
        d: float = 10,
        f: float = 1,
        gamma: float = 0.3,
        check_input: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if check_input:
        y_true, y_score, clv = _validate_input_mp(y_true, y_score, gamma, clv, d, f)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)
    if isinstance(clv, np.ndarray):
        clv = np.mean(clv)
    cost_benefits = _compute_cost_benefits(gamma, clv, d, f)
    return _compute_profits(y_true, y_score, cost_benefits)


@lru_cache(maxsize=1)
def _compute_cost_benefits(gamma: float, clv: float, d: float, f: float) -> np.ndarray:
    delta = d / clv
    phi = f / clv

    true_positive_benefit = clv * (gamma * (1 - delta) - phi)
    false_positive_cost = -1 * clv * (delta + phi)
    return np.array([true_positive_benefit, false_positive_cost])
