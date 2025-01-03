from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_deterministic
from ..common import _compute_profits


def mpa_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        contribution: float = 8_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
        check_input: bool = True
) -> float:
    """
    :func:`~empulse.metrics.mpa()` but only returning the MPA score.

    MPA presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    .. seealso::

        :func:`~empulse.metrics.mpa` : to also return the fraction of the leads
        that should be targeted to maximize profit.

        :func:`~empulse.metrics.empa_score` : for a stochastic version of this metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    contribution : float, default=7000
        Average contribution of a new customer (``contribution ≥ 0``).

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (``sales_cost ≥ 0``).

    contact_cost : float, default=50
        Average contact cost of targeted leads (``contact_cost ≥ 0``).

    direct_selling : float, default=1
        Fraction of leads sold to directly (``0 ≤ direct_selling ≤ 1``).
        ``direct_selling = 0`` for indirect channel.
        ``direct_selling = 1`` for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermediaries (``0 ≤ commission ≤ 1``).

        .. note::
            The commission is only relevant when there is an indirect channel (``direct_selling < 1``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empa : float
        Expected Maximum Profit measure for customer Acquisition.

    Examples
    --------
    Direct channel:

    >>> from empulse.metrics import mpa_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpa_score(y_true, y_score, direct_selling=1)
    3706.25

    Indirect channel using scorer:

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
    ...     mpa_score,
    ...     response_method='predict_proba',
    ...     contribution=7_000,
    ...     sales_cost=2_000,
    ...     contact_cost=100,
    ...     direct_selling=0,
    ... )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    3099.0
    """
    return mpa(
        y_true,
        y_score,
        contribution=contribution,
        contact_cost=contact_cost,
        sales_cost=sales_cost,
        direct_selling=direct_selling,
        commission=commission,
        check_input=check_input,
    )[0]


def mpa(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        contribution: float = 8_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
        check_input: bool = True
) -> tuple[float, float]:
    """
    Maximum Profit measure for customer Acquisition (MPA).

    MPA presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    .. seealso::

        :func:`~empulse.metrics.mpa_score` : to only return the EMPA score.

        :func:`~empulse.metrics.empa` : for a stochastic version of this metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    contribution : float, default=7000
        Average contribution of a new customer (``contribution ≥ 0``).

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (``sales_cost ≥ 0``).

    contact_cost : float, default=50
        Average contact cost of targeted leads (``contact_cost ≥ 0``).

    direct_selling : float, default=1
        Fraction of leads sold to directly (``0 ≤ direct_selling ≤ 1``).
        ``direct_selling = 0`` for indirect channel.
        ``direct_selling = 1`` for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermediaries (``0 ≤ commission ≤ 1``).

        .. note::
            The commission is only relevant when there is an indirect channel (``direct_selling < 1``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    mpa : float
        Maximum Profit measure for customer Acquisition

    threshold : float
        Fraction of the leads that should be targeted to maximize profit

    Examples
    --------

    >>> from empulse.metrics import mpa
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpa(y_true, y_score)
    (3706.25, 0.875)
    """
    profits, customer_thresholds = compute_profit_acquisition(
        y_true,
        y_score,
        contribution,
        contact_cost,
        sales_cost,
        direct_selling,
        commission,
        check_input,
    )
    max_profit_index = np.argmax(profits)
    return profits[max_profit_index], customer_thresholds[max_profit_index]


def compute_profit_acquisition(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        contribution: float = 8_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
        check_input: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if check_input:
        y_true, y_pred = _validate_input_deterministic(
            y_true,
            y_pred,
            contribution,
            contact_cost,
            sales_cost,
            direct_selling,
            commission
        )
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
    cost_benefits = _compute_cost_benefits(contribution, contact_cost, sales_cost, direct_selling, commission)
    return _compute_profits(y_true, y_pred, cost_benefits)


@lru_cache(maxsize=1)
def _compute_cost_benefits(
        contribution: float,
        contact_cost: float,
        sales_cost: float,
        direct_selling: float,
        commission: float,
) -> np.ndarray:
    true_positive_benefit = direct_selling * (contribution - contact_cost - sales_cost) + (1 - direct_selling) * (
            (1 - commission) * contribution - contact_cost
    )
    false_positive_cost = -contact_cost
    return np.array([true_positive_benefit, false_positive_cost])
