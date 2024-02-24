from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_deterministic
from ..common import _compute_profits


def mpa_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        contribution: float = 8_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
) -> float:
    """
    Convenience function around :func:`~empulse.metrics.mpa()` only returning MPA score

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_pred : array-like of shape (n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    contribution : float, default=8000
        Average contribution of a new customer (`contribution` ≥ 0).

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (`sales_cost` ≥ 0).

    contact_cost : float, default=50
        Average contact cost of targeted leads (`contact_cost` ≥ 0).

    direct_selling : float, default=1
        Fraction of leads sold to directly (0 ≤ `direct_selling` ≤ 1).
        `direct_selling` = 0 for indirect channel.
        `direct_selling` = 1 for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermediaries (0 ≤ commission ≤ 1).

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
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpa_score(y_true, y_pred, direct_selling=1)
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
    >>> cv = StratifiedKFold(n_splits=5, random_state=42)
    >>> scorer = make_scorer(
    >>>     mpa_score,
    >>>     needs_proba=True,
    >>>     contribution=7_000,
    >>>     sales_cost=2_000,
    >>>     contact_cost=100,
    >>>     direct_selling=0,
    >>> )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    299.0
    """
    return mpa(
        y_true,
        y_pred,
        contribution=contribution,
        contact_cost=contact_cost,
        sales_cost=sales_cost,
        direct_selling=direct_selling,
        commission=commission
    )[0]


def mpa(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        contribution: float = 8_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
) -> tuple[float, float]:
    """
    Maximum Profit measure for customer Acquisition (MPA)
    =====================================================

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_pred : array-like of shape (n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    contribution : float, default=8000
        Average contribution of a new customer (`contribution` ≥ 0).

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (`sales_cost` ≥ 0).

    contact_cost : float, default=50
        Average contact cost of targeted leads (`contact_cost` ≥ 0).

    direct_selling : float, default=1
        Fraction of leads sold to directly (0 ≤ `direct_selling` ≤ 1).
        `direct_selling` = 0 for indirect channel.
        `direct_selling` = 1 for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermedaries (0 ≤ commission ≤ 1).

    Returns
    -------
    mpa : float
        Maximum Profit measure for customer Acquisition

    threshold : float
        Threshold at which the maximum profit is achieved

    Examples
    --------

    >>> from empulse.metrics import mpa
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpa(y_true, y_pred)
    (3706.25, 0.875)
    """
    profits, customer_thresholds = compute_profit_acquisition(
        y_true,
        y_pred,
        contribution,
        contact_cost,
        sales_cost,
        direct_selling,
        commission
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
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = _validate_input_deterministic(
        y_true,
        y_pred,
        contribution,
        contact_cost,
        sales_cost,
        direct_selling,
        commission
    )
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
