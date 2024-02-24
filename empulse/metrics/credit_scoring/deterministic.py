import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_mp
from ..common import _compute_profits


def mpcs_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        loan_lost_rate: float = 0.275,
        roi: float = 0.2644
) -> float:
    """
    Convenience function around :func:`~empulse.metrics.mpcs()` only returning MPCS score

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    loan_lost_rate : float, default=0.275
        The fraction of the loan amount which is lost after default (``loan_lost_rate ≥ 0``).

    roi : float, default=0.2644
        Return on investment on the loan (``roi ≥ 0``).

    Returns
    -------
    mpcs : float
        Maximum Profit measure for customer Credit Scoring.

    Notes
    -----
    The MP measure for Credit Scoring is defined as [1]_:

    .. math:: \max_t \lambda \pi_0 F_0(t) - ROI \pi_1 F_1(t)

    The MP measure for Credit Scoring requires that the default class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('default': 1, 'no default': 0).

    References
    ----------
    .. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
        Development and application of consumer credit scoring models using profit-based classification measures.
        European Journal of Operational Research, 238(2), 505-513.

    Examples
    --------

    >>> from empulse.metrics import mpcs_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpcs_score(y_true, y_pred)
    0.038349999999999995

    Using scorer:

    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_score, StratifiedKFold
    >>> from sklearn.metrics import make_scorer
    >>> from empulse.metrics import mpcs_score
    >>>
    >>> X, y = make_classification(random_state=42)
    >>> model = LogisticRegression()
    >>> cv = StratifiedKFold(n_splits=5, random_state=42)
    >>> scorer = make_scorer(
    >>>     mpcs_score,
    >>>     needs_proba=True,
    >>>     roi=0.2,
    >>>     loan_lost_rate=0.25
    >>> )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    0.123
    """
    return mpcs(y_true, y_pred, loan_lost_rate=loan_lost_rate, roi=roi)[0]


def mpcs(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        loan_lost_rate: float = 0.275,
        roi: float = 0.2644
) -> tuple[float, float]:
    """
    Maximum Profit measure for Credit Scoring
    =========================================

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    loan_lost_rate : float, default=0.275
        The fraction of the loan amount which is lost after default (``loan_lost_rate ≥ 0``).

    roi : float, default=0.2644
        Return on investment on the loan (``roi ≥ 0``).

    Returns
    -------
    mpcs : float
        Maximum Profit measure for customer Credit Scoring

    threshold : float
        Threshold at which the maximum profit is achieved

    Notes
    -----
    The MP measure for Credit Scoring is defined as [1]_:

    .. math:: \max_t \lambda \pi_0 F_0(t) - ROI \pi_1 F_1(t)

    The MP measure for Credit Scoring requires that the default class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('default': 1, 'no default': 0).

    References
    ----------
    .. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
        Development and application of consumer credit scoring models using profit-based classification measures.
        European Journal of Operational Research, 238(2), 505-513.

    Examples
    --------

    >>> from empulse.metrics import mpcs
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpcs(y_true, y_pred)
    (0.038349999999999995, 0.875)
    """
    profits, customer_thresholds = compute_profit_credit_scoring(y_true, y_pred, loan_lost_rate, roi)
    max_profit_index = np.argmax(profits)

    return profits[max_profit_index], customer_thresholds[max_profit_index]


def compute_profit_credit_scoring(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        frac_loan_lost: float = 0.275,
        roi: float = 0.2644
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = _validate_input_mp(y_true, y_pred, frac_loan_lost, roi)
    cost_benefits = np.array([frac_loan_lost, -roi])
    return _compute_profits(y_true, y_pred, cost_benefits)
