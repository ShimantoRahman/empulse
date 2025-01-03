import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_mp
from ..common import _compute_profits


def mpcs_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        loan_lost_rate: float = 0.275,
        roi: float = 0.2644,
        check_input: bool = True,
) -> float:
    """
    :func:`~empulse.metrics.mpcs()` but only returning the MPCS score.

    MPCS presumes a situation where a company is considering whether to grant a loan to a customer.
    Correctly identifying defaulters results in receiving a return on investment (ROI), while incorrectly
    identifying non-defaulters as defaulters results in a fraction of the loan amount being lost.
    For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.mpcs` : to also return the fraction of loan applications that
        should be accepted to maximize profit.

        :func:`~empulse.metrics.empcs_score` : for a stochastic version of this metric.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    loan_lost_rate : float, default=0.275
        The fraction of the loan amount which is lost after default (``loan_lost_rate ≥ 0``).

    roi : float, default=0.2644
        Return on investment on the loan (``roi ≥ 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    mpcs : float
        Maximum Profit measure for Credit Scoring.

    Notes
    -----
    The MP measure for Credit Scoring is defined as [1]_:

    .. math:: \\max_t \\lambda \\pi_0 F_0(t) - ROI \\pi_1 F_1(t)

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
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpcs_score(y_true, y_score)
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
    >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> scorer = make_scorer(
    ...     mpcs_score,
    ...     response_method='predict_proba',
    ...     roi=0.2,
    ...     loan_lost_rate=0.25
    ... )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    0.123
    """
    return mpcs(y_true, y_score, loan_lost_rate=loan_lost_rate, roi=roi, check_input=check_input)[0]


def mpcs(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        loan_lost_rate: float = 0.275,
        roi: float = 0.2644,
        check_input: bool = True,
) -> tuple[float, float]:
    """
    Maximum Profit measure for Credit Scoring.

    MPCS presumes a situation where a company is considering whether to grant a loan to a customer.
    Correctly identifying defaulters results in receiving a return on investment (ROI), while incorrectly
    identifying non-defaulters as defaulters results in a fraction of the loan amount being lost.
    For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.mpcs_score` : to only return the MPCS score.

        :func:`~empulse.metrics.empcs` : for a stochastic version of this metric.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    loan_lost_rate : float, default=0.275
        The fraction of the loan amount which is lost after default (``loan_lost_rate ≥ 0``).

    roi : float, default=0.2644
        Return on investment on the loan (``roi ≥ 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    mpcs : float
        Maximum Profit measure for Credit Scoring

    threshold : float
        Fraction of loan applications that should be accepted to maximize profit

    Notes
    -----
    The MP measure for Credit Scoring is defined as [1]_:

    .. math:: \\max_t \\lambda \\pi_0 F_0(t) - ROI \\pi_1 F_1(t)

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
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> mpcs(y_true, y_score)
    (0.038349999999999995, 0.875)
    """
    profits, customer_thresholds = compute_profit_credit_scoring(y_true, y_score, loan_lost_rate, roi, check_input)
    max_profit_index = np.argmax(profits)

    return profits[max_profit_index], customer_thresholds[max_profit_index]


def compute_profit_credit_scoring(
        y_true: ArrayLike,
        y_score: ArrayLike,
        frac_loan_lost: float = 0.275,
        roi: float = 0.2644,
        check_input: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if check_input:
        y_true, y_score = _validate_input_mp(y_true, y_score, frac_loan_lost, roi)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
    cost_benefits = np.array([frac_loan_lost, -roi])
    return _compute_profits(y_true, y_score, cost_benefits)
