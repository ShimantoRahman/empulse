import warnings

import numpy as np
from numpy.typing import ArrayLike

from ._validation import _validate_input_emp
from ..common import _compute_prior_class_probabilities, _compute_tpr_fpr_diffs
from .._convex_hull import _compute_convex_hull


def empcs_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        success_rate: float = 0.55,
        default_rate: float = 0.1,
        roi: float = 0.2644
) -> float:
    """
    Convenience function around `empcs()` only returning EMPCS score

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    success_rate : float, default=0.55
        Probability that the entire loan is paid back.

    default_rate : float, default=0.1
        Probability that the entire loan is lost.

    roi : float, default=0.2644
        Return on investment on the loan (roi ≥ 0).

    Returns
    -------
    empcs : float
        Expected Maximum Profit measure for customer Credit Scoring.

    Notes
    -----
    The EMP measure for Credit Scoring is defined as [1]_:

    .. math:: \int_0^1 \lambda \pi_0 F_0(T) - ROI \pi_1 F_1(T) \cdot h(\lambda) d\lambda

    The EMP measure for Credit Scoring requires that the default class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('default': 1, 'no default': 0).

    References
    ----------
    .. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
        Development and application of consumer credit scoring models using profit-based classification measures.
        European Journal of Operational Research, 238(2), 505-513.

    Examples
    --------

    >>> from empulse.metrics import empcs_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empcs_score(y_true, y_pred)
    0.09747017050000001

    Using scorer:

    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_score, StratifiedKFold
    >>> from sklearn.metrics import make_scorer
    >>> from empulse.metrics import empcs_score
    >>>
    >>> X, y = make_classification(random_state=42)
    >>> model = LogisticRegression()
    >>> cv = StratifiedKFold(n_splits=5, random_state=42)
    >>> scorer = make_scorer(
    >>>     empcs_score,
    >>>     needs_proba=True,
    >>>     roi=0.2,
    >>>     success_rate=0.5
    >>>     default_rate=0.1
    >>> )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    0.14904
    """
    return empcs(y_true, y_pred, success_rate=success_rate, default_rate=default_rate, roi=roi)[0]


def empcs(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        success_rate: float = 0.55,
        default_rate: float = 0.1,
        roi: float = 0.2644
) -> tuple[float, float]:
    """
    Expected Maximum Profit measure for Credit Scoring
    ==================================================

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    success_rate : float, default=0.55
        Probability that the entire loan is paid back.

    default_rate : float, default=0.1
        Probability that the entire loan is lost.

    roi : float, default=0.2644
        Return on investment on the loan (roi ≥ 0).

    Returns
    -------
    (empcs, threshold) : tuple[float, float, float]
        Expected Maximum Profit measure for customer Credit Scoring and
        the threshold η at which the expected maximum profit is achieved.

    Notes
    -----
    The EMP measure for Credit Scoring is defined as [1]_:

    .. math:: \int_0^1 \lambda \pi_0 F_0(T) - ROI \pi_1 F_1(T) \cdot h(\lambda) d\lambda

    The EMP measure for Credit Scoring requires that the default class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('default': 1, 'no default': 0).

    References
    ----------
    .. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
        Development and application of consumer credit scoring models using profit-based classification measures.
        European Journal of Operational Research, 238(2), 505-513.

    Examples
    --------

    >>> from empulse.metrics import empcs
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empcs(y_true, y_pred)
    (0.09747017050000001, 0.32434500000000005)
    """
    y_true, y_pred = _validate_input_emp(y_true, y_pred, success_rate, default_rate, roi)

    alpha = 1 - success_rate - default_rate
    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_pred)
    tpr_diff, fpr_diff = _compute_tpr_fpr_diffs(true_positive_rates, false_positive_rates)

    lambda_cdf_diff, lambda_cdf_sum = _compute_lambda_cdf(roi, tpr_diff, fpr_diff, positive_class_prob,
                                                          negative_class_prob)

    cutoff = len(true_positive_rates) - len(lambda_cdf_diff)
    if cutoff > 0:
        true_positive_rates = true_positive_rates[:-cutoff]
        false_positive_rates = false_positive_rates[:-cutoff]

    M = positive_class_prob * true_positive_rates * lambda_cdf_sum / 2
    N = roi * false_positive_rates * negative_class_prob
    partial_default_term = np.sum(alpha * lambda_cdf_diff * (M - N))
    full_default_term = default_rate * (
            positive_class_prob * true_positive_rates[-1] -
            roi * negative_class_prob * false_positive_rates[-1]
    )
    empcs = partial_default_term + full_default_term

    customer_threshold = np.sum(alpha * lambda_cdf_diff *
                                (positive_class_prob * true_positive_rates +
                                 negative_class_prob * false_positive_rates)) + \
                         default_rate * (positive_class_prob * true_positive_rates[-1] +
                                negative_class_prob * false_positive_rates[-1])

    return empcs, customer_threshold


def _compute_lambda_cdf(
        roi: float,
        tpr_diff: np.ndarray,
        fpr_diff: np.ndarray,
        positive_class_prob: float,
        negative_class_prob: float
) -> tuple[np.ndarray, np.ndarray]:
    # ignore division by zero warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        lambda_bounds = negative_class_prob * roi / positive_class_prob * (fpr_diff / tpr_diff)
    lambda_bounds = np.append(0, lambda_bounds)
    lambda_bounds = np.append(lambda_bounds[lambda_bounds < 1], 1)
    return np.diff(lambda_bounds), lambda_bounds[1:] + lambda_bounds[:-1]
