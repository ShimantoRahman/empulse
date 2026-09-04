"""
Reference (native/legacy) implementations of the credit scoring metrics.

These are the hand-written math implementations that used to live in
``empulse.metrics.credit_scoring`` before it was refactored to build ``mpcs_score`` and
``empcs_score`` from :class:`~empulse.metrics.Metric`/:class:`~empulse.metrics.MixtureMetric`
instances instead. They are kept here, unchanged, purely as ground truth to numerically verify
the new prebuilt metrics against, and to keep exercising the original edge-case and
input-validation test coverage in ``test_credit_scoring.py``.
"""

import warnings

import numpy as np

from empulse._types import FloatArrayLike, FloatNDArray
from empulse.metrics._cy_convex_hull import convex_hull
from empulse.metrics._validation import _check_fraction, _check_positive, _check_shape, _check_y_pred, _check_y_true

from ._common import _compute_prior_class_probabilities, _compute_profits, _compute_tpr_fpr_diffs

# --- Input validation ----------------------------------------------------------------------


def _validate_input(y_true: FloatArrayLike, y_pred: FloatArrayLike, roi: float) -> tuple[FloatNDArray, FloatNDArray]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    _check_positive(roi, 'roi')
    return y_true, y_pred


def _validate_input_mp(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, default_prob: float, roi: float
) -> tuple[FloatNDArray, FloatNDArray]:
    _check_fraction(default_prob, 'default_prob')
    return _validate_input(y_true, y_pred, roi)


def _validate_input_emp(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, p_0: float, p_1: float, roi: float
) -> tuple[FloatNDArray, FloatNDArray]:
    _check_fraction(p_0, 'p_0')
    _check_fraction(p_1, 'p_1')
    return _validate_input(y_true, y_pred, roi)


# --- Deterministic: mpcs / mpcs_score --------------------------------------------------------


def mpcs_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    loan_lost_rate: float = 0.275,
    roi: float = 0.2644,
    check_input: bool = True,
) -> float:
    """Maximum Profit measure for Credit Scoring, only returning the MPCS score."""
    return mpcs(y_true, y_score, loan_lost_rate=loan_lost_rate, roi=roi, check_input=check_input)[0]


def mpcs(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    loan_lost_rate: float = 0.275,
    roi: float = 0.2644,
    check_input: bool = True,
) -> tuple[float, float]:
    """Maximum Profit measure for Credit Scoring."""
    profits, customer_thresholds = _compute_profit_credit_scoring(y_true, y_score, loan_lost_rate, roi, check_input)
    max_profit_index = np.argmax(profits)

    return profits[max_profit_index], customer_thresholds[max_profit_index]


def _compute_profit_credit_scoring(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    frac_loan_lost: float = 0.275,
    roi: float = 0.2644,
    check_input: bool = True,
) -> tuple[FloatNDArray, FloatNDArray]:
    if check_input:
        y_true, y_score = _validate_input_mp(y_true, y_score, frac_loan_lost, roi)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
    cost_benefits = np.array([frac_loan_lost, -roi])
    return _compute_profits(y_true, y_score, cost_benefits)


# --- Stochastic: empcs / empcs_score ---------------------------------------------------------


def empcs_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    success_rate: float = 0.55,
    default_rate: float = 0.1,
    roi: float = 0.2644,
    check_input: bool = True,
) -> float:
    """Expected Maximum Profit measure for Credit Scoring, only returning the EMPCS score."""
    return empcs(
        y_true,
        y_score,
        success_rate=success_rate,
        default_rate=default_rate,
        roi=roi,
        check_input=check_input,
    )[0]


def empcs(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    success_rate: float = 0.55,
    default_rate: float = 0.1,
    roi: float = 0.2644,
    check_input: bool = True,
) -> tuple[float, float]:
    """Expected Maximum Profit measure for Credit Scoring."""
    if check_input:
        y_true, y_score = _validate_input_emp(y_true, y_score, success_rate, default_rate, roi)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    alpha = 1 - success_rate - default_rate
    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = convex_hull(y_true, y_score)
    tpr_diff, fpr_diff = _compute_tpr_fpr_diffs(true_positive_rates, false_positive_rates)

    lambda_cdf_diff, lambda_cdf_sum = _compute_lambda_cdf(
        roi, tpr_diff, fpr_diff, positive_class_prob, negative_class_prob
    )

    cutoff = len(true_positive_rates) - len(lambda_cdf_diff)
    if cutoff > 0:
        true_positive_rates = true_positive_rates[:-cutoff]
        false_positive_rates = false_positive_rates[:-cutoff]

    temp_1 = positive_class_prob * true_positive_rates * lambda_cdf_sum / 2
    temp_2 = roi * false_positive_rates * negative_class_prob
    partial_default_term: float = np.sum(alpha * lambda_cdf_diff * (temp_1 - temp_2))
    full_default_term = default_rate * (
        positive_class_prob * true_positive_rates[-1] - roi * negative_class_prob * false_positive_rates[-1]
    )
    empcs = partial_default_term + full_default_term

    customer_threshold = np.sum(
        alpha
        * lambda_cdf_diff
        * (positive_class_prob * true_positive_rates + negative_class_prob * false_positive_rates)
    ) + default_rate * (positive_class_prob * true_positive_rates[-1] + negative_class_prob * false_positive_rates[-1])

    return empcs, customer_threshold


def _compute_lambda_cdf(
    roi: float, tpr_diff: FloatNDArray, fpr_diff: FloatNDArray, positive_class_prob: float, negative_class_prob: float
) -> tuple[FloatNDArray, FloatNDArray]:
    # ignore division by zero warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        lambda_bounds = negative_class_prob * roi / positive_class_prob * (fpr_diff / tpr_diff)  # type: ignore[operator]
    lambda_bounds = np.append(0, lambda_bounds)
    lambda_bounds = np.append(lambda_bounds[lambda_bounds < 1], 1)
    return np.diff(lambda_bounds), lambda_bounds[1:] + lambda_bounds[:-1]
