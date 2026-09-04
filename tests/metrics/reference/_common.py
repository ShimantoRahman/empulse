"""
Shared helper functions used by the reference (native/legacy) metric implementations.

These functions used to live in ``empulse.metrics.common`` but became unused by the package
itself once ``metrics.acquisition``, ``metrics.churn``, and ``metrics.credit_scoring`` were
refactored to build their metrics from :class:`~empulse.metrics.Metric` instances instead of
hand-written math. They are kept here, verbatim, purely so the reference implementations below
(used to numerically verify the new prebuilt metrics) keep working unchanged.
"""

import numpy as np

from empulse._types import FloatNDArray


def _compute_confusion_matrix(
    y_true: FloatNDArray, y_pred: FloatNDArray
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    # sort true labels and predictions by highest to the lowest predicted score
    sorted_indices = y_pred.argsort()[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_predictions = y_pred[sorted_indices]

    # calculate the TP & FP at each new lead targeted
    true_positives = np.pad(np.cumsum(sorted_labels), pad_width=(1, 0))
    false_positives = np.pad(np.cumsum(sorted_labels == 0), pad_width=(1, 0))

    # merge consecutive equal prediction values
    duplicated_prediction_indices = np.where(np.diff(sorted_predictions) == 0)[0] + 1
    true_positives = np.delete(true_positives, duplicated_prediction_indices)
    false_positives = np.delete(false_positives, duplicated_prediction_indices)

    return np.array([true_positives, false_positives]), sorted_indices, duplicated_prediction_indices


def _compute_prior_class_probabilities(y_true: FloatNDArray) -> tuple[float, float]:
    """Calculate prior class probabilities from target values."""
    positive_class_prob = float(np.mean(y_true))  # pi_0
    negative_class_prob = 1 - positive_class_prob  # pi_1

    return positive_class_prob, negative_class_prob


def _compute_tpr_fpr_diffs(
    true_positive_rates: FloatNDArray, false_positive_rates: FloatNDArray
) -> tuple[FloatNDArray, FloatNDArray]:
    """Calculate differences between subsequent true positive rates and false positive rates."""
    tpr_diff = np.diff(true_positive_rates, axis=0)  # F_0(T_i) - F_0(T_{i-1})
    fpr_diff = np.diff(false_positive_rates, axis=0)  # F_1(T_i) - F_1(T_{i-1})

    return tpr_diff, fpr_diff


def _compute_profits(
    y_true: FloatNDArray, y_pred: FloatNDArray, cost_benefits: FloatNDArray
) -> tuple[FloatNDArray, FloatNDArray]:
    n_samples = y_pred.shape[0]
    confusion_matrix, _, _ = _compute_confusion_matrix(y_true, y_pred)
    profit_matrix = np.dot(confusion_matrix.T, cost_benefits) / n_samples
    customer_thresholds = np.sum(confusion_matrix, axis=0) / n_samples
    return profit_matrix, customer_thresholds
