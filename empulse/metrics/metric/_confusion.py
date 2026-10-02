"""
The ranked confusion counts that every threshold and rate computation builds on.

Sorting by predicted score and accumulating true and false positives as the targeted fraction grows
is shared by :func:`~empulse.metrics.classification_threshold` and the deterministic
:class:`~empulse.metrics.MaxProfit` path (``strategies/max_profit_strategy/deterministic.py``).
It lives in ``metric/`` because the strategy modules import it.
"""

import numpy as np

from ..._types import FloatNDArray


def _compute_confusion_matrix(
    y_true: FloatNDArray, y_pred: FloatNDArray
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    sorted_indices = y_pred.argsort()[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_predictions = y_pred[sorted_indices]

    # Counts after targeting 0, 1, ... samples.
    true_positives = np.pad(np.cumsum(sorted_labels), pad_width=(1, 0))
    false_positives = np.pad(np.cumsum(sorted_labels == 0), pad_width=(1, 0))

    # Tied scores form one threshold: keep only the counts after the last sample of each tie.
    duplicated_prediction_indices = np.where(np.diff(sorted_predictions) == 0)[0] + 1
    true_positives = np.delete(true_positives, duplicated_prediction_indices)
    false_positives = np.delete(false_positives, duplicated_prediction_indices)

    return np.array([true_positives, false_positives]), sorted_indices, duplicated_prediction_indices
