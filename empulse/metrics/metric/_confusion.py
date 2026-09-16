"""
The ranking-based confusion counts every threshold/rate computation is built from.

Sorting by predicted score and accumulating true/false positives as the targeted fraction grows
is the shared arithmetic behind :func:`~empulse.metrics.classification_threshold` (in the parent
``metrics`` package) and the deterministic :class:`~empulse.metrics.MaxProfit` path
(``strategies/max_profit_strategy/deterministic.py``). It lives here, inside ``metric/``, rather
than in the top-level ``metrics/common.py`` module that used to hold it, because the deepest
strategy modules need it directly -- reaching up to a top-level module for it was a layering
inversion the public :mod:`empulse.metrics` package now avoids by importing it back down.
"""

import numpy as np

from ..._types import FloatNDArray


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
