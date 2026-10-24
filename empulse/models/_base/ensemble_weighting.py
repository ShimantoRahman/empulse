"""Shared out-of-bag weighting utilities for CSForestClassifier and CSBaggingClassifier."""

import threading
from collections.abc import Callable

import numpy as np

from ..._types import FloatNDArray


def goodness_weights(losses: FloatNDArray) -> FloatNDArray:
    """
    Convert per-estimator loss values into normalized, non-negative voting weights.

    A loss points the wrong way for weighting -- a lower value is a better estimator, so it must be
    inverted before "more weight = better" -- and normalizing signed values by their raw sum is
    unsound (the sum can be near zero, or individual weights can end up negative). This maps every
    loss to a "goodness" score, shifts it to be non-negative, and normalizes it to sum to 1.

    Parameters
    ----------
    losses : ndarray of shape (n_estimators,)
        The loss computed for each estimator (on its out-of-bag samples), lower is better. ``nan``
        for an estimator that has no out-of-bag samples to be judged on.

    Returns
    -------
    weights : ndarray of shape (n_estimators,)
        Non-negative weights summing to 1. An estimator without a finite loss gets the average
        goodness of the others. Falls back to a uniform distribution when every estimator has the
        same goodness (nothing to distinguish them by) or none has a finite loss.
    """
    goodness: FloatNDArray = -np.asarray(losses, dtype=np.float64)
    n_estimators = goodness.shape[0]
    finite = np.isfinite(goodness)
    if not finite.any():
        return np.full(n_estimators, 1.0 / n_estimators)
    goodness = np.where(finite, goodness, np.mean(goodness[finite]))
    shifted = goodness - goodness.min()
    total = shifted.sum()
    if total == 0.0:
        return np.full(n_estimators, 1.0 / n_estimators)
    weights: FloatNDArray = shifted / total
    return weights


def accumulate_weighted_prediction(
    predict: Callable[[FloatNDArray], FloatNDArray],
    X: FloatNDArray,
    out: FloatNDArray,
    weight: float,
    lock: threading.Lock,
) -> None:
    """Add ``weight * predict(X)`` into *out* in-place, under *lock*."""
    prediction = predict(X)
    with lock:
        out += prediction * weight  # type: ignore[misc]
