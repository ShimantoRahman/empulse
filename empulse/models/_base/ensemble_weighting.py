"""Shared out-of-bag weighting utilities for CSForestClassifier and CSBaggingClassifier."""

import threading
from collections.abc import Callable

import numpy as np

from ..._types import FloatNDArray


def goodness_weights(losses: FloatNDArray, scale: float = 1.0) -> FloatNDArray:
    """
    Convert per-estimator loss values into normalized, non-negative voting weights.

    Each estimator weighs ``1 + (mean_loss - loss) / scale``, clipped at zero, so its weight grows
    by how much better than average it is, measured in units of *scale*. Estimators whose losses
    differ little relative to the scale vote almost equally, and only one worse than average by a
    whole *scale* or more loses its vote.

    Parameters
    ----------
    losses : ndarray of shape (n_estimators,)
        The loss computed for each estimator (on its out-of-bag samples), lower is better. ``nan``
        for an estimator that has no out-of-bag samples to be judged on.
    scale : float, default=1.0
        The loss difference that counts as large: the average cost of a wrong decision, in the
        units of the loss (see ``loss_scale``).

    Returns
    -------
    weights : ndarray of shape (n_estimators,)
        Non-negative weights summing to 1. An estimator without a finite loss gets the average
        weight. Uniform when no estimator has a finite loss.
    """
    goodness: FloatNDArray = -np.asarray(losses, dtype=np.float64)
    n_estimators = goodness.shape[0]
    finite = np.isfinite(goodness)
    if not finite.any():
        return np.full(n_estimators, 1.0 / n_estimators)
    goodness = np.where(finite, goodness, np.mean(goodness[finite]))
    # The unclipped weights sum to n_estimators, so the clipped ones cannot sum to zero.
    shifted = np.maximum(1.0 + (goodness - goodness.mean()) / scale, 0.0)
    weights: FloatNDArray = shifted / shifted.sum()
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
