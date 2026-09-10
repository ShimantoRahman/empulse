"""Shared out-of-bag weighting utilities for CSForestClassifier and CSBaggingClassifier."""

import threading
from collections.abc import Callable
from typing import Any

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
        The loss computed for each estimator (on its out-of-bag samples), lower is better.

    Returns
    -------
    weights : ndarray of shape (n_estimators,)
        Non-negative weights summing to 1. Falls back to a uniform distribution when every
        estimator has the same goodness (nothing to distinguish them by) or when the values are
        non-finite.
    """
    goodness: FloatNDArray = -np.asarray(losses, dtype=np.float64)
    n_estimators = goodness.shape[0]
    if not np.isfinite(goodness).all():
        return np.full(n_estimators, 1.0 / n_estimators)
    shifted = goodness - goodness.min()
    total = shifted.sum()
    if total == 0.0:
        return np.full(n_estimators, 1.0 / n_estimators)
    weights: FloatNDArray = shifted / total
    return weights


def subset_loss_params(loss_params: dict[str, Any], index: Any, n_samples: int) -> dict[str, Any]:
    """
    Subset every instance-dependent (array-valued) entry of *loss_params* by *index*.

    Mirrors how ``y`` and ``y_score`` are subset to the out-of-bag rows before being passed to the
    loss: an array-valued parameter of length *n_samples* describes one value per training sample
    and must be subset the same way, or the metric would evaluate it against the wrong rows (and,
    for a length that no longer matches the OOB subset, raise a shape-mismatch error). Class-dependent
    (scalar) parameters, and arrays not aligned with the full training set, pass through unchanged.

    Parameters
    ----------
    loss_params : dict
        Keyword arguments that will be passed to the loss/metric.
    index : ndarray of int or bool
        The out-of-bag row selector (indices or boolean mask) to apply.
    n_samples : int
        The number of samples in the full training set (``y.shape[0]`` at fit time).

    Returns
    -------
    subset : dict
        *loss_params* with every length-``n_samples`` array entry subset by *index*.
    """
    subset: dict[str, Any] = {}
    for key, value in loss_params.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_samples:
            subset[key] = value[index]
        else:
            subset[key] = value
    return subset


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
