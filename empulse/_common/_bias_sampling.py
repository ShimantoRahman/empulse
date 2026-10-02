"""
Relabelling and resampling for bias mitigation, free of imbalanced-learn.

:class:`~empulse.samplers.BiasRelabler` and :class:`~empulse.samplers.BiasResampler` wrap these in the
imbalanced-learn sampler API, and :class:`~empulse.models.BiasRelabelingClassifier` and
:class:`~empulse.models.BiasResamplingClassifier` call them directly, so the models work without the
``sampling`` extra installed.
"""

import warnings
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.utils import _safe_indexing, check_random_state

from .._types import FloatNDArray, IntNDArray
from ._strategies import Strategy, StrategyFn, _independent_weights

PairsStrategyFn = Callable[[NDArray[Any], NDArray[Any]], int]


def _independent_pairs(y_true: ArrayLike, sensitive_feature: NDArray[Any]) -> int:
    """Determine promotion and demotion pairs so that y is statistically independent of sensitive feature."""
    sensitive_indices = np.where(sensitive_feature == 0)[0]
    not_sensitive_indices = np.where(sensitive_feature == 1)[0]
    n_sensitive = len(sensitive_indices)
    n_not_sensitive = len(not_sensitive_indices)
    n = n_sensitive + n_not_sensitive

    # no swapping needed if one of the groups is empty
    if n_sensitive == 0 or n_not_sensitive == 0:
        warnings.warn(
            'sensitive_feature only contains one class, no relabeling is performed.',
            UserWarning,
            stacklevel=2,
        )
        return 0

    pos_ratio_sensitive = np.sum(_safe_indexing(y_true, sensitive_indices)) / n_sensitive
    pos_ratio_not_sensitive = np.sum(_safe_indexing(y_true, not_sensitive_indices)) / n_not_sensitive

    discrimination = pos_ratio_not_sensitive - pos_ratio_sensitive

    # number of pairs to swap label
    return int(abs(round((discrimination * n_sensitive * n_not_sensitive) / n)))


RELABEL_STRATEGIES: dict[Strategy, PairsStrategyFn] = {
    'statistical parity': _independent_pairs,
    'demographic parity': _independent_pairs,
}

RESAMPLE_STRATEGIES: dict[Strategy, StrategyFn] = {
    'statistical parity': _independent_weights,
    'demographic parity': _independent_weights,
}


def _get_demotion_candidates(y_pred: FloatNDArray, y_true: FloatNDArray, n_pairs: int) -> FloatNDArray:
    """Return the n_pairs instances with the lowest probability of being positive class label."""
    positive_indices = np.where(y_true == 1)[0]
    positive_predictions = y_pred[positive_indices]
    demotion_candidates: FloatNDArray = positive_indices[np.argsort(positive_predictions)[:n_pairs]]
    return demotion_candidates


def _get_promotion_candidates(y_pred: FloatNDArray, y_true: FloatNDArray, n_pairs: int) -> FloatNDArray:
    """Return the n_pairs instances with the lowest probability of being negative class label."""
    negative_indices = np.where(y_true == 0)[0]
    negative_predictions = y_pred[negative_indices]
    promotion_candidates: FloatNDArray = negative_indices[np.argsort(negative_predictions)[-n_pairs:]]
    return promotion_candidates


def relabel(
    X: NDArray[Any],
    y: Any,
    sensitive_feature: NDArray[Any],
    classes: NDArray[Any],
    *,
    estimator: Any,
    strategy: PairsStrategyFn | Strategy,
    transform_feature: Callable[[NDArray[Any]], IntNDArray] | None,
) -> tuple[Any, Any]:
    """
    Fit a clone of ``estimator`` and swap labels between groups according to ``strategy``.

    ``y`` must be binary with two classes, ``classes`` being its sorted unique values.

    Returns
    -------
    y : array-like
        The relabelled target, of the same container type as the input ``y``.
    estimator : Estimator instance
        The fitted clone of ``estimator`` that ranked the relabelling candidates.
    """
    y_binarized = np.where(y == classes[1], 1, 0)

    if transform_feature is not None:
        sensitive_feature = transform_feature(sensitive_feature)

    fitted_estimator = clone(estimator)
    fitted_estimator.fit(X, y)
    y_pred = fitted_estimator.predict_proba(X)[:, 1]

    strategy_fn = RELABEL_STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    n_pairs = strategy_fn(y_binarized, sensitive_feature)
    if n_pairs <= 0:
        return np.asarray(y), fitted_estimator

    sensitive_indices = np.where(sensitive_feature == 0)[0]
    non_sensitive = np.where(sensitive_feature == 1)[0]
    probas_non_sensitive = y_pred[non_sensitive]
    probas_sensitive = y_pred[sensitive_indices]

    # Candidates are chosen on the 0/1-encoded target, and relabelled with the original labels.
    demotion_candidates = _get_demotion_candidates(probas_non_sensitive, y_binarized[non_sensitive], n_pairs)
    promotion_candidates = _get_promotion_candidates(probas_sensitive, y_binarized[sensitive_indices], n_pairs)
    negative_label, positive_label = classes

    # map promotion and demotion candidates to original indices
    indices = np.arange(len(y))
    demotion_candidates = indices[non_sensitive][demotion_candidates]
    promotion_candidates = indices[sensitive_indices][promotion_candidates]

    # relabel the data
    if hasattr(y, 'copy'):
        relabeled_y = y.copy()
    elif hasattr(y, 'clone'):
        relabeled_y = y.clone()
    else:
        relabeled_y = np.copy(y)

    if hasattr(relabeled_y, 'loc'):
        relabeled_y.loc[demotion_candidates] = negative_label
        relabeled_y.loc[promotion_candidates] = positive_label
    else:
        relabeled_y[demotion_candidates] = negative_label
        relabeled_y[promotion_candidates] = positive_label

    return relabeled_y, fitted_estimator


def resample_indices(
    y: NDArray[Any],
    sensitive_feature: NDArray[Any],
    classes: NDArray[Any],
    *,
    strategy: StrategyFn | Strategy,
    transform_feature: Callable[[NDArray[Any]], IntNDArray] | None,
    random_state: RandomState | int | None,
) -> IntNDArray:
    """
    Return the row indices of a resample that reweights groups according to ``strategy``.

    ``y`` must be binary with two classes, ``classes`` being its sorted unique values. When no
    resampling is needed the indices are simply ``arange(len(y))``.
    """
    y_binarized = np.where(y == classes[1], 1, 0)
    rng = check_random_state(random_state)

    if transform_feature is not None:
        sensitive_feature = transform_feature(sensitive_feature)

    strategy_fn = RESAMPLE_STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    class_weights = strategy_fn(y_binarized, sensitive_feature)
    # if class_weights are all 1, no resampling is needed
    if np.allclose(class_weights, np.ones(class_weights.shape)):
        return np.arange(len(y))

    unique_attr = np.unique(sensitive_feature)
    if len(unique_attr) == 1:
        warnings.warn(
            'sensitive_feature only contains one class, no resampling is performed.',
            UserWarning,
            stacklevel=3,
        )
        return np.arange(len(y))

    indices = np.empty((0,), dtype=int)
    # determine the number of samples to be drawn for each class and sensitive_feature value
    for target_class, sensitive_val in product(np.unique(y_binarized), unique_attr):
        sensitive_val = int(sensitive_val)
        idx_class = np.flatnonzero(y_binarized == target_class)
        idx_sensitive_feature = np.flatnonzero(sensitive_feature == sensitive_val)
        idx_class_sensitive = np.intersect1d(idx_class, idx_sensitive_feature)
        n_samples = int(class_weights[target_class, sensitive_val] * len(idx_class_sensitive))
        if n_samples > len(idx_class_sensitive):  # oversampling
            indices = np.concatenate((indices, idx_class_sensitive))
            indices = np.concatenate((
                indices,
                rng.choice(idx_class_sensitive, n_samples - len(idx_class_sensitive), replace=True),
            ))
        else:  # undersampling
            indices = np.concatenate((indices, rng.choice(idx_class_sensitive, n_samples, replace=False)))

    return indices
