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


def _check_sensitive_feature(
    sensitive_feature: ArrayLike,
    n_samples: int,
    transform_feature: Callable[[NDArray[Any]], IntNDArray] | None,
) -> IntNDArray:
    """
    Return the (transformed) sensitive feature as 0/1 integers, or raise if it is not binary.

    The strategies index their group weights by the feature's value and treat 1 and 0 as the two
    groups, so any other coding would silently pick the wrong groups or fail with an index error.
    """
    values = np.asarray(sensitive_feature)
    if len(values) != n_samples:
        raise ValueError(f'sensitive_feature must have the same length as y, got {len(values)} and {n_samples}.')
    if transform_feature is not None:
        values = np.asarray(transform_feature(values))
    if not np.isin(values, [0, 1]).all():
        raise ValueError(
            'sensitive_feature must be coded as 0 and 1, got the values '
            f'{np.unique(values).tolist()[:10]}. Pass a transform_feature that maps it to 0 and 1.'
        )
    return values.astype(np.intp)


def _independent_pairs(y_true: ArrayLike, sensitive_feature: NDArray[Any]) -> int:
    """
    Determine promotion and demotion pairs so that y is statistically independent of sensitive feature.

    The count is positive when the group with ``sensitive_feature == 1`` has the higher positive rate,
    and negative when the group with ``sensitive_feature == 0`` does.
    """
    sensitive_indices = np.where(sensitive_feature == 0)[0]
    not_sensitive_indices = np.where(sensitive_feature == 1)[0]
    n_sensitive = len(sensitive_indices)
    n_not_sensitive = len(not_sensitive_indices)
    n = n_sensitive + n_not_sensitive

    # With one group empty there is nothing to swap.
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

    return int(np.rint(discrimination * n_sensitive * n_not_sensitive / n))


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
    sensitive_feature: ArrayLike,
    classes: NDArray[Any],
    *,
    estimator: Any,
    strategy: PairsStrategyFn | Strategy,
    transform_feature: Callable[[NDArray[Any]], IntNDArray] | None,
    fit_params: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """
    Fit a clone of ``estimator`` and swap labels between groups according to ``strategy``.

    ``y`` must be binary with two classes, ``classes`` being its sorted unique values. ``fit_params``
    are passed to the estimator's ``fit``.

    Returns
    -------
    y : array-like
        The relabelled target, of the same container type as the input ``y``.
    estimator : Estimator instance
        The fitted clone of ``estimator`` that ranked the relabelling candidates.
    """
    y_binarized = np.where(y == classes[1], 1, 0)
    sensitive_feature = _check_sensitive_feature(sensitive_feature, len(y_binarized), transform_feature)

    fitted_estimator = clone(estimator)
    fitted_estimator.fit(X, y, **(fit_params or {}))
    y_pred = fitted_estimator.predict_proba(X)[:, 1]

    strategy_fn = RELABEL_STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    n_pairs = strategy_fn(y_binarized, sensitive_feature)
    if n_pairs == 0:
        return np.asarray(y), fitted_estimator

    # A positive count favours group 1, so its positives are demoted and group 0's negatives promoted.
    favoured_group, disfavoured_group = (1, 0) if n_pairs > 0 else (0, 1)
    n_pairs = abs(n_pairs)
    favoured_indices = np.where(sensitive_feature == favoured_group)[0]
    disfavoured_indices = np.where(sensitive_feature == disfavoured_group)[0]

    # Candidates are chosen on the 0/1-encoded target, and relabelled with the original labels.
    demotion_candidates = _get_demotion_candidates(y_pred[favoured_indices], y_binarized[favoured_indices], n_pairs)
    promotion_candidates = _get_promotion_candidates(
        y_pred[disfavoured_indices], y_binarized[disfavoured_indices], n_pairs
    )
    negative_label, positive_label = classes

    demotion_candidates = favoured_indices[demotion_candidates]
    promotion_candidates = disfavoured_indices[promotion_candidates]

    if hasattr(y, 'copy'):
        relabeled_y = y.copy()
    elif hasattr(y, 'clone'):
        relabeled_y = y.clone()
    else:
        relabeled_y = np.copy(y)

    if hasattr(relabeled_y, 'iloc'):
        relabeled_y.iloc[demotion_candidates] = negative_label
        relabeled_y.iloc[promotion_candidates] = positive_label
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
    sensitive_feature = _check_sensitive_feature(sensitive_feature, len(y_binarized), transform_feature)

    strategy_fn = RESAMPLE_STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    class_weights = strategy_fn(y_binarized, sensitive_feature)
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

    groups = []
    for target_class, sensitive_val in product(np.unique(y_binarized), unique_attr):
        idx_class_sensitive = np.flatnonzero((y_binarized == target_class) & (sensitive_feature == sensitive_val))
        groups.append((idx_class_sensitive, class_weights[target_class, int(sensitive_val)] * len(idx_class_sensitive)))

    # Largest-remainder rounding, so the group sizes add up to the (rounded) total the weights call for.
    targets = np.array([target for _, target in groups])
    sizes = np.floor(targets).astype(int)
    shortfall = round(float(targets.sum())) - int(sizes.sum())
    sizes[np.argsort(sizes - targets, kind='stable')[:shortfall]] += 1

    indices = np.empty((0,), dtype=int)
    for (idx_class_sensitive, _), n_samples in zip(groups, sizes, strict=True):
        if n_samples > len(idx_class_sensitive):
            indices = np.concatenate((indices, idx_class_sensitive))
            indices = np.concatenate((
                indices,
                rng.choice(idx_class_sensitive, n_samples - len(idx_class_sensitive), replace=True),
            ))
        else:
            indices = np.concatenate((indices, rng.choice(idx_class_sensitive, n_samples, replace=False)))

    # Mixed, so the resample is not ordered by class and group.
    return np.asarray(rng.permutation(indices))
