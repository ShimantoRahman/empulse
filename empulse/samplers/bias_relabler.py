from typing import Union, Callable, Optional, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import OneToOneFeatureMixin, BaseEstimator, clone
from sklearn.utils import _safe_indexing

from empulse.samplers._strategies import Strategy, StrategyFn

_XT = TypeVar('_XT', bound=ArrayLike)


def _independent_pairs(y_true: ArrayLike, protected_attr: np.ndarray) -> int:
    """
    Determine the number of promotion and demotion pairs
    so that y would be statistically independent of the protected attribute.
    """
    protected_indices = np.where(protected_attr == 0)[0]
    unprotected_indices = np.where(protected_attr == 1)[0]
    n_protected = len(protected_indices)
    n_unprotected = len(unprotected_indices)
    n = n_protected + n_unprotected

    pos_ratio_protected = np.sum(_safe_indexing(y_true, protected_indices)) / n_protected
    pos_ratio_unprotected = np.sum(_safe_indexing(y_true, unprotected_indices)) / n_unprotected

    discrimination = pos_ratio_unprotected - pos_ratio_protected

    # number of pairs to swap label
    return abs(round((discrimination * n_protected * n_unprotected) / n))


class BiasRelabler(OneToOneFeatureMixin, BaseEstimator):
    """
    Sampler which relabels instances to remove bias against a subgroup

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used to determine the number of promotion and demotion pairs.
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the group weights are computed.
        Group weights determine how many instances to relabel for each combination of target and protected attribute.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of protected attribute.

        - ``Callable``: function which computes the group weights based on the target and protected attribute. \
        Callable accepts two arguments: y_true and protected_attr and returns the group weights. \
        Group weights are a 2x2 matrix where the rows represent the target variable and the columns represent the \
        protected attribute. \
        The element at position (i, j) is the weight for the pair (y_true == i, protected_attr == j).
    transform_attr : Optional[Callable], default=None
        Function which transforms protected attribute before resampling the training data.
    """
    _estimator_type = "sampler"

    strategy_mapping: dict[str, StrategyFn] = {
        'statistical parity': _independent_pairs,
        'demographic parity': _independent_pairs,
    }

    def __init__(
            self,
            estimator,
            *,
            strategy: Union[Callable, Strategy] = 'statistical parity',
            transform_attr: Optional[Callable] = None,
    ):
        self.estimator = clone(estimator)
        self.transform_attr = transform_attr
        if isinstance(strategy, str):
            strategy = self.strategy_mapping[strategy]
        self.strategy = strategy

    def fit_resample(
            self,
            X: _XT,
            y: ArrayLike,
            *,
            protected_attr: Optional[ArrayLike] = None
    ) -> tuple[_XT, np.ndarray]:
        """
        Fit the estimator and relabel the data according to the strategy.

        Parameters
        ----------
        X : ArrayLike
            Training data.
        y : ArrayLike
            Target values.
        protected_attr : Optional[ArrayLike]
            Protected attribute used to determine the number of promotion and demotion pairs.

        Returns
        -------
        X : ArrayLike
            Training data.
        y : np.ndarray
            Relabeled target values.
        """
        if self.transform_attr is not None:
            protected_attr = self.transform_attr(protected_attr)

        self.estimator.fit(X, y)
        y_pred = self.estimator.predict_proba(X)[:, 1]

        n_pairs = self.strategy(y, protected_attr)
        if n_pairs <= 0:
            return X, np.asarray(y)

        protected_indices = np.where(protected_attr == 0)[0]
        unprotected_indices = np.where(protected_attr == 1)[0]
        probas_unprotected = y_pred[unprotected_indices]
        probas_protected = y_pred[protected_indices]

        demotion_candidates = _get_demotion_candidates(
            probas_unprotected,
            _safe_indexing(y, unprotected_indices),
            n_pairs
        )
        promotion_candidates = _get_promotion_candidates(
            probas_protected,
            _safe_indexing(y, protected_indices),
            n_pairs
        )

        # map promotion and demotion candidates to original indices
        indices = np.arange(len(y))
        demotion_candidates = indices[unprotected_indices][demotion_candidates]
        promotion_candidates = indices[protected_indices][promotion_candidates]

        # relabel the data
        relabled_y = np.asarray(y).copy()
        relabled_y[demotion_candidates] = 0
        relabled_y[promotion_candidates] = 1

        return X, relabled_y


def _get_demotion_candidates(y_pred, y_true, n_pairs):
    """Returns the n_pairs instances with the lowest probability of being positive class label"""
    positive_indices = np.where(y_true == 1)[0]
    positive_predictions = y_pred[positive_indices]
    return positive_indices[np.argsort(positive_predictions)[:n_pairs]]


def _get_promotion_candidates(y_pred, y_true, n_pairs):
    """Returns the n_pairs instances with the lowest probability of being negative class label"""
    negative_indices = np.where(y_true == 0)[0]
    negative_predictions = y_pred[negative_indices]
    return negative_indices[np.argsort(negative_predictions)[-n_pairs:]]
