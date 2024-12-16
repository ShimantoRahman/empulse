import warnings
from typing import Union, Callable, Optional, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import OneToOneFeatureMixin, BaseEstimator, clone
from sklearn.utils import _safe_indexing
from sklearn.utils._metadata_requests import MetadataRequest, RequestMethod
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

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

    # no swapping needed if one of the groups is empty
    if n_protected == 0 or n_unprotected == 0:
        warnings.warn("protected_attribute only contains one class, no relabeling is performed.", UserWarning)
        return 0

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


    Attributes
    ----------
    estimator_ : Estimator instance
        Fitted estimator.
    """
    _estimator_type = "sampler"
    __metadata_request__fit_resample = {'protected_attr': True}

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
        self.estimator = estimator
        self.transform_attr = transform_attr
        self.strategy = strategy


    def _get_metadata_request(self) -> MetadataRequest:
        """
        Get requested data properties.

        Returns
        -------
        request : MetadataRequest
            A :class:`sklearn:sklearn.utils.metadata_routing.MetadataRequest` instance.
        """
        routing = MetadataRequest(owner=self.__class__.__name__)
        routing.fit_resample.add_request(param='protected_attr', alias=True)
        return routing

    set_fit_resample_request = RequestMethod('fit_resample', ['protected_attr'])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        tags.requires_fit = False
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike) -> 'BiasRelabler':
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)

        Returns
        -------
        self : BiasRelabler
        """
        X, y = validate_data(self, X, y)
        return self

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
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        protected_attr : 1D array-like, shape=(n_samples,)
            Protected attribute used to determine the number of promotion and demotion pairs.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Original training data.
        y : np.ndarray
            Relabeled target values.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                'Only binary classification is supported. The type of the target '
                f'is {y_type}.'
            )

        if self.transform_attr is not None:
            protected_attr = self.transform_attr(protected_attr)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        y_pred = self.estimator_.predict_proba(X)[:, 1]

        if isinstance(self.strategy, str):
            strategy = self.strategy_mapping[self.strategy]
        else:
            strategy = self.strategy

        n_pairs = strategy(y, protected_attr)
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
