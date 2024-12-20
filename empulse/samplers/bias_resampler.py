import warnings
from itertools import product
from typing import Union, Callable, Optional, TypeVar

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import OneToOneFeatureMixin, BaseEstimator
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils._metadata_requests import MetadataRequest, RequestMethod
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

from ._strategies import _independent_weights, Strategy, StrategyFn

_XT = TypeVar('_XT', bound=ArrayLike)
_YT = TypeVar('_YT', bound=ArrayLike)


class BiasResampler(OneToOneFeatureMixin, BaseEstimator):
    """
    Sampler which resamples instances to remove bias against a subgroup

    Parameters
    ----------
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the group weights are computed.
        Group weights determine how much to over or undersample each combination of target and protected attribute.
        For example, a weight of 2 for the pair (y_true == 1, protected_attr == 0) means that the resampled dataset
        should have twice as many instances with y_true == 1 and protected_attr == 0 compared to the original dataset.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of protected attribute.

        - ``Callable``: function which computes the group weights based on the target and protected attribute. \
        Callable accepts two arguments: y_true and protected_attr and returns the group weights. \
        Group weights are a 2x2 matrix where the rows represent the target variable and the columns represent the \
        protected attribute. \
        The element at position (i, j) is the weight for the pair (y_true == i, protected_attr == j).
    transform_attr : Optional[Callable], default=None
        Function which transforms protected attribute before resampling the training data.
    random_state : int or :class:`numpy:numpy.random.RandomState`, optional
        Random number generator seed for reproducibility.

    Attributes
    ----------
    sample_indices_ : ndarray
        Indices of the samples that were selected.
    """
    _estimator_type = "sampler"
    __metadata_request__fit_resample = {'protected_attr': True}

    strategy_mapping: dict[str, StrategyFn] = {
        'statistical parity': _independent_weights,
        'demographic parity': _independent_weights,
    }

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

    def __init__(
            self,
            *,
            strategy: Union[Callable, Strategy] = 'statistical parity',
            transform_attr: Optional[Callable] = None,
            random_state: Optional[Union[RandomState, int]] = None
    ):
        self.strategy = strategy
        self.transform_attr = transform_attr
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        tags.requires_fit = False
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike, protected_attr: Optional[ArrayLike] = None) -> 'BiasResampler':
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)

        Returns
        -------
        self : BiasResampler
        """
        X, y = validate_data(self, X, y)
        return self

    def fit_resample(
            self,
            X: _XT,
            y: _YT,
            *,
            protected_attr: Optional[ArrayLike] = None,
    ) -> tuple[_XT, _YT]:
        """
        Resample the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        protected_attr : 1D array-like, shape=(n_samples,)
            Protected attribute used to determine which instances to resample.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Resampled training data.
        y : 1D array-like, shape=(n_samples,)
            Resampled target values.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                'Only binary classification is supported. The type of the target '
                f'is {y_type}.'
            )
        random_state = check_random_state(self.random_state)
        if protected_attr is None:
            return X, y

        if self.transform_attr is not None:
            protected_attr = self.transform_attr(protected_attr)

        if isinstance(self.strategy, str):
            strategy = self.strategy_mapping[self.strategy]
        else:
            strategy = self.strategy
        class_weights = strategy(y, protected_attr)
        # if class_weights are all 1, no resampling is needed
        if np.allclose(class_weights, np.ones(class_weights.shape)):
            return X, y
        indices = np.empty((0,), dtype=int)

        unique_attr = np.unique(protected_attr)
        if len(unique_attr) == 1:
            warnings.warn(
                "protected_attribute only contains one class, no resampling is performed.", UserWarning
            )
            return X, y

        # determine the number of samples to be drawn for each class and protected attribute value
        for target_class, protected_val in product(np.unique(y), unique_attr):
            protected_val = int(protected_val)
            idx_class = np.flatnonzero(y == target_class)
            idx_protected_attr = np.flatnonzero(protected_attr == protected_val)
            idx_class_prot = np.intersect1d(idx_class, idx_protected_attr)
            n_samples = int(class_weights[target_class, protected_val] * len(idx_class_prot))
            if n_samples > len(idx_class_prot):  # oversampling
                indices = np.concatenate((indices, idx_class_prot))
                indices = np.concatenate((
                    indices,
                    random_state.choice(idx_class_prot, n_samples - len(idx_class_prot), replace=True)
                ))
            else:  # undersampling
                indices = np.concatenate((
                    indices,
                    random_state.choice(idx_class_prot, n_samples, replace=False)
                ))

        self.sample_indices_ = indices

        return _safe_indexing(X, indices), _safe_indexing(y, indices)
