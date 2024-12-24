import warnings
from itertools import product
from typing import Union, Callable, Optional, TypeVar, TYPE_CHECKING

import numpy as np
from imblearn.base import BaseSampler
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.estimator_checks import ClassifierTags
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

from ._strategies import _independent_weights, Strategy, StrategyFn

_XT = TypeVar('_XT', bound=ArrayLike)
_YT = TypeVar('_YT', bound=ArrayLike)


class BiasResampler(BaseSampler):
    """
    Sampler which resamples instances to remove bias against a subgroup

    Parameters
    ----------
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the group weights are computed.
        Group weights determine how much to over or undersample each combination of target and sensitive feature.
        For example, a weight of 2 for the pair (y_true == 1, sensitive_feature == 0) means that the resampled dataset
        should have twice as many instances with y_true == 1 and sensitive_feature == 0 compared to the original dataset.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of sensitive feature.

        - ``Callable``: function which computes the group weights based on the target and sensitive feature. \
        Callable accepts two arguments: y_true and sensitive_feature and returns the group weights. \
        Group weights are a 2x2 matrix where the rows represent the target variable and the columns represent the \
        sensitive feature. \
        The element at position (i, j) is the weight for the pair (y_true == i, sensitive_feature == j).
    transform_feature : Optional[Callable], default=None
        Function which transforms sensitive_feature before resampling the training data.
    random_state : int or :class:`numpy:numpy.random.RandomState`, optional
        Random number generator seed for reproducibility.

    Attributes
    ----------
    sample_indices_ : ndarray
        Indices of the samples that were selected.
    """
    _estimator_type = "sampler"
    _sampling_type = 'bypass'
    _parameter_constraints = {
        'strategy': [StrOptions({'statistical parity', 'demographic parity'}), callable],
        'transform_attr': [callable, None],
        'random_state': ['random_state'],
    }
    _strategy_mapping: dict[str, StrategyFn] = {
        'statistical parity': _independent_weights,
        'demographic parity': _independent_weights,
    }

    if TYPE_CHECKING:
        # BaseEstimator should dynamically generate the method signature at runtime
        def set_fit_resample_request(self, sensitive_feature=False): pass

    def __init__(
            self,
            *,
            strategy: Union[Callable, Strategy] = 'statistical parity',
            transform_feature: Optional[Callable] = None,
            random_state: Optional[Union[RandomState, int]] = None
    ):
        super().__init__()
        self.strategy = strategy
        self.transform_feature = transform_feature
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        return tags

    def fit_resample(
            self,
            X: _XT,
            y: _YT,
            *,
            sensitive_feature: Optional[ArrayLike] = None,
    ) -> tuple[_XT, _YT]:
        """
        Resample the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive attribute used to determine which instances to resample.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Resampled training data.
        y : 1D array-like, shape=(n_samples,)
            Resampled target values.
        """
        return super().fit_resample(X, y, sensitive_feature=sensitive_feature)

    def _fit_resample(
            self,
            X: _XT,
            y: _YT,
            *,
            sensitive_feature: Optional[ArrayLike] = None,
    ) -> tuple[_XT, _YT]:
        """
        Resample the data according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,)
            Sensitive attribute used to determine which instances to resample.

        Returns
        -------
        X : 2D array-like, shape=(n_samples, n_features)
            Resampled training data.
        y : 1D array-like, shape=(n_samples,)
            Resampled target values.
        """
        X, y = validate_data(self, X, y, accept_sparse=['csr', 'csc', 'coo'])
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                'Only binary classification is supported. The type of the target '
                f'is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            return X, y
        y_binarized = np.where(y == self.classes_[1], 1, 0)

        random_state = check_random_state(self.random_state)
        if sensitive_feature is None:
            return X, y

        if self.transform_feature is not None:
            sensitive_feature = self.transform_feature(sensitive_feature)

        if isinstance(self.strategy, str):
            strategy = self._strategy_mapping[self.strategy]
        else:
            strategy = self.strategy
        class_weights = strategy(y_binarized, sensitive_feature)
        # if class_weights are all 1, no resampling is needed
        if np.allclose(class_weights, np.ones(class_weights.shape)):
            return X, y
        indices = np.empty((0,), dtype=int)

        unique_attr = np.unique(sensitive_feature)
        if len(unique_attr) == 1:
            warnings.warn(
                "sensitive_feature only contains one class, no resampling is performed.", UserWarning
            )
            return X, y

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
                    random_state.choice(idx_class_sensitive, n_samples - len(idx_class_sensitive), replace=True)
                ))
            else:  # undersampling
                indices = np.concatenate((
                    indices,
                    random_state.choice(idx_class_sensitive, n_samples, replace=False)
                ))

        self.sample_indices_ = indices

        return _safe_indexing(X, indices), _safe_indexing(y, indices)
