from itertools import product
from typing import Union, Callable, Optional, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import OneToOneFeatureMixin, BaseEstimator
from sklearn.utils import check_random_state, _safe_indexing

from ._strategies import _independent_weights, Strategy

_XT = TypeVar('_XT', bound=ArrayLike)
_YT = TypeVar('_YT', bound=ArrayLike)


class BiasResampler(OneToOneFeatureMixin, BaseEstimator):
    """
    Sampler which resamples instances to remove bias against a subgroup.

    Parameters
    ----------
    strategy : Literal or Callable, default = 'statistical parity'
        Function which computes the group weights based on the target and protected attribute.
        if ``Literal`` group weights are computed so:
            - `'statistical_parity'` or `'demographic parity'`: probability of positive predictions
            are equal between subgroups of protected attribute.
            - other strategies coming in future versions.
    transform_attr : Optional[Callable], default = None
        Function which transforms protected attribute before computing sample weights.
    """
    _estimator_type = "sampler"

    strategy_mapping = {
        'statistical parity': _independent_weights,
        'demographic parity': _independent_weights,
    }

    def __init__(
            self,
            *,
            strategy: Union[Callable, Strategy] = 'statistical parity',
            transform_attr: Optional[Callable] = None,
            random_state=None
    ):
        if isinstance(strategy, str):
            strategy = self.strategy_mapping[strategy]
        self.strategy = strategy
        self.transform_attr = transform_attr
        self.random_state = check_random_state(random_state)
        self.sample_indices_ = None

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
        X : ArrayLike
            Training data.
        y : ArrayLike
            Target values.
        protected_attr : Optional[ArrayLike]
            Protected attribute used to determine the number of promotion and demotion pairs.
        Returns
        -------
        X : ArrayLike
            Resampled training data.
        y : np.ndarray
            Resampled target values.
        """
        if protected_attr is None:
            return X, y

        if self.transform_attr is not None:
            protected_attr = self.transform_attr(protected_attr)

        class_weights = self.strategy(y, protected_attr)
        # if class_weights are all 1, no resampling is needed
        if np.allclose(class_weights, np.ones(class_weights.shape)):
            return X, y
        indices = np.empty((0,), dtype=int)

        # determine the number of samples to be drawn for each class and protected attribute value
        for target_class, protected_val in product(np.unique(y), np.unique(protected_attr)):
            protected_val = int(protected_val)
            idx_class = np.flatnonzero(y == target_class)
            idx_protected_attr = np.flatnonzero(protected_attr == protected_val)
            idx_class_prot = np.intersect1d(idx_class, idx_protected_attr)
            n_samples = int(class_weights[target_class, protected_val] * len(idx_class_prot))
            if n_samples > len(idx_class_prot):  # oversampling
                indices = np.concatenate((indices, idx_class_prot))
                indices = np.concatenate((
                    indices,
                    self.random_state.choice(idx_class_prot, n_samples - len(idx_class_prot), replace=True)
                ))
            else:  # undersampling
                indices = np.concatenate((
                    indices,
                    self.random_state.choice(idx_class_prot, n_samples, replace=False)
                ))

        self.sample_indices_ = indices

        return _safe_indexing(X, indices), _safe_indexing(y, indices)
