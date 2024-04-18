from typing import Callable, Union, Optional
from itertools import product

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator, clone

from ..samplers._strategies import _independent_weights, Strategy, StrategyFn
from ._wrapper import WrapperMixin


def _to_sample_weights(group_weights: np.ndarray, y_true: np.ndarray, protected_attr: np.ndarray):
    """Convert group weights to sample weights."""
    sample_weight = np.empty(len(y_true))
    for target_class, protected_val in product(np.unique(y_true), np.unique(protected_attr)):
        protected_val = int(protected_val)
        idx_class = np.flatnonzero(y_true == target_class)
        idx_prot_attr = np.flatnonzero(protected_attr == protected_val)
        idx_class_prot = np.intersect1d(idx_class, idx_prot_attr)
        sample_weight[idx_class_prot] = group_weights[target_class, protected_val]
    return sample_weight / np.max(sample_weight)


def _independent_sample_weights(y_true: np.ndarray, protected_attr: np.ndarray) -> np.ndarray:
    group_weights = _independent_weights(y_true, protected_attr)
    return _to_sample_weights(group_weights, y_true, protected_attr)


class BiasReweighingClassifier(BaseEstimator, ClassifierMixin, WrapperMixin):
    """
    Classifier which reweighs instances during training to remove bias against a subgroup.

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used for fitting and predicting.
        Base estimator must accept `sample_weight` as an argument in its `fit` method.
    strategy : Literal or Callable, default = 'statistical parity'
        Function which computes the sample weights based on the target and protected attribute.
        .. note::
            Sample weights should be normalized to fall between 0 and 1.
        if ``Literal`` sample weights are computed so:
            - `'statistical_parity'` or `'demographic parity'`: probability of positive predictions
            are equal between subgroups of protected attribute.
            - other strategies coming in future versions.
    transform_attr : Optional[Callable], default = None
        Function which transforms protected attribute before computing sample weights.
    """

    strategy_mapping: dict[str, StrategyFn] = {
        'statistical parity': _independent_sample_weights,
        'demographic parity': _independent_sample_weights,
    }

    def __init__(
            self,
            estimator,
            *,
            strategy: Union[StrategyFn, Strategy] = 'statistical parity',
            transform_attr: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        self.estimator = estimator
        self.strategy = strategy
        self.transform_attr = transform_attr

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            protected_attr: Optional[ArrayLike] = None,
            **fit_params
    ) -> 'BiasReweighingClassifier':
        """
        Fit the estimator and reweigh the instances according to the strategy.
        Parameters
        ----------
        X : ArrayLike
            Training data.
        y : ArrayLike
            Target values.
        protected_attr : Optional[ArrayLike]
            Protected attribute used to determine the sample weights.
        fit_params : dict
            Additional parameters passed to the estimator's `fit` method.

        Returns
        -------
        self : BiasReweighingClassifier
        """
        X, y = np.asarray(X), np.asarray(y)
        if protected_attr is None:
            self.estimator.fit(X, y, **fit_params)
            return self
        protected_attr = np.asarray(protected_attr)

        if isinstance(self.strategy, str):
            strategy_fn = self.strategy_mapping[self.strategy]
        else:
            strategy_fn = self.strategy

        if self.transform_attr is not None:
            protected_attr = self.transform_attr(protected_attr)

        sample_weights = strategy_fn(y, protected_attr)
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, y, sample_weight=sample_weights, **fit_params)

        return self

