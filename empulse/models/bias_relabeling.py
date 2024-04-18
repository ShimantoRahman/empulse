from typing import Callable, Union, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator, clone

from ..samplers._strategies import Strategy, StrategyFn
from ..samplers import BiasRelabler
from ._wrapper import WrapperMixin


class BiasRelabelingClassifier(BaseEstimator, ClassifierMixin, WrapperMixin):
    """
    Classifier which relabels instances during training to remove bias against a subgroup.

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used for fitting and predicting.
    strategy : Literal or Callable, default = 'statistical parity'
        Function which computes the group weights based on the target and protected attribute.
        if ``Literal`` group weights are computed so:
            - `'statistical_parity'` or `'demographic parity'`: probability of positive predictions
            are equal between subgroups of protected attribute.
            - other strategies coming in future versions.
    transform_attr : Optional[Callable], default = None
        Function which transforms protected attribute before relabeling the training data.
    """

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
    ) -> 'BiasRelabelingClassifier':
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
        self : BiasRelabelingClassifier
        """
        X, y = np.asarray(X), np.asarray(y)
        if protected_attr is None:
            self.estimator.fit(X, y, **fit_params)
            return self
        protected_attr = np.asarray(protected_attr)

        sampler = BiasRelabler(
            estimator=self.estimator,
            strategy=self.strategy,
            transform_attr=self.transform_attr
        )
        X, y = sampler.fit_resample(X, y, protected_attr=protected_attr)
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, y, **fit_params)

        return self

