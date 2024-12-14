from typing import Callable, Union, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator, clone

from ..samplers._strategies import Strategy, StrategyFn
from ..samplers import BiasRelabler
from ._wrapper import WrapperMixin


class BiasRelabelingClassifier(ClassifierMixin, WrapperMixin, BaseEstimator):
    """
    Classifier which relabels instances during training to remove bias against a subgroup.

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used for fitting and predicting.
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

    Examples
    --------
    1. Using the `BiasRelabelingClassifier` with a logistic regression model:

    .. code-block:: python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasRelabelingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        model = BiasRelabelingClassifier(estimator=LogisticRegression())
        model.fit(X, y, protected_attr=high_clv)

    2. Converting a continuous attribute to a binary attribute:

    .. code-block:: python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasRelabelingClassifier

        X, y = make_classification()
        clv = np.random.rand(X.shape[0]) * 100

        model = BiasRelabelingClassifier(
            estimator=LogisticRegression(),
            transform_attr=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
        )
        model.fit(X, y, protected_attr=clv)

    3. Using a custom strategy function:

    .. code-block:: python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasRelabelingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        # Simple strategy to double the weight for the protected attribute
        def strategy(y_true, protected_attr):
            return np.array([
                [1, 2],
                [1, 2]
            ])

        model = BiasRelabelingClassifier(
            estimator=LogisticRegression(),
            strategy=strategy
        )
        model.fit(X, y, protected_attr=high_clv)

    4. Passing the protected attribute in a cross-validation grid search:

    .. code-block:: python

        from sklearn import config_context
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from empulse.models import BiasRelabelingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasRelabelingClassifier(LogisticRegression()))
            ])
            search = GridSearchCV(pipeline, param_grid)
            search.fit(X, y, model__protected_attr=high_clv)

    5. Passing the protected attribute through metadata routing in a cross-validation grid search:

    .. code-block:: python

        from sklearn import config_context
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from empulse.models import BiasRelabelingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasRelabelingClassifier(LogisticRegression()).set_fit_request(protected_attr=True))
            ])
            search = GridSearchCV(pipeline, param_grid)
            search.fit(X, y, protected_attr=high_clv)
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
        X : 2D array-like, shape=(n_samples, n_dim)
            Training data.
        y : 1D array-like, shape=(n_samples,)
            Target values.
        protected_attr : 1D array-like, shape=(n_samples,), default = None
            Protected attribute used to determine the sample group weights.
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

