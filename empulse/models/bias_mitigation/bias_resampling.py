from typing import Callable, Union, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted

from ...samplers import BiasResampler
from ...samplers._strategies import Strategy, StrategyFn


class BiasResamplingClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier which resamples instances during training to remove bias against a subgroup.

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used for fitting and predicting.
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

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : Estimator instance
        Fitted base estimator.

    Examples
    --------
    1. Using the `BiasResamplingClassifier` with a logistic regression model:

    .. code-block:: python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasResamplingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        model = BiasResamplingClassifier(estimator=LogisticRegression())
        model.fit(X, y, protected_attr=high_clv)

    2. Converting a continuous attribute to a binary attribute:

    .. code-block:: python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasResamplingClassifier

        X, y = make_classification()
        clv = np.random.rand(X.shape[0]) * 100

        model = BiasResamplingClassifier(
            estimator=LogisticRegression(),
            transform_attr=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
        )
        model.fit(X, y, protected_attr=clv)

    3. Using a custom strategy function:

    .. code-block:: python

        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasResamplingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        # Simple strategy to double the weight for the protected attribute
        def strategy(y_true, protected_attr):
            return np.array([
                [1, 2],
                [1, 2]
            ])

        model = BiasResamplingClassifier(
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
        from empulse.models import BiasResamplingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasResamplingClassifier(LogisticRegression()))
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
        from empulse.models import BiasResamplingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasResamplingClassifier(LogisticRegression()).set_fit_request(protected_attr=True))
            ])
            search = GridSearchCV(pipeline, param_grid)
            search.fit(X, y, protected_attr=high_clv)
    """

    __metadata_request__fit = {'protected_attr': True}

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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            protected_attr: Optional[ArrayLike] = None,
            **fit_params
    ) -> 'BiasResamplingClassifier':
        """
        Fit the estimator and resample the instances according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
            Training data.
        y : 1D array-like, shape=(n_samples,)
            Target values.
        protected_attr : 1D array-like, shape=(n_samples,), default = None
            Protected attribute used to determine the group weights.
        fit_params : dict
            Additional parameters passed to the estimator's `fit` method.

        Returns
        -------
        self : BiasResamplingClassifier
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                'Only binary classification is supported. The type of the target '
                f'is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        if protected_attr is None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)
            return self
        protected_attr = np.asarray(protected_attr)

        sampler = BiasResampler(strategy=self.strategy, transform_feature=self.transform_attr)
        X, y = sampler.fit_resample(X, y, sensitive_feature=protected_attr)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)

        Returns
        -------
        y_pred : 1D numpy.ndarray, shape=(n_samples,)
            Predicted class labels.
        """
        y_pred = self.predict_proba(X)
        return self.classes_[np.argmax(y_pred, axis=1)]
