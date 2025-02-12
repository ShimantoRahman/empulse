from collections.abc import Callable
from itertools import product
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.validation import check_is_fitted

from ...samplers._strategies import Strategy, StrategyFn, _independent_weights
from ...utils._sklearn_compat import type_of_target, validate_data


def _to_sample_weights(group_weights: np.ndarray, y_true: np.ndarray, sensitive_feature: np.ndarray) -> np.ndarray:
    """Convert group weights to sample weights."""
    sample_weight = np.empty(len(y_true))
    for target_class, sensitive_val in product(np.unique(y_true), np.unique(sensitive_feature)):
        sensitive_val = int(sensitive_val)
        idx_class = np.flatnonzero(y_true == target_class)
        idx_sensitive_feature = np.flatnonzero(sensitive_feature == sensitive_val)
        idx_class_sensitive = np.intersect1d(idx_class, idx_sensitive_feature)
        sample_weight[idx_class_sensitive] = group_weights[target_class, sensitive_val]
    return sample_weight / np.max(sample_weight)


def _independent_sample_weights(y_true: np.ndarray, sensitive_feature: np.ndarray) -> np.ndarray:
    group_weights = _independent_weights(y_true, sensitive_feature)
    return _to_sample_weights(group_weights, y_true, sensitive_feature)


class BiasReweighingClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier which reweighs instances during training to remove bias against a subgroup.

    Read more in the :ref:`User Guide <bias_mitigation>`.

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used for fitting and predicting.
        Base estimator must accept `sample_weight` as an argument in its `fit` method.
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the sample weights are computed. Sample weights are passed to the estimator's `fit` method.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of sensitive feature.

        - ``Callable``: function which computes the sample weights based on the target and sensitive feature. \
        Callable accepts two arguments: y_true and sensitive_feature and returns the sample weights. \
        Sample weights are a numpy array where each represents the weight given to that respective instance. \
        Sample weights should be normalized to fall between 0 and 1.

    transform_feature : Optional[Callable], default=None
        Function which transforms sensitive feature before computing sample weights.

    Examples
    --------
    1. Using the `BiasReweighingClassifier` with a logistic regression model:

    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasReweighingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        model = BiasReweighingClassifier(estimator=LogisticRegression())
        model.fit(X, y, sensitive_feature=high_clv)

    2. Converting a continuous attribute to a binary attribute:

    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasReweighingClassifier

        X, y = make_classification()
        clv = np.random.rand(X.shape[0]) * 100

        model = BiasReweighingClassifier(
            estimator=LogisticRegression(),
            transform_feature=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
        )
        model.fit(X, y, sensitive_feature=clv)

    3. Using a custom strategy function:

    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasReweighingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        # Simple strategy to double the weight for the sensitive feature
        def strategy(y_true, sensitive_feature):
            sample_weights = np.ones(len(sensitive_feature))
            sample_weights[np.where(sensitive_feature == 0)] = 0.5
            return sample_weights

        model = BiasReweighingClassifier(
            estimator=LogisticRegression(),
            strategy=strategy
        )
        model.fit(X, y, sensitive_feature=high_clv)

    4. Passing the sensitive feature in a cross-validation grid search:

    .. code-block:: python

        import numpy as np
        from sklearn import config_context
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from empulse.models import BiasReweighingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasReweighingClassifier(LogisticRegression()).set_fit_request(sensitive_feature=True))
            ])
            search = GridSearchCV(pipeline, param_grid)
            search.fit(X, y, sensitive_feature=high_clv)

    5. Passing the sensitive feature through metadata routing in a cross-validation grid search:

    .. code-block:: python

        import numpy as np
        from sklearn import config_context
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from empulse.models import BiasReweighingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasReweighingClassifier(LogisticRegression()).set_fit_request(sensitive_feature=True))
            ])
            search = GridSearchCV(pipeline, param_grid)
            search.fit(X, y, sensitive_feature=high_clv)

    References
    ----------

    .. [1] Rahman, S., Janssens, B., & Bogaert, M. (2025).
           Profit-driven pre-processing in B2B customer churn modeling using fairness techniques.
           Journal of Business Research, 189, 115159. doi:10.1016/j.jbusres.2024.115159
    """

    _parameter_constraints: ClassVar[dict[str, list]] = {
        'estimator': [HasMethods(['fit', 'predict_proba']), None],
        'strategy': [callable, StrOptions({'statistical parity', 'demographic parity'}), None],
        'transform_feature': [callable, None],
    }

    strategy_mapping: ClassVar[dict[str, StrategyFn]] = {
        'statistical parity': _independent_sample_weights,
        'demographic parity': _independent_sample_weights,
    }

    def __init__(
        self,
        estimator: Any,
        *,
        strategy: StrategyFn | Strategy = 'statistical parity',
        transform_feature: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.estimator = estimator
        self.strategy = strategy
        self.transform_feature = transform_feature

    def _more_tags(self):
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self, X: ArrayLike, y: ArrayLike, *, sensitive_feature: ArrayLike | None = None, **fit_params: Any
    ) -> 'BiasReweighingClassifier':
        """
        Fit the estimator and reweigh the instances according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        sensitive_feature : 1D array-like, shape=(n_samples,), default = None
            Sensitive attribute used to determine the sample weights.
        fit_params : dict
            Additional parameters passed to the estimator's `fit` method.

        Returns
        -------
        self : BiasReweighingClassifier
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                f'Unknown label type: Only binary classification is supported. The type of the target is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")

        if sensitive_feature is None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)
            return self
        sensitive_feature = np.asarray(sensitive_feature)

        strategy_fn = self.strategy_mapping[self.strategy] if isinstance(self.strategy, str) else self.strategy

        if self.transform_feature is not None:
            sensitive_feature = self.transform_feature(sensitive_feature)

        sample_weights = strategy_fn(y, sensitive_feature)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=sample_weights, **fit_params)

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
