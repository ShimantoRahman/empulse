from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.validation import check_is_fitted

from ...samplers import BiasRelabler
from ...samplers._strategies import Strategy, StrategyFn
from ...utils._sklearn_compat import type_of_target, validate_data


class BiasRelabelingClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier which relabels instances during training to remove bias against a subgroup.

    Read more in the :ref:`User Guide <bias_mitigation>`.

    Parameters
    ----------
    estimator : Estimator instance
        Base estimator which is used for fitting and predicting.
    strategy : {'statistical parity', 'demographic parity'} or Callable, default='statistical parity'
        Determines how the group weights are computed.
        Group weights determine how many instances to relabel for each combination of target and sensitive feature.

        - ``'statistical_parity'`` or ``'demographic parity'``: \
        probability of positive predictions are equal between subgroups of sensitive feature.

        - ``Callable``: function which computes the number of labels swaps based on the target and sensitive feature. \
        Callable accepts two arguments: \
        y_true and sensitive_feature and returns the number of pairs needed to be swapped.
    transform_feature : Optional[Callable], default=None
        Function which transforms sensitive feature before resampling the training data.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : Estimator instance
        Fitted base estimator.

    Examples
    --------
    1. Using the `BiasRelabelingClassifier` with a logistic regression model:

    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasRelabelingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        model = BiasRelabelingClassifier(estimator=LogisticRegression())
        model.fit(X, y, sensitive_feature=high_clv)

    2. Converting a continuous attribute to a binary attribute:

    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasRelabelingClassifier

        X, y = make_classification()
        clv = np.random.rand(X.shape[0]) * 100

        model = BiasRelabelingClassifier(
            estimator=LogisticRegression(),
            transform_feature=lambda clv: (clv > np.quantile(clv, 0.8)).astype(int)
        )
        model.fit(X, y, sensitive_feature=clv)

    3. Using a custom strategy function:

    .. code-block:: python

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        from empulse.models import BiasRelabelingClassifier

        X, y = make_classification()
        high_clv = np.random.randint(0, 2, size=X.shape[0])

        # Simple strategy to swap 2 labels
        def strategy(y_true, sensitive_feature):
            return 2

        model = BiasRelabelingClassifier(
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
        from empulse.models import BiasRelabelingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasRelabelingClassifier(LogisticRegression()).set_fit_request(sensitive_feature=True))
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
        from empulse.models import BiasRelabelingClassifier

        with config_context(enable_metadata_routing=True):
            X, y = make_classification()
            high_clv = np.random.randint(0, 2, size=X.shape[0])

            param_grid = {'model__estimator__C': [0.1, 1, 10]}
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', BiasRelabelingClassifier(LogisticRegression()).set_fit_request(sensitive_feature=True))
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
    ) -> 'BiasRelabelingClassifier':
        """
        Fit the estimator and reweigh the instances according to the strategy.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
            Training data.
        y : 1D array-like, shape=(n_samples,)
            Target values.
        sensitive_feature : 1D array-like, shape=(n_samples,), default = None
            Sensitive feature used to determine the sample group weights.
        fit_params : dict
            Additional parameters passed to the estimator's `fit` method.

        Returns
        -------
        self : BiasRelabelingClassifier
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

        sampler = BiasRelabler(
            estimator=self.estimator, strategy=self.strategy, transform_feature=self.transform_feature
        )
        X, y = sampler.fit_resample(X, y, sensitive_feature=sensitive_feature)
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
