from typing import Any

from sklearn.base import clone

from ..._types import FloatNDArray, IntNDArray
from ...samplers import BiasRelabler
from ._base import BaseBiasMitigationClassifier


class BiasRelabelingClassifier(BaseBiasMitigationClassifier):
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

        - ``'statistical parity'`` or ``'demographic parity'``: \
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

    References
    ----------

    .. [1] Rahman, S., Janssens, B., & Bogaert, M. (2025).
           Profit-driven pre-processing in B2B customer churn modeling using fairness techniques.
           Journal of Business Research, 189, 115159. doi:10.1016/j.jbusres.2024.115159
    """

    def _fit_mitigated(self, X: FloatNDArray, y: IntNDArray, sensitive_feature: IntNDArray, **fit_params: Any) -> Any:
        sampler = BiasRelabler(
            estimator=self.estimator, strategy=self.strategy, transform_feature=self.transform_feature
        )
        X, y = sampler.fit_resample(X, y, sensitive_feature=sensitive_feature)
        estimator_ = clone(self.estimator)
        estimator_.fit(X, y, **fit_params)
        return estimator_
