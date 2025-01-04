from typing import Literal

import numpy as np
import scipy.stats as st
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone, check_is_fitted
from sklearn.linear_model import HuberRegressor
from sklearn.utils._available_if import available_if
from sklearn.utils.validation import _estimator_has

from ._cs_mixin import CostSensitiveMixin
from ..._common import Parameter

CostStr = Literal['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost']


class RobustCSClassifier(ClassifierMixin, MetaEstimatorMixin, CostSensitiveMixin, BaseEstimator):
    """
    Classifier that fits a cost-sensitive classifier with costs adjusted for outliers.

    The costs are adjusted by fitting an outlier estimator to the costs and imputing the costs for the outliers.
    Outliers are detected by the standardized residuals of the cost and the predicted cost.
    The costs passed to the cost-sensitive classifier are a combination of the original costs (not non-outliers) and
    the imputed predicted costs (for outliers).

    Parameters
    ----------
    estimator : Estimator
        The cost-sensitive classifier to fit.
        The estimator must take tp_cost, tn_cost, fn_cost, and fp_cost as keyword arguments in its fit method.

    outlier_estimator : Estimator, optional
        The outlier estimator to fit to the costs.

        If not provided, a :class:`sklearn:sklearn.linear_model.HuberRegressor` is used with default settings.
    outlier_threshold : float, default=2.5
        The threshold for the standardized residuals to detect outliers.
        If the absolute value of the standardized residual is greater than the threshold,
        the cost is an outlier and will be imputed with the predicted cost.

    detect_outliers_for : {'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', list}, default='all'
        The costs for which to detect outliers.
        By default, all instance-dependent costs are used for outlier detection.
        If a single cost is passed, only that cost is used for outlier detection.
        If a list of costs is passed, only those costs are used for outlier detection.
        tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.


    Attributes
    ----------
    estimator_ : Estimator
        The fitted cost-sensitive classifier.
    outlier_estimators_ : dict{str, Estimator or None}
        The fitted outlier estimators.
        If no outliers are detected for this cost, the value is None.
        The keys of the directory are 'tp_cost', 'tn_cost', 'fn_cost', and 'fp_cost'.
    costs_ : dict
        The imputed costs for the cost-sensitive classifier.

    Notes
    -----
    Constant costs are not used for outlier detection and imputation.

    Code adapted from [1]_.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.models import CSLogitClassifier, RobustCSClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()
        fn_cost = np.random.rand(y.size)  # instance-dependent cost
        fp_cost = 5  # constant cost

        model = RobustCSClassifier(CSLogitClassifier(C=0.1))
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

    Example with passing instance-dependent costs through cross-validation:

    .. code-block:: python

        import numpy as np
        from empulse.models import CSBoostClassifier, RobustCSClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        set_config(enable_metadata_routing=True)

        X, y = make_classification()
        fn_cost = np.random.rand(y.size)
        fp_cost = 5

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RobustCSClassifier(
                CSBoostClassifier()
            ).set_fit_request(fn_cost=True, fp_cost=True))
        ])

        cross_val_score(pipeline, X, y, params={'fn_cost': fn_cost, 'fp_cost': fp_cost})

    Example with passing instance-dependent costs through a grid search:

    .. code-block:: python

        import numpy as np
        from empulse.metrics import expected_cost_loss
        from empulse.models import CSLogitClassifier, RobustCSClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        set_config(enable_metadata_routing=True)

        X, y = make_classification(n_samples=50)
        fn_cost = np.random.rand(y.size)
        fp_cost = 5

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RobustCSClassifier(
                CSLogitClassifier()
            ).set_fit_request(fn_cost=True, fp_cost=True))
        ])
        param_grid = {'model__estimator__C': np.logspace(-5, 2, 5)}
        scorer = make_scorer(
            expected_cost_loss,
            response_method='predict_proba',
            greater_is_better=False,
            normalize=True
        )
        scorer = scorer.set_score_request(fn_cost=True, fp_cost=True)

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer)
        grid_search.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

    References
    ----------
    .. [1] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
           Robust instance-dependent cost-sensitive classification.
           Advances in Data Analysis and Classification, 17(4), 1057-1079.
    """

    def __init__(
            self,
            estimator,
            outlier_estimator=None,
            *,
            outlier_threshold: float = 2.5,
            detect_outliers_for: Literal['all'] | CostStr | list[CostStr] = 'all',
            tp_cost: ArrayLike | float = 0.0,
            tn_cost: ArrayLike | float = 0.0,
            fn_cost: ArrayLike | float = 0.0,
            fp_cost: ArrayLike | float = 0.0,
    ):
        super().__init__()
        self.estimator = estimator
        self.outlier_estimator = outlier_estimator
        self.outlier_threshold = outlier_threshold
        self.detect_outliers_for = detect_outliers_for
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            tp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            tn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            fn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            fp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            **fit_params
    ) -> 'RobustCSLogitClassifier':
        """
        Fit the estimator with the adjusted costs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

        tp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of true positives. If ``float``, then all true positives have the same cost.
            If array-like, then it is the cost of each true positive classification.

        fp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        tn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of true negatives. If ``float``, then all true negatives have the same cost.
            If array-like, then it is the cost of each true negative classification.

        fn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        fit_params : dict
            Additional keyword arguments to pass to the estimator's fit method.

        Returns
        -------
        self : RobustCSLogitClassifier
            Fitted RobustCSLogitClassifier model.
        """

        tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost
        )

        self.costs_ = {
            'tp_cost': tp_cost if isinstance(tp_cost, (int, float)) else np.array(tp_cost),  # take copy of the array
            'tn_cost': tn_cost if isinstance(tn_cost, (int, float)) else np.array(tn_cost),
            'fn_cost': fn_cost if isinstance(fn_cost, (int, float)) else np.array(fn_cost),
            'fp_cost': fp_cost if isinstance(fp_cost, (int, float)) else np.array(fp_cost)
        }
        # only fit on the costs that are arrays and have a standard deviation greater than 0
        should_fit = [cost_name for cost_name, cost in self.costs_.items() if
                      isinstance(cost, np.ndarray) and np.std(cost) > 0]

        if self.detect_outliers_for != 'all':
            if isinstance(self.detect_outliers_for, str):
                if self.detect_outliers_for in self.costs_:  # single cost
                    if self.detect_outliers_for not in should_fit:
                        raise ValueError(
                            f"Cost '{self.detect_outliers_for}' is not an array or has a standard deviation of 0."
                            " Cannot detect outliers for this cost."
                        )
                    should_fit = [self.detect_outliers_for]
                else:
                    raise ValueError(
                        f"Invalid cost name '{self.detect_outliers_for}' in detect_outliers_for."
                        " Must be one of 'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', or a list of these."
                    )
            elif isinstance(self.detect_outliers_for, list):
                for cost_name in self.detect_outliers_for:
                    if cost_name not in self.costs_:
                        raise ValueError(f"Invalid cost name '{cost_name}' in detect_outliers_for.")
                    if cost_name not in should_fit:
                        raise ValueError(
                            f"Cost '{cost_name}' is not an array or has a standard deviation of 0."
                            " Cannot detect outliers for this cost."
                        )
                should_fit = [cost_name for cost_name in self.detect_outliers_for if cost_name in should_fit]
            else:
                raise TypeError(
                    f"Invalid type '{type(self.detect_outliers_for)}' for detect_outliers_for."
                    " Must be one of 'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', or a list of these."
                )

        self.outlier_estimators_ = {}

        for cost_name in self.costs_:
            if cost_name in should_fit:
                target = self.costs_[cost_name]
                if cost_name in ['tp_cost', 'fn_cost']:
                    X_relevant, target_relevant = X[y > 0], target[y > 0]
                else:
                    X_relevant, target_relevant = X[y == 0], target[y == 0]

                if X_relevant.size > 0:
                    outlier_estimator = clone(
                        self.outlier_estimator if self.outlier_estimator is not None else HuberRegressor()
                    ).fit(X_relevant, target_relevant)
                    cost_predictions = outlier_estimator.predict(X)
                    residuals = np.abs(target - cost_predictions)
                    std_residuals = residuals / st.sem(target)
                    outliers = std_residuals > self.outlier_threshold
                    self.costs_[cost_name] = np.where(outliers, cost_predictions, target)
                    self.outlier_estimators_[cost_name] = outlier_estimator
                else:
                    self.outlier_estimators_[cost_name] = None
            else:
                self.outlier_estimators_[cost_name] = None

        # with the imputed costs fit the estimator
        self.estimator_ = clone(self.estimator).fit(X, y, **self.costs_, **fit_params)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)

    @property
    def classes_(self):
        return self.estimator_.classes_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags
