from typing import Union

import numpy as np
import scipy.stats as st
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone, check_is_fitted
from sklearn.linear_model import HuberRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils._available_if import available_if
from sklearn.utils.validation import _estimator_has


class RobustCSClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Robust Cost-Sensitive Classifier

    This classifier fits a cost-sensitive classifier with costs adjusted for outliers.
    The costs are adjusted by fitting an outlier estimator to the costs and imputing the costs for the outliers.
    Outliers are detected by the standardized residuals of the cost and the predicted cost.
    The costs passed to the cost-sensitive classifier are a combination of the original costs (not non-outliers) and
    the imputed predicted costs (for outliers).

    Only the costs that are arrays and have a standard deviation greater than 0 are used for outlier detection.
    Hence, constant costs or costs with no variation are not imputed.

    Parameters
    ----------
    estimator : BaseEstimator
        The cost-sensitive classifier to fit.
        The estimator must take tp_cost, tn_cost, fn_cost, and fp_cost as keyword arguments in its fit method.
    outlier_estimator : BaseEstimator, optional
        The outlier estimator to fit to the costs.
        If not provided, a HuberRegressor is used with default settings.
    outlier_threshold : float, default=2.5
        The threshold for the standardized residuals to detect outliers.
        If the absolute value of the standardized residual is greater than the threshold,
        the cost is an outlier and will be imputed with the predicted cost.

    Attributes
    ----------
    estimator_ : BaseEstimator
        The fitted cost-sensitive classifier.
    outlier_estimator_ : BaseEstimator
        The fitted outlier estimator. If multiple costs are passed, this is a MultiOutputRegressor.
    costs_ : dict
        The imputed costs for the cost-sensitive classifier.
    n_treatments_ : int
        The number of costs which have been imputed.

    References
    ----------
    .. [1] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
           Robust instance-dependent cost-sensitive classification.
           Advances in Data Analysis and Classification, 17(4), 1057-1079.
    """
    def __init__(self, estimator, outlier_estimator=None, outlier_threshold: float = 2.5):
        super().__init__()
        self.estimator = estimator
        self.outlier_estimator = outlier_estimator
        self.outlier_threshold = outlier_threshold

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            tp_cost: Union[ArrayLike, float] = 0.0,
            tn_cost: Union[ArrayLike, float] = 0.0,
            fn_cost: Union[ArrayLike, float] = 0.0,
            fp_cost: Union[ArrayLike, float] = 0.0,
            **fit_params
    ) -> 'RobustCSLogitClassifier':
        """
        Fit the estimator with the adjusted costs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

        tp_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of true positives. If ``float``, then all true positives have the same cost.
            If array-like, then it is the cost of each true positive classification.

        fp_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        tn_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of true negatives. If ``float``, then all true negatives have the same cost.
            If array-like, then it is the cost of each true negative classification.

        fn_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        fit_params : dict
            Additional keyword arguments to pass to the estimator's fit method.

        Returns
        -------
        self

        """

        self.costs_ = {
            'tp_cost': tp_cost if isinstance(tp_cost, (int, float)) else np.array(tp_cost),  # take copy of the array
            'tn_cost': tn_cost if isinstance(tn_cost, (int, float)) else np.array(tn_cost),
            'fn_cost': fn_cost if isinstance(fn_cost, (int, float)) else np.array(fn_cost),
            'fp_cost': fp_cost if isinstance(fp_cost, (int, float)) else np.array(fp_cost)
        }
        # only fit on the costs that are arrays and have a standard deviation greater than 0
        should_fit = [cost_name for cost_name, cost in self.costs_.items() if
                      isinstance(cost, np.ndarray) and np.std(cost) > 0]
        self.n_treatments_ = len(should_fit)
        if self.n_treatments_ == 0:
            pass  # no outlier detection needed
        elif self.n_treatments_ == 1:
            target = self.costs_[should_fit[0]]
            self.outlier_estimator_ = clone(
                self.outlier_estimator if self.outlier_threshold is not None else HuberRegressor()
            ).fit(X, target)
            cost_predictions = self.outlier_estimator_.predict(X)
            # outliers if the absolute value of the standardized residuals is greater than the threshold
            residuals = np.abs(target - cost_predictions)
            std_residuals = residuals / st.sem(target)
            # TODO: check if this is correct
            # std_residuals = residuals / residuals.std()
            outliers = std_residuals > self.outlier_threshold
            # for the outliers impute the cost with the predicted cost
            self.costs_[should_fit[0]] = np.where(outliers, cost_predictions, target)
        else:
            targets = np.column_stack([self.costs_[cost_name] for cost_name in should_fit])
            self.outlier_estimator_ = MultiOutputRegressor(
                self.outlier_estimator if self.outlier_threshold is not None else HuberRegressor()
            ).fit(X, targets)
            cost_predictions = self.outlier_estimator_.predict(X)
            residuals = np.abs(targets - cost_predictions)
            std_residuals = residuals / st.sem(targets, axis=0)
            # TODO: check if this is correct
            # std_residuals = residuals / residuals.std(axis=0)
            outliers = std_residuals > self.outlier_threshold
            for i, cost_name in enumerate(should_fit):
                self.costs_[cost_name] = np.where(outliers[:, i], cost_predictions[:, i], self.costs_[cost_name])

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

    @available_if(_estimator_has("score"))
    def score(self, X: ArrayLike, y: ArrayLike, **kwargs) -> float:
        check_is_fitted(self, "estimator_")
        return self.estimator_.score(X, y, **kwargs)

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
