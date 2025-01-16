from abc import ABC, abstractmethod

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, _fit_context
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


class BaseBoostClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator, ABC):
    """
    Base class for cost-sensitive boosting classifiers.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        The base estimator to be fit with desired hyperparameters.
        If not provided, a default base estimator is used.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : BaseEstimator
        Fitted base estimator.

    Notes
    -----
    The base class for cost-sensitive boosting classifiers is designed to be extended by subclasses.
    The subclasses should implement the `fit` method to fit the base estimator with instance-specific costs.
    """

    _parameter_constraints: dict = {
        'estimator': [HasMethods(['fit', 'predict_proba']), None],
    }

    def __init__(self, estimator=None):
        self.estimator = estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, **fit_params):
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(f'Only binary classification is supported. The type of the target is {y_type}.')
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        y = np.where(y == self.classes_[1], 1, 0)

        return self._fit(X, y, **fit_params)

    @abstractmethod
    def _fit(self, X, y, **fit_params): ...

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_features)

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if LGBMClassifier is not None and isinstance(self.estimator_, LGBMClassifier):
            y_proba = self.estimator_.predict_proba(X, raw_score=True)
            y_proba = expit(y_proba)
            return np.column_stack([1 - y_proba, y_proba])

        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_features)

        Returns
        -------
        y_pred : 1D numpy.ndarray, shape=(n_samples,)
            Predicted class labels.
        """
        y_pred = self.predict_proba(X)
        return self.classes_[np.argmax(y_pred, axis=1)]
