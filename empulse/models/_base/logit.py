from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class BaseLogitClassifier(ABC, ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            C: float = 1.0,
            fit_intercept: bool = True,
            soft_threshold: bool = True,
            l1_ratio: float = 1.0,
            loss: Union[str | Callable] = None,
            optimize_fn: Optional[Callable] = None,
            optimizer_params: Optional[dict[str, Any]] = None,
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.soft_threshold = soft_threshold
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.optimizer_params = optimizer_params
        self.optimize_fn = optimize_fn

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X, y, **fit_params):
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
        y = np.where(y == self.classes_[1], 1, 0)

        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return self._fit(X, y, **fit_params)

    @abstractmethod
    def _fit(self, X, y, **fit_params):
        ...

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, 2)
            Predicted probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = self.result_.x
        logits = np.dot(X, theta)
        y_pred = expit(logits)
        # create 2D array with complementary probabilities
        y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predicted labels.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
            Features.

        Returns
        -------
        y_pred : 1D numpy.ndarray, shape=(n_samples,)
            Predicted labels.
        """
        y_pred = self.predict_proba(X)
        return self.classes_[np.argmax(y_pred, axis=1)]
