from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.utils import Tags
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...samplers._strategies import Strategy

# Each subclass's `strategy` callable has a different, more specific contract (documented on the
# subclass itself): BiasRelabelingClassifier's returns an int (a pair count), BiasResamplingClassifier's
# returns a 2x2 group-weight matrix, and BiasReweighingClassifier's returns a per-sample weight
# array. This shared, deliberately loose type covers all three without narrowing to any one of them.
AnyStrategyFn = Callable[[NDArray[Any], NDArray[Any]], Any]


class BaseBiasMitigationClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Shared fit/predict scaffolding for the fairness meta-estimator classifiers.

    :class:`~empulse.models.BiasRelabelingClassifier`, :class:`~empulse.models.BiasResamplingClassifier`,
    and :class:`~empulse.models.BiasReweighingClassifier` differ only in how they mitigate bias against
    ``sensitive_feature`` once one is supplied. Subclasses implement :meth:`_fit_mitigated`.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'estimator': [HasMethods(['fit', 'predict_proba']), None],
        'strategy': [callable, StrOptions({'statistical parity', 'demographic parity'}), None],
        'transform_feature': [callable, None],
    }

    def __init__(
        self,
        estimator: Any,
        *,
        strategy: AnyStrategyFn | Strategy = 'statistical parity',
        transform_feature: Callable[[NDArray[Any]], IntNDArray] | None = None,
    ):
        self.estimator = estimator
        self.strategy = strategy
        self.transform_feature = transform_feature

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    @abstractmethod
    def _fit_mitigated(self, X: FloatNDArray, y: IntNDArray, sensitive_feature: IntNDArray, **fit_params: Any) -> Any:
        """Fit the base estimator using the subclass's bias-mitigation strategy and return it."""

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(self, X: ArrayLike, y: ArrayLike, *, sensitive_feature: ArrayLike | None = None, **fit_params: Any) -> Self:
        """
        Fit the estimator, mitigating bias against ``sensitive_feature`` if one is provided.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
            Training data.
        y : 1D array-like, shape=(n_samples,)
            Target values.
        sensitive_feature : 1D array-like, shape=(n_samples,), default = None
            Sensitive feature used to mitigate bias. If ``None``, the base estimator is fit as-is.
        fit_params : dict
            Additional parameters passed to the estimator's `fit` method.

        Returns
        -------
        self : BaseBiasMitigationClassifier
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
        if len(sensitive_feature) != len(y):
            raise ValueError(
                f'sensitive_feature must have the same length as y, got {len(sensitive_feature)} and {len(y)}.'
            )

        self.estimator_ = self._fit_mitigated(X, y, sensitive_feature, **fit_params)

        return self

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
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
        y_proba: FloatNDArray = self.estimator_.predict_proba(X)
        return y_proba

    def predict(self, X: FloatArrayLike) -> NDArray[Any]:
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
        y_proba = self.predict_proba(X)
        y_pred: NDArray[Any] = self.classes_[np.argmax(y_proba, axis=1)]
        return y_pred
