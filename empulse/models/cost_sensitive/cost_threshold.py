from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._classification_threshold import BaseThresholdClassifier
from sklearn.utils._metadata_requests import MetadataRouter, MethodMapping, process_routing
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.validation import check_is_fitted


class CSThresholdClassifier(BaseThresholdClassifier):

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict_proba"]), ],
        "calibrator": [
            HasMethods(["fit", "predict_proba"]),
            StrOptions({"sigmoid", "isotonic"}),
            None
        ],
        "pos_label": [Real, str, "boolean", None],
    }

    def __init__(
            self,
            estimator,
            *,
            calibrator: Literal['sigmoid', 'isotonic'] | object | None = 'sigmoid',
            pos_label=None,
    ):
        super().__init__(estimator, response_method='predict_proba')
        self.calibrator = calibrator
        self.pos_label = pos_label

    @property
    def classes_(self):
        if estimator := getattr(self, "estimator_", None):
            return estimator.classes_
        try:
            check_is_fitted(self.estimator)
            return self.estimator.classes_
        except NotFittedError:
            raise AttributeError(
                "The underlying estimator is not fitted yet."
            ) from NotFittedError

    def _get_calibrator(self, estimator):
        if self.calibrator == 'sigmoid':
            return CalibratedClassifierCV(
                estimator,
                method='sigmoid',
                ensemble=False
            )
        elif self.calibrator == 'isotonic':
            return CalibratedClassifierCV(
                estimator,
                method='isotonic',
                ensemble=False
            )
        else:
            return self.calibrator.set_params(estimator=estimator)

    def _fit(self, X, y, **params):
        """Fit the classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        **params : dict
            Parameters to pass to the `fit` method of the underlying classifier.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        routed_params = process_routing(self, "fit", **params)
        if self.calibrator is not None:
            self.estimator_ = self._get_calibrator(self.estimator).fit(X, y, **routed_params.estimator.fit)
        else:
            self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        return self

    def predict(
            self,
            X,
            tp_cost: ArrayLike | float = 0.0,
            tn_cost: ArrayLike | float = 0.0,
            fn_cost: ArrayLike | float = 0.0,
            fp_cost: ArrayLike | float = 0.0,
    ):
        """Predict the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

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

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self)

        estimator = getattr(self, "estimator_", self.estimator)

        if (all(isinstance(cost, Real) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) and
                sum((tp_cost, tn_cost, fn_cost, fp_cost)) == 0):
            return estimator.predict(X)

        y_score = estimator.predict_proba(X)[:, 1]

        if self.pos_label is None:
            map_thresholded_score_to_label = np.array([0, 1])
        else:
            pos_label_idx = np.flatnonzero(self.classes_ == self.pos_label)[0]
            neg_label_idx = np.flatnonzero(self.classes_ != self.pos_label)[0]
            map_thresholded_score_to_label = np.array([neg_label_idx, pos_label_idx])

        denominator = fp_cost - tn_cost + fn_cost - tp_cost
        denominator = np.clip(denominator, np.finfo(float).eps, denominator)  # Avoid division by zero
        optimal_thresholds = (fp_cost - tn_cost) / denominator

        return self.classes_[map_thresholded_score_to_label[(y_score >= optimal_thresholds).astype(int)]]

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`sklearn:User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`sklearn:sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee="fit", caller="fit"),
        )
        return router
