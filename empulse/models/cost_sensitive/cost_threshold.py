import warnings
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
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.model_selection import StratifiedKFold


class CSThresholdClassifier(BaseThresholdClassifier):
    """
    Cost-sensitive threshold classifier.

    Classifier which sets the decision threshold to optimize the instance-specific cost loss.

    Parameters
    ----------
    estimator : object
        A classifier with a `predict_proba` method.

    calibrator : {'sigmoid', 'isotonic'}, Estimator or None, default='sigmoid'
        The calibrator to use.
            - If 'sigmoid', then a :class:`sklearn:sklearn.calibration.CalibratedClassifierCV` with `method='sigmoid'`
            and `ensemble=False` is used.
            - If 'isotonic', then a :class:`sklearn:sklearn.calibration.CalibratedClassifierCV` with `method='isotonic'`
            and `ensemble=False` is used.
            - If an Estimator, then it should have a `fit` and `predict_proba` method.
            - If None, probabilities are assumed to be well-calibrated.

    pos_label : int, str, 'boolean' or None, default=None
        The positive label. If None, the positive label is assumed to be 1.

    random_state : int or None, default=None
        Random state for the calibrator. Ignored when `calibrator` is an Estimator.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The classes labels.

    estimator_ : Estimator
        The fitted classifier.

    Notes
    -----
    The optimal threshold is computed as [1]_:

    .. math:: t^*_i = \\frac{C_i(1|0) - C_i(0|0)}{C_i(1|0) - C_i(0|0) + C_i(0|1) - C_i(1|1)}

    .. note:: The optimal decision threshold is only accurate when the probabilities are well-calibrated.
              Therefore, it is recommended to use a calibrator when the probabilities are not well-calibrated.
              See `scikit-learn's user guide <https://scikit-learn.org/stable/modules/calibration.html>`_
              for more information.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict_proba"]), ],
        "calibrator": [
            HasMethods(["fit", "predict_proba"]),
            StrOptions({"sigmoid", "isotonic"}),
            None
        ],
        "pos_label": [Real, str, "boolean", None],
        "random_state": ["random_state"],
    }

    def __init__(
            self,
            estimator,
            *,
            calibrator: Literal['sigmoid', 'isotonic'] | object | None = 'sigmoid',
            pos_label=None,
            random_state=None,
    ):
        super().__init__(estimator, response_method='predict_proba')
        self.calibrator = calibrator
        self.pos_label = pos_label
        self.random_state = random_state

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
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(
                estimator,
                method='sigmoid',
                cv=cv,
                ensemble=False
            )
        elif self.calibrator == 'isotonic':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(
                estimator,
                method='isotonic',
                cv=cv,
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
        X, y = validate_data(self, X, y)
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
        """
        Predict the target of new samples.

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

        Notes
        -----
        If all costs are zero, then ``fp_cost=1`` and ``fn_cost=1`` are used to avoid division by zero.
        """
        check_is_fitted(self)

        estimator = getattr(self, "estimator_", self.estimator)

        if (all(isinstance(cost, Real) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) and
                sum((tp_cost, tn_cost, fn_cost, fp_cost)) == 0):
            warnings.warn(
                "All costs are zero. Setting fp_cost=1 and fn_cost=1. "
                f"To avoid this warning, set costs explicitly in the {self.__class__.__name__}.predict() method.",
                UserWarning)
            fp_cost = 1
            fn_cost = 1

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
