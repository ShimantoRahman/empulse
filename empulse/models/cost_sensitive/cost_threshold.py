import sys
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._classification_threshold import BaseThresholdClassifier
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping, process_routing
from sklearn.utils.validation import check_is_fitted

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, ParameterConstraint
from ...metrics import Metric
from ...utils._sklearn_compat import Tags, validate_data  # type: ignore[attr-defined]
from ._cs_mixin import CostSensitiveMixin

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class CSThresholdClassifier(CostSensitiveMixin, BaseThresholdClassifier):  # type: ignore[misc]
    r"""
    Classifier that sets the decision threshold to optimize the instance-dependent cost loss.

    Parameters
    ----------
    estimator : object
        A classifier with a `predict_proba` method.

    calibrator : {'sigmoid', 'isotonic'}, Estimator or None, default='sigmoid'
        The calibrator to use.

        - If 'sigmoid', then a :class:`~sklearn:sklearn.calibration.CalibratedClassifierCV` with `method='sigmoid'`
          and `ensemble=False` is used.
        - If 'isotonic', then a :class:`~sklearn:sklearn.calibration.CalibratedClassifierCV` with `method='isotonic'`
          and `ensemble=False` is used.
        - If an Estimator, then it should have a `fit` and `predict_proba` method.
        - If None, probabilities are assumed to be well-calibrated.

    pos_label : int, str, 'boolean' or None, default=None
        The positive label. If None, the positive label is assumed to be 1.

    random_state : int or None, default=None
        Random state for the calibrator. Ignored when `calibrator` is an Estimator.

    loss : Metric or None, default=None
        The loss function to use for computing the optimal decision threshold.

        - If None, the optimal decision threshold is computed based on
          ``tp_cost``, ``tn_cost``, ``fn_cost``, and ``fp_cost``.
        - If a :class:`~empulse.metrics.Metric`,
          the optimal decision threshold is computed based on the loss parameters provided to
          the :meth:`predict` method.

        Read the :ref:`User Guide <metric_class_in_model>` for more information.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``predict`` method.

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
        Is overwritten if another `fp_cost` is passed to the ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``predict`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``predict`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``predict`` method.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The classes labels.

    estimator_ : Estimator
        The fitted classifier.

    Notes
    -----
    The optimal threshold is computed as [1]_:

    .. math:: t^*_i = \frac{C_i(1|0) - C_i(0|0)}{C_i(1|0) - C_i(0|0) + C_i(0|1) - C_i(1|1)}

    .. note:: The optimal decision threshold is only accurate when the probabilities are well-calibrated.
              Therefore, it is recommended to use a calibrator when the probabilities are not well-calibrated.
              See `scikit-learn's user guide <https://scikit-learn.org/stable/modules/calibration.html>`_
              for more information.

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'estimator': [HasMethods(['fit', 'predict_proba'])],
        'calibrator': [HasMethods(['fit', 'predict_proba']), StrOptions({'sigmoid', 'isotonic'}), None],
        'pos_label': [Real, str, 'boolean', None],
        'random_state': ['random_state'],
        'loss': [HasMethods('_gradient_boost_objective'), None],
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
    }

    def __init__(
        self,
        estimator: Any,
        *,
        calibrator: Literal['sigmoid', 'isotonic'] | BaseEstimator | None = 'sigmoid',
        pos_label: int | bool | str | None = None,
        random_state: int | np.random.RandomState | None = None,
        loss: Metric | None = None,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
    ):
        super().__init__(estimator, response_method='predict_proba')
        self.calibrator = calibrator
        self.pos_label = pos_label
        self.random_state = random_state
        self.loss = loss
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags

    @property
    def classes_(self) -> NDArray[Any]:  # noqa: D102
        if estimator := getattr(self, 'estimator_', None):
            classes: NDArray[Any] = estimator.classes_
            return classes
        try:
            check_is_fitted(self.estimator)
            classes: NDArray[Any] = self.estimator.classes_  # type: ignore[no-redef]
            return classes
        except NotFittedError:
            raise AttributeError('The underlying estimator is not fitted yet.') from NotFittedError

    def _get_calibrator(self, estimator: Any) -> Any:
        if self.calibrator == 'sigmoid':
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(estimator, method='sigmoid', cv=cv, ensemble=False)
        elif self.calibrator == 'isotonic':
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(estimator, method='isotonic', cv=cv, ensemble=False)
        else:
            return self.calibrator.set_params(estimator=estimator)  # type: ignore[union-attr]

    def _fit(self, X: FloatArrayLike, y: ArrayLike, **params: Any) -> Self:
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
        routed_params = process_routing(self, 'fit', **params)
        if self.calibrator is not None:
            self.estimator_ = self._get_calibrator(self.estimator).fit(X, y, **routed_params.estimator.fit)
        else:
            self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        return self

    def predict(
        self,
        X: FloatNDArray,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> NDArray[Any]:
        """
        Predict the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

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

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.

        Notes
        -----
        If all costs are zero, then ``fp_cost=1`` and ``fn_cost=1`` are used to avoid division by zero.
        """
        check_is_fitted(self)

        if self.loss is None:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, caller='predict'
            )

            denominator = fp_cost - tn_cost + fn_cost - tp_cost
            # Avoid division by zero
            if isinstance(denominator, float | int):
                if denominator == 0:
                    denominator += float(np.finfo(float).eps)
            else:
                denominator = np.clip(denominator, float(np.finfo(float).eps), denominator)
            optimal_thresholds = (fp_cost - tn_cost) / denominator
        else:
            optimal_thresholds = self.loss._bayes_minimum_risk_thresholds(**loss_params)

        if self.pos_label is None:
            map_thresholded_score_to_label = np.array([0, 1])
        else:
            pos_label_idx = np.flatnonzero(self.classes_ == self.pos_label)[0]
            neg_label_idx = np.flatnonzero(self.classes_ != self.pos_label)[0]
            map_thresholded_score_to_label = np.array([neg_label_idx, pos_label_idx])

        estimator: Any = getattr(self, 'estimator_', self.estimator)
        y_score = estimator.predict_proba(X)[:, 1]
        y_pred: NDArray[Any] = self.classes_[
            map_thresholded_score_to_label[(y_score >= optimal_thresholds).astype(int)]
        ]
        return y_pred

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <sklearn:metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`sklearn:sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee='fit', caller='fit'),
        )
        return router
