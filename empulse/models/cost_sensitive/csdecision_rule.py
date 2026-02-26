from abc import abstractmethod
from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
import sklearn
from numpy.typing import ArrayLike, NDArray
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, _fit_context
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._metadata_requests import RequestMethod
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.fixes import parse_version
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping, process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _estimator_has, check_is_fitted, indexable

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import MaxProfit, Metric
from ...metrics.metric.prebuilt_metrics import make_generic_cost_metric
from ...utils._sklearn_compat import Tags, validate_data  # type: ignore[attr-defined]
from .._cs_mixin import CostSensitiveMixin

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


def _to_class_dependent_cost(cost: float | FloatNDArray) -> float:
    return float(np.mean(cost)) if isinstance(cost, np.ndarray) else cost


def _extract_loss_params(params: dict[str, Any], loss: Metric) -> dict[str, Any]:
    loss_params = {}
    loss_param_names = loss._all_symbols
    for param_name in list(params.keys()):
        if param_name in loss_param_names:
            loss_params[param_name] = params[param_name]
    return loss_params


class CSDecisionRuleClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator, CostSensitiveMixin):  # type: ignore[misc]
    """Base class for cost-sensitive binary classifiers.

    Provides the common fit/predict skeleton for classifiers that optimize a
    cost-sensitive metric by learning a decision rule (threshold, rate, etc.)
    during fitting.

    Subclasses must implement:

    * :meth:`_compute_decision` — compute the decision attribute from the loss.
    * :meth:`_apply_decision` — apply the learned decision to produce class labels.

    And define:

    * ``_decision_attr_name`` — the name of the fitted attribute (e.g. ``'threshold_'``).
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'estimator': [HasMethods(['fit', 'predict_proba'])],
        'pos_label': [Real, str, 'boolean', None],
        'loss': [Metric, None],
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
    }

    #: Name of the fitted decision attribute (e.g. ``'threshold_'``, ``'rate_'``).
    _decision_attr_name: ClassVar[str]

    def __init__(
        self,
        estimator: Any,
        *,
        pos_label: int | bool | str | None = None,
        loss: Metric | None = None,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
    ):
        self.estimator = estimator
        self.pos_label = pos_label
        self.loss = loss
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        super().__init__()

    def __post_init__(self) -> None:
        # Override to handle both fit and predict request routing.
        # Cannot call super().__post_init__() because the mixin uses
        # router.fit.requests which is incompatible with add_self_request routers.
        if isinstance(self._get_metric_loss(), Metric):
            self.__class__.set_fit_request = RequestMethod(  # type: ignore[attr-defined]
                'fit',
                sorted(self.get_metadata_routing()._self_request.fit.requests.keys() | self.loss._all_symbols),  # type: ignore[attr-defined, union-attr]
            )
            self.__class__.set_predict_request = RequestMethod(  # type: ignore[attr-defined]
                'predict',
                sorted(self.get_metadata_routing()._self_request.predict.requests.keys() | self.loss._all_symbols),  # type: ignore[attr-defined, union-attr]
            )

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
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

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None

    def _get_loss_or_default(self) -> Metric:
        """Return the configured loss or a generic cost metric."""
        return self.loss if self.loss is not None else make_generic_cost_metric()

    def _all_init_costs_zero(self) -> bool:
        """Check if all init costs are zero/default."""
        return all((
            not (isinstance(self.tp_cost, np.ndarray) or self.tp_cost != 0.0),
            not (isinstance(self.tn_cost, np.ndarray) or self.tn_cost != 0.0),
            not (isinstance(self.fn_cost, np.ndarray) or self.fn_cost != 0.0),
            not (isinstance(self.fp_cost, np.ndarray) or self.fp_cost != 0.0),
        ))

    @staticmethod
    def _all_costs_unchanged(
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
    ) -> bool:
        """Check if all costs are unchanged (i.e., not provided by the caller)."""
        return all((
            tp_cost is Parameter.UNCHANGED,
            tn_cost is Parameter.UNCHANGED,
            fn_cost is Parameter.UNCHANGED,
            fp_cost is Parameter.UNCHANGED,
        ))

    def _should_skip_cost_sensitive_fit(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        params: dict[str, Any],
    ) -> bool:
        """
        Check if cost-sensitive fitting should be skipped.

        Returns True if:
        1. All init costs are zero/default AND all fit costs are unchanged AND no loss function
        2. Loss function exists but no loss-specific parameters were provided
        """
        # First condition: no costs provided and no loss function
        no_costs_no_loss = (
            self._all_init_costs_zero()
            and self._all_costs_unchanged(tp_cost, tn_cost, fn_cost, fp_cost)
            and self.loss is None
        )

        # Second condition: loss exists but no loss-specific params provided
        loss_without_params = (
            self.loss is not None
            and not any(self.loss._all_symbols.intersection(params.keys()))
            and not self.loss.cost_matrix._defaults
        )

        return no_costs_no_loss or loss_without_params

    def _should_use_fitted_decision(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        loss_params: dict[str, Any],
    ) -> bool:
        """
        Check if the decision attribute (threshold/rate) fitted during training should be used.

        Returns True when no new cost information is provided at predict time,
        meaning the caller wants to use whatever decision was learned during fit.
        """
        return self._all_costs_unchanged(tp_cost, tn_cost, fn_cost, fp_cost) and not loss_params

    def _fit_estimator(
        self, X: FloatArrayLike, y: ArrayLike, **params: Any
    ) -> tuple[Any, FloatNDArray, dict[str, Any]]:
        """Fit the underlying estimator and return scores and loss parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **params : dict
            Parameters containing both loss params and estimator routing params.

        Returns
        -------
        estimator_ : Estimator
            The fitted estimator.
        y_score : ndarray of shape (n_samples,)
            Predicted probabilities for the positive class.
        loss_params : dict
            Parameters extracted for the loss function.
        """
        loss = self._get_loss_or_default()

        loss_params = _extract_loss_params(params, loss)
        routing = self.estimator.get_metadata_routing()
        routing_params = {k: v for k, v in params.items() if k in routing.fit.requests}
        routed_params = process_routing(self, 'fit', **routing_params)
        estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

        y_score: FloatNDArray = estimator_.predict_proba(X)[:, 1]
        return estimator_, y_score, loss_params

    @abstractmethod
    def _compute_decision(
        self,
        loss: Metric,
        y: IntNDArray,
        y_score: FloatNDArray,
        loss_params: dict[str, Any],
    ) -> Any:
        """Compute the decision attribute from the loss function.

        Called during :meth:`_fit` to compute the subclass-specific decision
        attribute (e.g. threshold, rate).

        Parameters
        ----------
        loss : Metric
            The loss function to optimize.
        y : array-like of shape (n_samples,)
            True labels.
        y_score : ndarray of shape (n_samples,)
            Predicted probabilities for the positive class.
        loss_params : dict
            Parameters for the loss function.

        Returns
        -------
        decision : Any
            The computed decision value to store as the fitted attribute.
        """

    @abstractmethod
    def _compute_decision_at_predict(
        self,
        y_score: FloatNDArray,
        loss: Metric,
        loss_params: dict[str, Any],
    ) -> Any:
        """Compute the decision attribute at predict time from the loss function.

        Called during :meth:`predict` when cost parameters are provided at predict time
        instead of using the fitted decision.

        Parameters
        ----------
        loss : Metric
            The loss function to optimize.
        loss_params : dict
            Parameters for the loss function.

        Returns
        -------
        decision : Any
            The computed decision value.
        """

    @abstractmethod
    def _apply_decision(self, X: FloatArrayLike, decision: Any) -> NDArray[Any]:
        """Apply the decision rule to produce class label predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        decision : Any
            The decision value (threshold, rate, etc.).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **params: Any,
    ) -> Self:
        """Fit the classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

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

        **params : dict
            Parameters to pass to the `fit` method of the underlying classifier.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X, y = indexable(X, y)

        y_type = type_of_target(y, input_name='y')
        if y_type != 'binary':
            raise ValueError(f'Only binary classification is supported. Unknown label type: {y_type}')

        if self._should_skip_cost_sensitive_fit(tp_cost, tn_cost, fn_cost, fp_cost, params):
            X, y = validate_data(self, X, y)
            self.estimator_ = clone(self.estimator).fit(X, y, **params)
        else:
            params = self._add_standard_costs_to_params(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, params=params
            )
            self._fit(X, y, **params)

        if hasattr(self.estimator_, 'n_features_in_'):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, 'feature_names_in_'):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    def _fit(self, X: FloatArrayLike, y: ArrayLike, **params: Any) -> Self:
        """Fit the classifier with cost-sensitive optimization.

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

        loss = self._get_loss_or_default()

        self.estimator_, y_score, loss_params = self._fit_estimator(X, y, **params)

        decision = self._compute_decision(loss, y, y_score, loss_params)
        setattr(self, self._decision_attr_name, decision)

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
        if self._should_use_fitted_decision(tp_cost, tn_cost, fn_cost, fp_cost, loss_params):
            check_is_fitted(self)
            decision = getattr(self, self._decision_attr_name, None)
            if decision is None:
                raise ValueError(
                    f'{self._decision_attr_name[:-1]} has not been set during fit. '
                    'Either provide costs/benefits to fit first or provide costs to predict.'
                )
            if isinstance(decision, float) and np.isnan(decision):
                estimator: Any = getattr(self, 'estimator_', self.estimator)
                y_pred: NDArray[Any] = estimator.predict(X)
                return y_pred
        else:
            if getattr(self, 'estimator_', None) is None:
                raise NotFittedError
            check_is_fitted(self.estimator_)

            loss = self._get_loss_or_default()

            if isinstance(loss.strategy, MaxProfit):
                raise ValueError(f'Cannot use {loss.strategy.__class__.__name__} at predict time.')

            loss_params = self._add_standard_costs_to_params(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, params=loss_params
            )
            y_proba = self.predict_proba(X)
            decision = self._compute_decision_at_predict(y_proba, loss, loss_params)

        return self._apply_decision(X, decision)

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """Predict class probabilities for `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        estimator = getattr(self, 'estimator_', self.estimator)
        y_proba: FloatNDArray = estimator.predict_proba(X)
        return y_proba

    @available_if(_estimator_has('predict_log_proba'))
    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """Predict logarithm class probabilities for `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        log_probabilities : ndarray of shape (n_samples, n_classes)
            The logarithm class probabilities of the input samples.
        """
        check_is_fitted(self)
        estimator = getattr(self, 'estimator_', self.estimator)
        y_log_proba: FloatNDArray = estimator.predict_log_proba(X)
        return y_log_proba

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X: FloatArrayLike) -> FloatNDArray:
        """Decision function for samples in `X` using the fitted estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,)
            The decision function computed the fitted estimator.
        """
        check_is_fitted(self)
        estimator = getattr(self, 'estimator_', self.estimator)
        y_score: FloatNDArray = estimator.decision_function(X)
        return y_score

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
        if sklearn_version < parse_version('1.8'):
            router = MetadataRouter(owner=self.__class__.__name__)  # type: ignore[arg-type]
        else:
            router = MetadataRouter(owner=self)  # type: ignore[arg-type]

        router.add_self_request(self)
        router.add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee='fit', caller='fit'),
        )

        return router


class CSThresholdClassifier(CSDecisionRuleClassifier):
    r"""
    Binary Classifier that sets the decision threshold to optimize the cost-sensitive metric.

    Users can learn the optimal decision threshold during fitting the model
    and apply that threshold during inference. This is done by passing the costs/benefits to the fit method.

    Alternatively, users can determine the optimal decision threshold during inference
    by passing the costs to the predict method.

    By default, the expected cost loss is optimized, but a custom loss function can be passed to the init method.

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
        Is overwritten if another `tp_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The classes labels.

    estimator_ : Estimator
        The fitted classifier.

    threshold_ : float
        The optimal decision threshold determined during fitting.

    Notes
    -----

    .. note:: The optimal decision threshold is only accurate when the probabilities are well-calibrated.
              Therefore, it is recommended to use a calibrator when the probabilities are not well-calibrated.
              See `scikit-learn's user guide <https://scikit-learn.org/stable/modules/calibration.html>`_
              for more information.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CSDecisionRuleClassifier._parameter_constraints,
        'calibrator': [HasMethods(['fit', 'predict_proba']), StrOptions({'sigmoid', 'isotonic'}), None],
        'random_state': ['random_state'],
    }

    _decision_attr_name: ClassVar[str] = 'threshold_'

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
        self.calibrator = calibrator
        self.random_state = random_state
        super().__init__(
            estimator,
            pos_label=pos_label,
            loss=loss,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )

    def _get_calibrator(self, estimator: Any) -> Any:
        if self.calibrator == 'sigmoid':
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(estimator, method='sigmoid', cv=cv, ensemble=False)
        elif self.calibrator == 'isotonic':
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(estimator, method='isotonic', cv=cv, ensemble=False)
        else:
            return self.calibrator.set_params(estimator=estimator)  # type: ignore[union-attr]

    def _fit_estimator(
        self, X: FloatArrayLike, y: ArrayLike, **params: Any
    ) -> tuple[Any, FloatNDArray, dict[str, Any]]:
        """Fit the underlying estimator, handling calibration if configured.

        Overrides the base implementation to support probability calibration.
        """
        loss = self._get_loss_or_default()

        loss_params = _extract_loss_params(params, loss)
        routing = self.estimator.get_metadata_routing()
        routing_params = {k: v for k, v in params.items() if k in routing.fit.requests}
        routed_params = process_routing(self, 'fit', **routing_params)
        if self.calibrator is not None:
            estimator_ = self._get_calibrator(self.estimator).fit(X, y, **routed_params.calibrator.fit)
        else:
            estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

        y_score: FloatNDArray = estimator_.predict_proba(X)[:, 1]
        return estimator_, y_score, loss_params

    def _compute_decision(
        self,
        loss: Metric,
        y: IntNDArray,
        y_score: FloatNDArray,
        loss_params: dict[str, Any],
    ) -> float | FloatNDArray:
        # Convert instance-dependent costs to class-dependent so we only get a single threshold.
        for key, value in list(loss_params.items()):
            loss_params[key] = _to_class_dependent_cost(value)
        return loss.optimal_threshold(y, y_score, **loss_params)

    def _compute_decision_at_predict(
        self,
        y_score: FloatNDArray,
        loss: Metric,
        loss_params: dict[str, Any],
    ) -> float | FloatNDArray:
        return loss.optimal_threshold(np.array([]), np.array([]), **loss_params)

    def _apply_decision(self, X: FloatArrayLike, decision: Any) -> NDArray[Any]:
        if self.pos_label is None:
            map_thresholded_score_to_label = np.array([0, 1])
        else:
            pos_label_idx = np.flatnonzero(self.classes_ == self.pos_label)[0]
            neg_label_idx = np.flatnonzero(self.classes_ != self.pos_label)[0]
            map_thresholded_score_to_label = np.array([neg_label_idx, pos_label_idx])

        estimator: Any = getattr(self, 'estimator_', self.estimator)
        y_score = estimator.predict_proba(X)[:, 1]
        y_pred: NDArray[Any] = self.classes_[map_thresholded_score_to_label[(y_score >= decision).astype(int)]]
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
        if sklearn_version < parse_version('1.8'):
            router = MetadataRouter(owner=self.__class__.__name__)  # type: ignore[arg-type]
        else:
            router = MetadataRouter(owner=self)  # type: ignore[arg-type]

        router.add_self_request(self)

        if self.calibrator is not None:
            router.add(
                calibrator=self._get_calibrator(self.estimator),
                method_mapping=MethodMapping().add(callee='fit', caller='fit'),
            )
        else:
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(callee='fit', caller='fit'),
            )

        return router


class CSRateClassifier(CSDecisionRuleClassifier):
    r"""
    Binary Classifier that sets the positive rate to optimize the cost-sensitive metric.

    This classifier classifies the top fraction of samples (by predicted probability)
    as positive, where the fraction is determined by the optimal rate computed during fitting.

    Parameters
    ----------
    estimator : object
        A binary classifier that implements `fit` and `predict_proba`.

    pos_label : int, str, 'boolean' or None, default=None
        The label of the positive class.

    loss : Metric or None, default=None
        The cost-sensitive metric to optimize.

        - If None, the optimal positive rate is computed based on
          ``tp_cost``, ``tn_cost``, ``fn_cost``, and ``fp_cost``.
        - If a :class:`~empulse.metrics.Metric`,
          the optimal positive rate is computed based on the loss parameters provided to
          the :meth:`fit` or :meth:`predict` method.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The class labels.

    estimator_ : Estimator
        The fitted classifier.

    rate_ : float
        The optimal positive rate determined during fitting.
    """

    _decision_attr_name: ClassVar[str] = 'rate_'

    def _compute_decision(
        self,
        loss: Metric,
        y: IntNDArray,
        y_score: FloatNDArray,
        loss_params: dict[str, Any],
    ) -> float:
        return loss.optimal_rate(y, y_score, **loss_params)

    def _compute_decision_at_predict(
        self,
        y_score: FloatNDArray,
        loss: Metric,
        loss_params: dict[str, Any],
    ) -> float:
        return loss.optimal_rate(np.array([]), y_score, **loss_params)

    def _apply_decision(self, X: FloatArrayLike, decision: Any) -> NDArray[Any]:
        y_score = self.estimator_.predict_proba(X)[:, 1]
        n_samples = len(y_score)

        if np.isnan(decision):
            return np.full(n_samples, self.classes_[0])

        n_positive = int(np.ceil(decision * n_samples))

        if n_positive == 0:
            return np.full(n_samples, self.classes_[0])
        if n_positive >= n_samples:
            return np.full(n_samples, self.classes_[1])

        threshold_idx = np.argsort(y_score)[-n_positive]
        threshold = y_score[threshold_idx]

        return np.where(y_score >= threshold, self.classes_[1], self.classes_[0])
