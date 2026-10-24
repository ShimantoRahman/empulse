from abc import abstractmethod
from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import clone, get_config
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold, check_cv, cross_val_predict
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.fixes import parse_version
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping, get_routing_for_object, process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, has_fit_parameter, indexable

from ..._common import Parameter
from ..._common._cost_routing import _all_costs_zero
from ..._common._metaestimators import estimator_has
from ..._common._sklearn_compat import Tags, get_tags, sklearn_version
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, Capability, CostMatrix, Metric
from ...metrics.metric.prebuilt_metrics import make_generic_cost_metric
from .._base.cost_sensitive import CostSensitiveClassifier


def _to_class_dependent_cost(cost: float | FloatNDArray) -> float:
    return float(np.mean(cost)) if isinstance(cost, np.ndarray) else cost


def _scalar_if_single(value: Any) -> Any:
    """Return a one-element array as its scalar, which applies to every sample; anything else as is."""
    return float(value.reshape(-1)[0]) if isinstance(value, np.ndarray) and value.size == 1 else value


# The plain costs as a classifier whose positive class is the other class sees them.
_COSTS_FOR_OTHER_POSITIVE_CLASS = {
    'tp_cost': 'tn_cost',
    'tn_cost': 'tp_cost',
    'fp_cost': 'fn_cost',
    'fn_cost': 'fp_cost',
}


def _class_dependent_metric(loss: BaseMetric, y: IntNDArray, loss_params: dict[str, Any]) -> Metric | None:
    """
    Return the metric with each cost replaced by its average over the samples, when a parameter is an array.

    Averaging the costs, rather than the parameters they are computed from, is what makes a cost such
    as the product of two instance-dependent parameters come out right. Only for a metric whose
    decision depends on nothing but the costs, which for a mixture of savings the labels *y* weigh;
    ``None`` otherwise.
    """
    if not loss._decides_from_costs:
        return None
    if not any(isinstance(value, np.ndarray) and value.size > 1 for value in loss_params.values()):
        return None
    fp_cost, fn_cost, tp_cost, tn_cost = loss._decision_costs(y, **loss_params)
    cost_matrix = (
        CostMatrix()
        .add_tp_cost(float(np.mean(tp_cost)))
        .add_tn_cost(float(np.mean(tn_cost)))
        .add_fp_cost(float(np.mean(fp_cost)))
        .add_fn_cost(float(np.mean(fn_cost)))
    )
    return Metric(cost_matrix, loss.strategy)


def _extract_loss_params(params: dict[str, Any], loss: BaseMetric) -> dict[str, Any]:
    loss_params = {}
    loss_param_names = loss._all_parameters
    for param_name in list(params.keys()):
        if param_name in loss_param_names:
            loss_params[param_name] = params[param_name]
    return loss_params


class CSDecisionRuleClassifier(MetaEstimatorMixin, CostSensitiveClassifier):  # type: ignore[misc]
    """
    Base class for cost-sensitive binary classifiers.

    Provides the common fit/predict skeleton for classifiers that optimize a
    cost-sensitive metric by learning a decision rule (threshold, rate, etc.) during fitting.

    Subclasses must implement:

    * :meth:`_compute_decision` — compute the decision attribute from the loss.
    * :meth:`_apply_decision` — apply the learned decision to produce class labels.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'estimator': [HasMethods(['fit', 'predict_proba'])],
        'pos_label': [Real, str, 'boolean', None],
        'cv': ['cv_object'],
    }
    # Keyword arguments other than the loss's parameters go to the estimator's fit.
    _extra_fit_params: ClassVar[frozenset[str] | None] = None

    _set_default_costs: ClassVar[bool] = False
    _routed_methods: ClassVar[tuple[str, ...]] = ('fit', 'predict')

    def __init__(
        self,
        estimator: Any,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: BaseMetric | None = None,
        pos_label: int | bool | str | None = None,
        cv: Any = None,
    ):
        self.estimator = estimator
        self.pos_label = pos_label
        self.cv = cv
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    @property
    def classes_(self) -> NDArray[Any]:
        """The class labels, taken from the wrapped estimator."""
        if estimator := getattr(self, 'estimator_', None):
            classes: NDArray[Any] = estimator.classes_
            return classes
        try:
            check_is_fitted(self.estimator)
            classes: NDArray[Any] = self.estimator.classes_  # type: ignore[no-redef]
            return classes
        except NotFittedError:
            raise AttributeError('The underlying estimator is not fitted yet.') from NotFittedError

    @classes_.setter
    def classes_(self, value: NDArray[Any]) -> None:
        if estimator := getattr(self, 'estimator_', None):
            estimator.classes_ = value
        else:
            raise AttributeError('The underlying estimator is not fitted yet.')

    def _get_loss_or_default(self) -> BaseMetric:
        """Return the configured loss or a generic cost metric."""
        return self.loss if self.loss is not None else make_generic_cost_metric()

    def _positive_class_index(self) -> int:
        """Return the index in :attr:`classes_` of the positive class: ``pos_label``, or ``classes_[1]``."""
        return self._positive_index_in(self.classes_)

    def _positive_index_in(self, classes: NDArray[Any]) -> int:
        """Return the index in the sorted *classes* of the positive class: ``pos_label``, or ``classes[1]``."""
        if self.pos_label is None:
            return 1
        matches = np.flatnonzero(classes == self.pos_label)
        if matches.size == 0:
            raise ValueError(f'pos_label={self.pos_label!r} is not one of the classes {list(classes)}.')
        return int(matches[0])

    def _decision_uses_scores(self, loss: BaseMetric) -> bool:
        """Return whether the decision learned with *loss* depends on the training scores, not only on costs."""
        return True

    def _positive_score(self, X: FloatArrayLike) -> FloatNDArray:
        """Return the predicted probability of the positive class for each sample of `X`."""
        y_score: FloatNDArray = self.predict_proba(X)[:, self._positive_class_index()]
        return y_score

    def _labels_for(self, is_positive: NDArray[np.bool_]) -> NDArray[Any]:
        """Map a boolean "predicted positive" array to the corresponding class labels."""
        positive_index = self._positive_class_index()
        y_pred: NDArray[Any] = np.where(is_positive, self.classes_[positive_index], self.classes_[1 - positive_index])
        return y_pred

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

    def _should_skip_cost_sensitive_fit(self, params: dict[str, Any]) -> bool:
        """
        Check if cost-sensitive fitting should be skipped.

        ``fit`` has already resolved the plain costs into *params*, so this is the case when there is no
        loss and every cost is zero, or when there is a loss but the caller did not supply a parameter it
        requires. ``_missing_parameters`` accepts a parameter under its symbol name or any alias, and
        ignores unrelated keys such as ``sample_weight``.
        """
        if self.loss is None:
            return _all_costs_zero({name: params[name] for name in self._cost_names})
        return bool(self.loss._missing_parameters(params.keys()))

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

    def _unfitted_estimator(self) -> tuple[str, Any]:
        """Return the routing name and an unfitted copy of the estimator to fit."""
        return 'estimator', clone(self.estimator)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        # X goes to the wrapped estimator as given, so it decides what X may contain.
        estimator_tags = get_tags(self.estimator)
        tags.input_tags.sparse = estimator_tags.input_tags.sparse
        tags.input_tags.allow_nan = estimator_tags.input_tags.allow_nan
        return tags

    def _child_fit_params(
        self,
        child: str,
        child_estimator: Any,
        params: dict[str, Any],
        loss_params: dict[str, Any],
        *,
        classes: NDArray[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return the ``fit`` keyword arguments for the wrapped estimator (or its calibrator) named `child`.

        With metadata routing enabled, the loss parameters the child requested are routed to it along
        with everything that is not a loss parameter, which scikit-learn checks was requested. Without
        routing, every argument that is not a loss parameter is forwarded, and so are the loss
        parameters the child's ``fit`` names, such as the costs of a cost-sensitive model. A
        :class:`~sklearn.calibration.CalibratedClassifierCV` forwards its keyword arguments to its
        estimator, so for it the estimator's ``fit`` is the one that has to name them.

        Given the sorted `classes`, plain costs are stated for ``pos_label``, so they are renamed for an
        estimator whose positive class is the other one: ``classes[1]``, or its own ``pos_label``.
        """
        consumer = child_estimator
        if isinstance(child_estimator, CalibratedClassifierCV):
            consumer = child_estimator.estimator
        if get_config()['enable_metadata_routing']:
            requested = get_routing_for_object(child_estimator).consumes('fit', loss_params.keys())
            routed = process_routing(
                self,
                'fit',
                **{key: value for key, value in params.items() if key not in loss_params or key in requested},
            )
            fit_params = dict(routed[child].fit)
        else:
            fit_params = {
                key: value
                for key, value in params.items()
                if key not in loss_params or has_fit_parameter(consumer, key)
            }
        if classes is not None:
            consumer_pos_label = consumer.get_params(deep=False).get('pos_label')
            consumer_index = 1
            if consumer_pos_label is not None:
                # An unknown ``pos_label`` is the consumer's to reject when it is fitted.
                matches = np.flatnonzero(classes == consumer_pos_label)
                consumer_index = int(matches[0]) if matches.size else 1
            if self._positive_index_in(classes) != consumer_index:
                fit_params = {_COSTS_FOR_OTHER_POSITIVE_CLASS.get(key, key): value for key, value in fit_params.items()}
        return fit_params

    @abstractmethod
    def _compute_decision(
        self,
        loss: BaseMetric,
        y: IntNDArray,
        y_score: FloatNDArray,
        loss_params: dict[str, Any],
    ) -> Any:
        """Compute the decision attribute from the loss function.

        Called during :meth:`_fit` to compute the subclass-specific decision
        attribute (e.g. threshold, rate).

        Parameters
        ----------
        loss : BaseMetric
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
        loss: BaseMetric,
        loss_params: dict[str, Any],
    ) -> Any:
        """Compute the decision attribute at predict time from the loss function.

        Called during :meth:`predict` when cost parameters are provided at predict time
        instead of using the fitted decision.

        Parameters
        ----------
        loss : BaseMetric
            The loss function to optimize.
        loss_params : dict
            Parameters for the loss function.

        Returns
        -------
        decision : Any
            The computed decision value.
        """

    @abstractmethod
    def _apply_decision(self, y_score: FloatNDArray, decision: Any) -> NDArray[Any]:
        """Apply the decision rule to produce class label predictions.

        Parameters
        ----------
        y_score : ndarray of shape (n_samples,)
            The predicted positive-class probabilities, already computed by the caller so this
            method doesn't need to score `X` a second time.
        decision : Any
            The decision value (threshold, rate, etc.).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """

    def _fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        loss: BaseMetric,
        **params: Any,
    ) -> Self:
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
        X, y = indexable(X, y)
        # Nothing a previous fit learned may outlive this one, such as a decision this fit does not learn.
        for attribute in ('decision_', 'n_features_in_', 'feature_names_in_'):
            if attribute in vars(self):
                delattr(self, attribute)

        plain_costs = self._get_metric_loss() is None
        loss_params = _extract_loss_params(params, self._get_loss_or_default())
        child, estimator = self._unfitted_estimator()
        child_params = self._child_fit_params(
            child,
            estimator,
            params,
            loss_params,
            classes=np.unique(y) if plain_costs else None,
        )
        self.estimator_ = estimator.fit(X, y, **child_params)
        if plain_costs:
            # The fallback for a cost not passed to predict.
            self._fit_costs = {name: _to_class_dependent_cost(params[name]) for name in self._cost_names}

        if not self._should_skip_cost_sensitive_fit(params):
            if self.cv is not None and self._decision_uses_scores(loss):
                y_proba = cross_val_predict(
                    self._unfitted_estimator()[1],
                    X,
                    y,
                    cv=check_cv(self.cv, y, classifier=True),
                    method='predict_proba',
                    params=child_params,
                )
            else:
                y_proba = self.estimator_.predict_proba(X)
            # Costs are defined against the positive class, so both the target and the score the
            # decision is computed from are those of `pos_label` (by default `classes_[1]`).
            positive_index = self._positive_class_index()
            y_binary = np.where(np.asarray(y) == self.classes_[positive_index], 1, 0)
            self.decision_ = self._compute_decision(loss, y_binary, y_proba[:, positive_index], loss_params)

        if hasattr(self.estimator_, 'n_features_in_'):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, 'feature_names_in_'):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    def predict(  # type: ignore[override]
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

        tn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of true negatives. If ``float``, then all true negatives have the same cost.
            If array-like, then it is the cost of each true negative classification.

        fn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        fp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        **loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.

        Notes
        -----
        Without a ``loss``, a cost not passed to ``predict`` keeps the value it had in ``fit``: its average
        over the samples, for an instance-dependent cost.
        """
        if self._should_use_fitted_decision(tp_cost, tn_cost, fn_cost, fp_cost, loss_params):
            check_is_fitted(self)
            decision = getattr(self, 'decision_', None)
            if decision is None:
                raise ValueError(
                    'decision has not been set during fit. '
                    'Either provide costs/benefits to fit first or provide costs to predict.'
                )
            if isinstance(decision, float) and np.isnan(decision):
                estimator: Any = getattr(self, 'estimator_', self.estimator)
                y_pred: NDArray[Any] = estimator.predict(X)
                return y_pred
            y_score = self._positive_score(X)
        else:
            if getattr(self, 'estimator_', None) is None:
                raise NotFittedError
            check_is_fitted(self.estimator_)

            loss = self._get_metric_loss() or make_generic_cost_metric()

            if Capability.COST_ONLY_DECISION not in loss.capabilities:
                raise ValueError(
                    f'Cannot use {loss.__name__!r} at predict time: its optimal '
                    'threshold/rate is a function of the observed scores or labels, not of the cost values '
                    'alone, so it cannot be recomputed from cost parameters passed to predict().'
                )

            costs = {'tp_cost': tp_cost, 'tn_cost': tn_cost, 'fn_cost': fn_cost, 'fp_cost': fp_cost}
            fit_costs = getattr(self, '_fit_costs', None)
            if fit_costs is not None and self._get_metric_loss() is None:
                costs = {
                    name: fit_costs.get(name, value) if value is Parameter.UNCHANGED else value
                    for name, value in costs.items()
                }
            loss_params = self._add_standard_costs_to_params(**costs, params=loss_params)
            y_score = self._positive_score(X)
            loss_params = self._normalize_cost_shapes(loss_params, size=len(y_score), names=loss._all_parameters)
            decision = self._compute_decision_at_predict(y_score, loss, loss_params)

        return self._apply_decision(y_score, decision)

    @available_if(estimator_has('predict_proba'))
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

    @available_if(estimator_has('predict_log_proba'))
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

    @available_if(estimator_has('decision_function'))
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
        """
        Get metadata routing of this object.

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

    Read more in the :ref:`User Guide <csthreshold>`.

    .. seealso::

        :class:`~empulse.models.CSRateClassifier` : Sets a target *rate* of positives to contact,
        for when capacity (e.g. a call centre) rather than a per-sample threshold is the constraint.

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

    pos_label : int, str, bool or None, default=None
        The label of the positive class, which the costs refer to.
        If None, the greater of the two classes (``classes_[1]``) is the positive class.

    random_state : int or None, default=None
        Random state for the calibrator. Ignored when `calibrator` is an Estimator.

    loss : BaseMetric or None, default=None
        The loss function to use for computing the optimal decision threshold.

        - If None, the optimal decision threshold is computed based on
          ``tp_cost``, ``tn_cost``, ``fn_cost``, and ``fp_cost``.
        - If a :class:`~empulse.metrics.BaseMetric`,
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

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    cv : int, cross-validation generator, iterable or None, default=None
        Where the scores the threshold is learned from come from, when the threshold depends on them,
        as it does for a ``loss`` such as :class:`~empulse.metrics.MaxProfit`. A threshold that depends
        only on the costs, such as the default, ignores it.

        - If None, the scores of the estimator fitted on all the training data, which flatter an
          estimator that overfits it.
        - Otherwise, out-of-fold scores from :func:`~sklearn.model_selection.cross_val_predict` with
          this splitting strategy (see :func:`~sklearn.model_selection.check_cv`), at the cost of
          fitting the estimator, with its calibrator if any, once more per fold. ``estimator_`` is
          still fitted on all the training data.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The classes labels.

    estimator_ : Estimator
        The fitted classifier.

    threshold_ : float
        The optimal decision threshold determined during fitting. It is ``inf`` when the costs make
        targeting nobody optimal, so that no score reaches it.

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

    def __init__(
        self,
        estimator: Any,
        *,
        calibrator: Literal['sigmoid', 'isotonic'] | BaseEstimator | None = 'sigmoid',
        pos_label: int | bool | str | None = None,
        random_state: int | np.random.RandomState | None = None,
        loss: BaseMetric | None = None,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        cv: Any = None,
    ):
        self.calibrator = calibrator
        self.random_state = random_state
        super().__init__(
            estimator,
            pos_label=pos_label,
            cv=cv,
            loss=loss,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )

    def _get_calibrator(self, estimator: Any) -> Any:
        if self.calibrator in {'sigmoid', 'isotonic'}:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            calibrator = CalibratedClassifierCV(estimator, method=self.calibrator, cv=cv, ensemble=False)
            # No caller can state the request of a calibrator built here, so it weights the calibration
            # by the sample weights when the estimator is fitted with them. Otherwise the request stays
            # unset: the calibrator routes its own ``sample_weight=None``, which an explicit ``False``
            # would reject.
            if get_config()['enable_metadata_routing'] and get_routing_for_object(estimator).consumes(
                'fit', ['sample_weight']
            ):
                calibrator.set_fit_request(sample_weight=True)
            return calibrator
        else:
            # Clone so refitting cannot mutate the calibrator the caller passed to __init__.
            return clone(self.calibrator).set_params(estimator=estimator)  # type: ignore[union-attr]

    def _unfitted_estimator(self) -> tuple[str, Any]:
        """Return the calibrator wrapping the estimator, if one is configured."""
        if self.calibrator is None:
            return super()._unfitted_estimator()
        return 'calibrator', self._get_calibrator(self.estimator)

    @property
    def threshold_(self) -> float | None:
        """The optimal decision threshold determined during fitting."""
        return getattr(self, 'decision_', None)

    def _decision_uses_scores(self, loss: BaseMetric) -> bool:
        return Capability.COST_ONLY_DECISION not in loss.capabilities

    def _compute_decision(
        self,
        loss: BaseMetric,
        y: IntNDArray,
        y_score: FloatNDArray,
        loss_params: dict[str, Any],
    ) -> float | FloatNDArray:
        # A single threshold needs class-dependent costs. A loss whose decision depends on the scores
        # reduces instance-dependent parameters itself, from the costs they make up.
        class_dependent_loss = _class_dependent_metric(loss, y, loss_params)
        if class_dependent_loss is not None:
            return class_dependent_loss.optimal_threshold(y, y_score)
        loss_params = {key: _scalar_if_single(value) for key, value in loss_params.items()}
        return loss.optimal_threshold(y, y_score, **loss_params)

    def _compute_decision_at_predict(
        self,
        y_score: FloatNDArray,
        loss: BaseMetric,
        loss_params: dict[str, Any],
    ) -> float | FloatNDArray:
        return loss.optimal_threshold(np.array([]), np.array([]), **loss_params)

    def _apply_decision(self, y_score: FloatNDArray, decision: Any) -> NDArray[Any]:
        return self._labels_for(y_score >= decision)

    def get_metadata_routing(self) -> MetadataRouter:
        """
        Get metadata routing of this object.

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

    Read more in the :ref:`User Guide <csrate>`.

    .. seealso::

        :class:`~empulse.models.CSThresholdClassifier` : Sets a per-sample decision *threshold*
        instead of a fixed rate, for when there is no fixed capacity constraint.

    Parameters
    ----------
    estimator : object
        A binary classifier that implements `fit` and `predict_proba`.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` or ``predict`` method.

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

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` or ``predict`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` or ``predict`` method.

    loss : :class:`~empulse.metrics.BaseMetric` or None, default=None
        The cost-sensitive metric to optimize.

        - If None, the optimal positive rate is computed based on
          ``tp_cost``, ``tn_cost``, ``fn_cost``, and ``fp_cost``.
        - If a :class:`~empulse.metrics.BaseMetric`,
          the optimal positive rate is computed based on the loss parameters provided to
          the :meth:`fit` or :meth:`predict` method.

    pos_label : int, str, bool or None, default=None
        The label of the positive class, which the costs refer to.
        If None, the greater of the two classes (``classes_[1]``) is the positive class.

    cv : int, cross-validation generator, iterable or None, default=None
        Where the scores the rate is learned from come from.

        - If None, the scores of the estimator fitted on all the training data, which flatter an
          estimator that overfits it.
        - Otherwise, out-of-fold scores from :func:`~sklearn.model_selection.cross_val_predict` with
          this splitting strategy (see :func:`~sklearn.model_selection.check_cv`), at the cost of
          fitting the estimator, with its calibrator if any, once more per fold. ``estimator_`` is
          still fitted on all the training data.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The class labels.

    estimator_ : Estimator
        The fitted classifier.

    rate_ : float
        The optimal positive rate determined during fitting.

    Notes
    -----
    The rate applies to each call of :meth:`predict`: exactly ``ceil(rate_ * n_samples)`` of the samples
    passed are predicted positive, those with the highest scores, with ties going to the samples passed
    first. A sample's prediction therefore depends on the others it is predicted with; a single sample is
    predicted positive whenever ``rate_ > 0``. Predict the whole population to target at once.
    """

    @property
    def rate_(self) -> float | None:
        """The optimal positive rate determined during fitting."""
        return getattr(self, 'decision_', None)

    def _compute_decision(
        self,
        loss: BaseMetric,
        y: IntNDArray,
        y_score: FloatNDArray,
        loss_params: dict[str, Any],
    ) -> float:
        return loss.optimal_rate(y, y_score, **loss_params)

    def _compute_decision_at_predict(
        self,
        y_score: FloatNDArray,
        loss: BaseMetric,
        loss_params: dict[str, Any],
    ) -> float:
        return loss.optimal_rate(np.array([]), y_score, **loss_params)

    def _apply_decision(self, y_score: FloatNDArray, decision: Any) -> NDArray[Any]:
        is_positive = np.zeros(len(y_score), dtype=bool)
        if not np.isnan(decision):
            # Exactly the top ``ceil(rate * n)`` scores, so tied scores cannot push the count past the rate.
            # Rounding first keeps a rate of k / n from targeting k + 1 samples, as 0.07 * 100 would.
            n_positive = int(np.ceil(np.round(decision * len(y_score), 9)))
            is_positive[np.argsort(-np.asarray(y_score, dtype=np.float64), kind='stable')[:n_positive]] = True
        return self._labels_for(is_positive)
