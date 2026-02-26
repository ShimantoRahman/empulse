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
from ..._types import FloatArrayLike, FloatNDArray, ParameterConstraint
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


class CSThresholdClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator, CostSensitiveMixin):  # type: ignore[misc]
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

    .. note:: The optimal decision threshold is only accurate when the probabilities are well-calibrated.
              Therefore, it is recommended to use a calibrator when the probabilities are not well-calibrated.
              See `scikit-learn's user guide <https://scikit-learn.org/stable/modules/calibration.html>`_
              for more information.
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
        self.calibrator = calibrator
        self.pos_label = pos_label
        self.random_state = random_state
        self.loss = loss
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.estimator = estimator
        super().__init__()

    def __post_init__(self) -> None:
        if isinstance(self._get_metric_loss(), Metric):
            self.__class__.set_predict_request = RequestMethod(  # type: ignore[attr-defined]
                'predict',
                sorted(self.get_metadata_routing()._self_request.predict.requests.keys() | self.loss._all_symbols),  # type: ignore[attr-defined, union-attr]
            )

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

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
        # Check if all init costs are zero or default
        all_init_costs_zero = all((
            not (isinstance(self.tp_cost, np.ndarray) or self.tp_cost != 0.0),
            not (isinstance(self.tn_cost, np.ndarray) or self.tn_cost != 0.0),
            not (isinstance(self.fn_cost, np.ndarray) or self.fn_cost != 0.0),
            not (isinstance(self.fp_cost, np.ndarray) or self.fp_cost != 0.0),
        ))

        # Check if all fit costs are unchanged
        all_fit_costs_unchanged = all((
            tp_cost is Parameter.UNCHANGED,
            tn_cost is Parameter.UNCHANGED,
            fn_cost is Parameter.UNCHANGED,
            fp_cost is Parameter.UNCHANGED,
        ))

        # First condition: no costs provided and no loss function
        no_costs_no_loss = all_init_costs_zero and all_fit_costs_unchanged and self.loss is None

        # Second condition: loss exists but no loss-specific params provided
        loss_without_params = (
            self.loss is not None
            and not any(self.loss._all_symbols.intersection(params.keys()))
            and not self.loss.cost_matrix._defaults
        )

        return no_costs_no_loss or loss_without_params

    def _should_use_fitted_threshold(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        loss_params: dict[str, Any],
    ) -> bool:
        """
        Check if the threshold fitted during training should be used.

        Returns True if all of the following are true:
        1. All init costs are zero/default
        2. All predict costs are unchanged
        3. No loss parameters provided
        """
        all_init_costs_zero = all((
            not (isinstance(self.tp_cost, np.ndarray) or self.tp_cost != 0.0),
            not (isinstance(self.tn_cost, np.ndarray) or self.tn_cost != 0.0),
            not (isinstance(self.fn_cost, np.ndarray) or self.fn_cost != 0.0),
            not (isinstance(self.fp_cost, np.ndarray) or self.fp_cost != 0.0),
        ))

        all_predict_costs_unchanged = all((
            tp_cost is Parameter.UNCHANGED,
            tn_cost is Parameter.UNCHANGED,
            fn_cost is Parameter.UNCHANGED,
            fp_cost is Parameter.UNCHANGED,
        ))

        return (
            all_init_costs_zero
            and all_predict_costs_unchanged
            and not loss_params
            and not self.loss.cost_matrix._defaults
        )

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

    def _get_calibrator(self, estimator: Any) -> Any:
        if self.calibrator == 'sigmoid':
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(estimator, method='sigmoid', cv=cv, ensemble=False)
        elif self.calibrator == 'isotonic':
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            return CalibratedClassifierCV(estimator, method='isotonic', cv=cv, ensemble=False)
        else:
            return self.calibrator.set_params(estimator=estimator)  # type: ignore[union-attr]

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None

    @_fit_context(
        # *ThresholdClassifier*.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
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

        **params : dict
            Parameters to pass to the `fit` method of the underlying classifier.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # _raise_for_params(params, self, None)

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

        loss = self.loss if self.loss is not None else make_generic_cost_metric()

        loss_params = _extract_loss_params(params, loss)
        routing = self.estimator.get_metadata_routing()
        routing_params = {k: v for k, v in params.items() if k in routing.fit.requests}
        routed_params = process_routing(self, 'fit', **routing_params)
        if self.calibrator is not None:
            self.estimator_ = self._get_calibrator(self.estimator).fit(X, y, **routed_params.calibrator.fit)
        else:
            self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

        # convert instance-dependent costs to class-dependent so we only get a single threshold
        for key, value in list(loss_params.items()):
            loss_params[key] = _to_class_dependent_cost(value)

        y_score = self.estimator_.predict_proba(X)[:, 1]
        self.threshold_ = loss.optimal_threshold(y, y_score, **loss_params)

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
        if self._should_use_fitted_threshold(tp_cost, tn_cost, fn_cost, fp_cost, loss_params):
            check_is_fitted(self)
            optimal_thresholds = getattr(self, 'threshold_', None)
            if optimal_thresholds is None:
                raise ValueError(
                    'Optimal threshold has not been set during fit. Either provide costs/benefits to fit first '
                    'or provide costs to predict.'
                )
        else:
            if getattr(self, 'estimator_', None) is None:
                raise NotFittedError
            check_is_fitted(self.estimator_)

            loss = self.loss if self.loss is not None else make_generic_cost_metric()

            if isinstance(loss.strategy, MaxProfit):
                raise ValueError(f'Cannot use {loss.strategy.__class__.__name__} at predict time.')

            loss_params = self._add_standard_costs_to_params(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, params=loss_params
            )
            optimal_thresholds = loss.optimal_threshold(np.array([]), np.array([]), **loss_params)

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


class CSRateClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator, CostSensitiveMixin):  # type: ignore[misc]
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

    loss : Metric
        The cost-sensitive metric to optimize. Must implement `optimal_rate`.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        The class labels.

    estimator_ : Estimator
        The fitted classifier.

    rate_ : float
        The optimal positive rate determined during fitting.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'estimator': [HasMethods(['fit', 'predict_proba'])],
        'pos_label': [Real, str, 'boolean', None],
        'loss': [HasMethods('optimal_rate')],
    }

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        pos_label: int | str | bool | None = None,
        loss: Metric,
    ):
        self.estimator = estimator
        self.pos_label = pos_label
        self.loss = loss
        super().__init__()

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = False
        return tags

    @property
    def classes_(self) -> NDArray[Any]:  # noqa: D102
        if estimator := getattr(self, 'estimator_', None):
            return estimator.classes_  # type: ignore[no-any-return]
        try:
            check_is_fitted(self.estimator)
            return self.estimator.classes_  # type: ignore[no-any-return]
        except NotFittedError:
            raise AttributeError('The underlying estimator is not fitted yet.') from NotFittedError

    def fit(self, X: FloatArrayLike, y: ArrayLike, **params: Any) -> Self:
        """
        Fit the model and compute the optimal positive fraction rate.

        Parameters
        ----------
        X : array-like of shape (n_samples,n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **params : Any
            Additional parameters to customize the fitting process or to route to the
            underlying estimator or loss function. These can include hyperparameters or
            specific settings for estimator or loss adjustments.
        """
        X, y = validate_data(self, X, y)

        # Extract loss parameters before routing
        loss_params = {}
        loss_param_names = self.loss._all_symbols
        for param_name in list(params.keys()):
            if param_name in loss_param_names:
                loss_params[param_name] = params[param_name]

        # Remove loss-only params from routing (keep those needed by estimator)
        routing = self.estimator.get_metadata_routing()
        routing_params = {k: v for k, v in params.items() if k in routing.fit.requests}
        routed_params = process_routing(self, 'fit', **routing_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

        # Compute optimal rate using training predictions
        y_score = self.estimator_.predict_proba(X)[:, 1]
        self.rate_ = self.loss.optimal_rate(y, y_score, **loss_params)

        return self

    def predict(self, X: FloatArrayLike) -> NDArray[Any]:
        """
        Predict the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict`.

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        class_labels : ndarray of shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, ['estimator_', 'rate_'])

        y_score = self.estimator_.predict_proba(X)[:, 1]
        n_samples = len(y_score)
        n_positive = int(np.ceil(self.rate_ * n_samples))

        if n_positive == 0:
            return np.full(n_samples, self.classes_[0])
        if n_positive >= n_samples:
            return np.full(n_samples, self.classes_[1])

        threshold_idx = np.argsort(y_score)[-n_positive]
        threshold = y_score[threshold_idx]

        return np.where(y_score >= threshold, self.classes_[1], self.classes_[0])

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict the predicted probabilities of the target of new samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict_proba`.

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        class_probabilties : ndarray of shape (n_samples,)
            The predicted class probabilities.
        """
        check_is_fitted(self, ['estimator_', 'rate_'])
        y_proba: FloatNDArray = self.estimator_.predict_proba(X)
        return y_proba

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
        router.add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee='fit', caller='fit'),
        ).add_self_request(self)
        return router

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None
