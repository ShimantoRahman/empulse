import warnings
from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, ClassVar, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, _fit_context
from sklearn.utils import Tags
from sklearn.utils._metadata_requests import RequestMethod
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

from .._common import Parameter
from .._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ..metrics import BaseMetric, Cost, MaxProfit, MetricStrategy
from ..metrics.metric.prebuilt_metrics import make_generic_metric


class MetricStrategyFactory(Protocol):
    """A factory for creating MetricStrategy instances."""

    def __call__(self, *args: Any, **kwargs: Any) -> MetricStrategy:
        """Instantiate a MetricStrategy object."""


class CostSensitiveClassifier(ABC, ClassifierMixin, BaseEstimator):
    """Base class for cost-sensitive classifiers."""

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'loss': [BaseMetric, None],
    }
    _default_metric_strategy: ClassVar[MetricStrategyFactory] = Cost
    _set_default_costs: ClassVar[bool] = True

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

    def __init__(
        self,
        *,
        tp_cost: FloatArrayLike | float,
        tn_cost: FloatArrayLike | float,
        fn_cost: FloatArrayLike | float,
        fp_cost: FloatArrayLike | float,
        loss: BaseMetric | None,
    ) -> None:
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
        self._append_params_to_metadata_routing()
        super().__init__()

    def _append_params_to_metadata_routing(self) -> None:
        # Allow passing costs accepted by the metric loss through metadata routing
        loss = self._get_metric_loss()
        if isinstance(loss, BaseMetric):
            self.__class__.set_fit_request = RequestMethod(  # type: ignore[attr-defined]
                'fit',
                sorted(self.get_metadata_routing().fit.requests.keys() | loss._all_symbols),  # type: ignore[attr-defined]
            )

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> Self:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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

        loss_params : Any
            Additional parameter to be passed to the loss function.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                f'Unknown label type: Only binary classification is supported. The type of the target is {y_type}.'
            )
        if not isinstance(self, MetaEstimatorMixin):
            self.classes_ = np.unique(y)
            if len(self.classes_) == 1:
                raise ValueError("Classifier can't train when only one class is present.")
            y = np.where(y == self.classes_[1], 1, 0)

        loss_ = self._get_metric_loss()
        if loss_ is None:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            loss_params['tp_cost'] = tp_cost
            loss_params['tn_cost'] = tn_cost
            loss_params['fn_cost'] = fn_cost
            loss_params['fp_cost'] = fp_cost
        else:
            loss_params = self._route_costs_to_loss(
                loss_,
                loss_params,
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
        loss_params = self._normalize_cost_shapes(loss_params, size=y.size)

        loss = loss_ if loss_ is not None else self._get_default_loss()

        return self._fit(X, y, loss=loss, **loss_params)

    @abstractmethod
    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self: ...

    def predict(self, X: FloatArrayLike) -> NDArray[Any]:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels for each sample.
        """
        y_proba = self.predict_proba(X)
        y_pred: NDArray[Any] = self.classes_[np.argmax(y_proba, axis=1)]
        return y_pred

    def _normalize_cost_shapes(self, loss_params: dict[str, Any], size: int) -> dict[str, Any]:
        for key, value in loss_params.items():
            if isinstance(value, np.ndarray):
                if value.size not in {1, size}:
                    raise ValueError(
                        f"Parameter '{key}' has length {value.size}, but expected length "
                        f'{size} (one value per sample, matching y) or a single value '
                        '(length 1, applied to every sample).'
                    )
                loss_params[key] = value.reshape(-1)
        return loss_params

    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """
        Check if costs are set and return them.

        Also convert them to numpy arrays if they are array-like.
        Overwrite costs set in constructor if they are set in the fit/predict method.
        """
        if tp_cost is Parameter.UNCHANGED:
            tp_cost = self.tp_cost  # type: ignore[attr-defined]
        if tn_cost is Parameter.UNCHANGED:
            tn_cost = self.tn_cost  # type: ignore[attr-defined]
        if fn_cost is Parameter.UNCHANGED:
            fn_cost = self.fn_cost  # type: ignore[attr-defined]
        if fp_cost is Parameter.UNCHANGED:
            fp_cost = self.fp_cost  # type: ignore[attr-defined]

        if self._set_default_costs and _all_costs_zero(tp_cost, tn_cost, fn_cost, fp_cost):
            warnings.warn(
                'All costs are zero. Setting fp_cost=1 and fn_cost=1. '
                f'To avoid this warning, set costs explicitly in the {self.__class__.__name__}.{caller}() method.',
                UserWarning,
                stacklevel=2,
            )
            fp_cost = 1
            fn_cost = 1

        if not isinstance(tp_cost, Real):
            tp_cost = np.asarray(tp_cost)
        if not isinstance(tn_cost, Real):
            tn_cost = np.asarray(tn_cost)
        if not isinstance(fn_cost, Real):
            fn_cost = np.asarray(fn_cost)
        if not isinstance(fp_cost, Real):
            fp_cost = np.asarray(fp_cost)

        return tp_cost, tn_cost, fn_cost, fp_cost  # type: ignore[return-value]

    def _route_costs_to_loss(
        self,
        loss: BaseMetric,
        params: dict[str, Any],
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
    ) -> dict[str, Any]:
        """
        Route explicitly passed cost arguments through to a :class:`~empulse.metrics.BaseMetric` loss.

        A cost matrix may legitimately name one of its symbols (or aliases) ``tp_cost``, ``tn_cost``,
        ``fn_cost`` or ``fp_cost`` -- several of the bundled datasets do. Those names collide with this
        method's own keyword parameters, so the value binds to the parameter instead of landing in
        ``**loss_params`` and would otherwise never reach the metric. Any such value that names a symbol
        of ``loss`` is forwarded here.

        Only values passed explicitly by the caller are forwarded. The ``__init__``-time cost attributes
        are deliberately *not* consulted: they describe plain costs and default to ``0.0``, so falling
        back to them would silently override a cost matrix default with zero.

        Parameters
        ----------
        loss : BaseMetric
            The metric loss the costs should be routed to.
        params : dict[str, Any]
            Loss parameters collected so far. Not mutated; an updated copy is returned.
        tp_cost, tn_cost, fn_cost, fp_cost : float or array-like or Parameter
            The cost arguments as passed by the caller. ``Parameter.UNCHANGED`` means "not passed".
        caller : str, default='fit'
            Name of the calling method, used in the warning message.

        Returns
        -------
        params : dict[str, Any]
            The loss parameters, extended with any cost argument that names a symbol of ``loss``.
        """
        params = dict(params)
        symbols = loss._all_symbols
        ignored = []
        for name, value in (
            ('tp_cost', tp_cost),
            ('tn_cost', tn_cost),
            ('fn_cost', fn_cost),
            ('fp_cost', fp_cost),
        ):
            if value is Parameter.UNCHANGED:
                continue
            if name in symbols:
                params[name] = value
            else:
                ignored.append(name)

        if ignored:
            warnings.warn(
                f'{", ".join(ignored)} passed to {self.__class__.__name__}.{caller}() '
                f'{"is" if len(ignored) == 1 else "are"} ignored because a `loss` metric is set '
                f'and the metric does not use {"that name" if len(ignored) == 1 else "those names"}. '
                f'Pass the parameters its cost matrix expects instead: {sorted(symbols)}.',
                UserWarning,
                stacklevel=3,
            )
        return params

    def _add_standard_costs_to_params(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        loss = self._get_metric_loss()
        if not isinstance(loss, BaseMetric):
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )

            params['tp_cost'] = tp_cost
            params['tn_cost'] = tn_cost
            params['fn_cost'] = fn_cost
            params['fp_cost'] = fp_cost
        else:
            params = self._route_costs_to_loss(
                loss,
                params,
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
                caller='predict',
            )
        return params

    def _get_metric_loss(self) -> BaseMetric | None:
        """Get the metric loss function if available."""
        return getattr(self, 'loss', None)

    def _get_default_loss(self) -> BaseMetric:
        return make_generic_metric(self._default_metric_strategy())

    def _prepare_class_costs(self, loss_params: dict[str, Any]) -> tuple[float, float, float, float]:
        """
        Reduce costs/benefits to four class-dependent scalars: tp_benefit, tn_benefit, fp_cost, fn_cost.

        If ``loss`` is a :class:`~empulse.metrics.BaseMetric` (e.g. a :class:`~empulse.metrics.Metric`
        or :class:`~empulse.metrics.MixtureMetric`) built with the :class:`~empulse.metrics.MaxProfit`
        strategy, any stochastic (random) variables in the cost/benefit expressions are first replaced
        by their mean so that a scalar value can be derived from them. Instance-dependent (array-like)
        costs are aggregated by taking their mean.

        Parameters
        ----------
        loss_params : dict[str, Any]
            Parameters to pass to the loss function, or plain ``tp_cost``/``tn_cost``/``fn_cost``/``fp_cost``
            values when no :class:`~empulse.metrics.BaseMetric` loss is set.

        Returns
        -------
        tp_benefit : float
            The (class-dependent) benefit of a true positive.
        tn_benefit : float
            The (class-dependent) benefit of a true negative.
        fp_cost : float
            The (class-dependent) cost of a false positive.
        fn_cost : float
            The (class-dependent) cost of a false negative.
        """
        loss_ = self._get_metric_loss()
        if loss_ is None:
            tp_cost = loss_params.get('tp_cost', 0.0)
            tn_cost = loss_params.get('tn_cost', 0.0)
            fn_cost = loss_params.get('fn_cost', 0.0)
            fp_cost = loss_params.get('fp_cost', 0.0)
        elif isinstance(loss_, BaseMetric):
            if isinstance(loss_.strategy, MaxProfit):
                fp_cost, fn_cost, tp_cost, tn_cost = loss_._evaluate_costs(replace_stochastic=True, **loss_params)
            else:
                raise ValueError(
                    f'{self.__class__.__name__} only supports losses built with the MaxProfit strategy, got {loss_}.'
                )
        else:
            raise ValueError(f'Unknown loss function: {loss_}.')

        # This model requires scalar (class-dependent) costs, so instance-dependent
        # (array-like) costs are aggregated to their mean value.
        tp_benefit = -float(np.mean(tp_cost))
        tn_benefit = -float(np.mean(tn_cost))
        fp_cost = float(np.mean(fp_cost))
        fn_cost = float(np.mean(fn_cost))
        return tp_benefit, tn_benefit, fp_cost, fn_cost


def _all_float(*arrays: ArrayLike | float | Parameter) -> bool:
    return all(isinstance(array, Real) and not isinstance(array, Parameter) for array in arrays)


def _all_costs_zero(
    tp_cost: FloatArrayLike | float | Parameter,
    tn_cost: FloatArrayLike | float | Parameter,
    fn_cost: FloatArrayLike | float | Parameter,
    fp_cost: FloatArrayLike | float | Parameter,
) -> bool:
    return (
        _all_float(tp_cost, tn_cost, fn_cost, fp_cost)
        and sum(abs(cost) for cost in (tp_cost, tn_cost, fn_cost, fp_cost)) == 0.0  # type: ignore[misc, arg-type]
    )
