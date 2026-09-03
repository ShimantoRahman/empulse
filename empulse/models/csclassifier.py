import warnings
from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, ClassVar, Literal, Protocol, Self, overload

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
    _cost_ndim: ClassVar[int] = 0
    _array_cost_ndim: ClassVar[int] = 1
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
            if self._cost_ndim > 0:
                tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                    force_array=True,
                    n_samples=int(X.shape[0]),
                )
            else:
                tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                )
            loss_params = self._add_standard_costs_to_params(
                tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, params=loss_params
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
        shape = -1 if self._array_cost_ndim == 1 else (1, -1)

        for key, value in loss_params.items():
            if isinstance(value, np.ndarray):
                if value.size != size:
                    raise ValueError(f'The size of the cost parameter {key} must be {size}, but got {value.size}.')
                loss_params[key] = value.reshape(shape)
        return loss_params

    @overload
    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
        force_array: Literal[True] = True,
        n_samples: int,
    ) -> tuple[
        FloatNDArray,
        FloatNDArray,
        FloatNDArray,
        FloatNDArray,
    ]: ...

    @overload
    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
        force_array: Literal[False] = False,
        n_samples: int | None = None,
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]: ...

    def _check_costs(
        self,
        *,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        caller: str = 'fit',
        force_array: bool = False,
        n_samples: int | None = None,
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

        if force_array:
            if n_samples is None:
                raise ValueError('n_samples should be set when force_array is True.')
            tp_cost = np.asarray(tp_cost) if not isinstance(tp_cost, Real) else np.full(n_samples, tp_cost)
            tn_cost = np.asarray(tn_cost) if not isinstance(tn_cost, Real) else np.full(n_samples, tn_cost)
            fn_cost = np.asarray(fn_cost) if not isinstance(fn_cost, Real) else np.full(n_samples, fn_cost)
            fp_cost = np.asarray(fp_cost) if not isinstance(fp_cost, Real) else np.full(n_samples, fp_cost)
        else:
            if not isinstance(tp_cost, Real):
                tp_cost = np.asarray(tp_cost)
            if not isinstance(tn_cost, Real):
                tn_cost = np.asarray(tn_cost)
            if not isinstance(fn_cost, Real):
                fn_cost = np.asarray(fn_cost)
            if not isinstance(fp_cost, Real):
                fp_cost = np.asarray(fp_cost)

        return tp_cost, tn_cost, fn_cost, fp_cost  # type: ignore[return-value]

    def _add_standard_costs_to_params(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(self._get_metric_loss(), BaseMetric):
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )

            params['tp_cost'] = tp_cost
            params['tn_cost'] = tn_cost
            params['fn_cost'] = fn_cost
            params['fp_cost'] = fp_cost
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
        if self.loss is None:
            tp_cost = loss_params.get('tp_cost', 0.0)
            tn_cost = loss_params.get('tn_cost', 0.0)
            fn_cost = loss_params.get('fn_cost', 0.0)
            fp_cost = loss_params.get('fp_cost', 0.0)
        elif isinstance(self.loss, BaseMetric):
            if isinstance(self.loss.strategy, MaxProfit):
                fp_cost, fn_cost, tp_cost, tn_cost = self.loss._evaluate_costs(replace_stochastic=True, **loss_params)
            else:
                raise ValueError(
                    f'{self.__class__.__name__} only supports losses built with the '
                    f'MaxProfit strategy, got {self.loss}.'
                )
        else:
            raise ValueError(f'Unknown loss function: {self.loss}.')

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
