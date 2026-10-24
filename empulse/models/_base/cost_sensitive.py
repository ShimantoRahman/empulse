from abc import ABC, abstractmethod
from collections.abc import Collection
from numbers import Real
from typing import Any, ClassVar, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, _fit_context
from sklearn.utils.validation import column_or_1d, indexable

from ..._common import Parameter
from ..._common._cost_routing import RoutesLossParameters
from ..._common._sklearn_compat import get_tags, validate_data
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, Capability, Cost, MetricStrategy
from ...metrics.metric.prebuilt_metrics import make_generic_metric
from .wrapper import BinaryClassifierTags, check_binary_target


def subset_loss_params(loss_params: dict[str, Any], index: Any, n_samples: int) -> dict[str, Any]:
    """
    Subset every instance-dependent (array-valued) entry of *loss_params* by *index*.

    Mirrors how ``y`` and ``y_score`` are subset to some of the rows (an ensemble member's out-of-bag
    rows, or the rows a symbolic regression program is scored on) before being passed to the loss: an
    array-valued parameter of length *n_samples* describes one value per training sample and must be
    subset the same way, or the metric would evaluate it against the wrong rows (and, for a length
    that no longer matches the subset, raise a shape-mismatch error). Class-dependent
    (scalar) parameters, and arrays not aligned with the full training set, pass through unchanged.

    Parameters
    ----------
    loss_params : dict
        Keyword arguments that will be passed to the loss/metric.
    index : ndarray of int or bool
        The row selector (indices or boolean mask) to apply.
    n_samples : int
        The number of samples in the full training set (``y.shape[0]`` at fit time).

    Returns
    -------
    subset : dict
        *loss_params* with every length-``n_samples`` array entry subset by *index*.
    """
    subset: dict[str, Any] = {}
    for key, value in loss_params.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_samples:
            subset[key] = value[index]
        else:
            subset[key] = value
    return subset


class MetricStrategyFactory(Protocol):
    """A factory for creating MetricStrategy instances."""

    def __call__(self, *args: Any, **kwargs: Any) -> MetricStrategy:
        """Instantiate a MetricStrategy object."""


class CostSensitiveClassifier(RoutesLossParameters, BinaryClassifierTags, ABC, ClassifierMixin, BaseEstimator):
    """Base class for cost-sensitive classifiers."""

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'loss': [BaseMetric, None],
    }
    _default_metric_strategy: ClassVar[MetricStrategyFactory] = Cost
    # Keyword arguments of ``fit`` that ``_fit`` consumes besides the loss's parameters. ``None``
    # means ``_fit`` forwards any keyword argument, so none is rejected.
    _extra_fit_params: ClassVar[frozenset[str] | None] = frozenset()

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
        super().__init__()

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
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

        tn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of true negatives. If ``float``, then all true negatives have the same cost.
            If array-like, then it is the cost of each true negative classification.

        fn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        fp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        **loss_params : Any
            Additional parameter to be passed to the loss function.

        Returns
        -------
        self
            Fitted estimator.
        """
        X_checked: Any
        y_checked: Any
        if isinstance(self, MetaEstimatorMixin):
            # X is the wrapped estimator's to validate: it may be a pipeline that encodes strings,
            # handles missing values, or needs the feature names.
            X_checked, y_checked = indexable(X, y)
            y_checked = column_or_1d(y_checked, warn=True)
        else:
            X_checked, y_checked = validate_data(
                self, X, y, ensure_all_finite='allow-nan' if get_tags(self).input_tags.allow_nan else True
            )
        X, y = X_checked, y_checked
        check_binary_target(y)
        if not isinstance(self, MetaEstimatorMixin):
            self.classes_ = np.unique(y)
            if len(self.classes_) == 1:
                raise ValueError("Classifier can't train when only one class is present.")
            y = np.where(y == self.classes_[1], 1, 0)

        self._take_fit_local_loss()
        loss_ = self._get_metric_loss()
        if loss_ is None:
            loss_params.update(
                self._check_costs(
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                )
            )
        else:
            loss_params = self._route_costs_to_loss(
                loss_,
                loss_params,
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
        loss = loss_ if loss_ is not None else self._get_default_loss()

        known = loss._all_parameters
        loss_params = self._normalize_cost_shapes(loss_params, size=y.size, names=known)
        if self._extra_fit_params is not None:
            unknown = sorted(set(loss_params) - known - self._extra_fit_params)
            if unknown:
                raise TypeError(
                    f'{type(self).__name__}.fit() got unexpected keyword argument(s) {", ".join(map(repr, unknown))}. '
                    f'Besides X and y it accepts the costs and the parameters of its loss: '
                    f'{", ".join(sorted(known | self._extra_fit_params))}.'
                )
        loss._validate_parameters(**{name: value for name, value in loss_params.items() if name in known})

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

    def _normalize_cost_shapes(self, loss_params: dict[str, Any], size: int, names: Collection[str]) -> dict[str, Any]:
        """Flatten the arrays among the parameters in *names*, checking each has one value or one per sample."""
        for key, value in loss_params.items():
            if key in names and isinstance(value, np.ndarray):
                if value.size not in {1, size}:
                    raise ValueError(
                        f"Parameter '{key}' has length {value.size}, but expected length "
                        f'{size} (one value per sample) or a single value '
                        '(length 1, applied to every sample).'
                    )
                loss_params[key] = value.reshape(-1)
        return loss_params

    def _add_standard_costs_to_params(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        loss = self._get_metric_loss()
        if loss is None:
            params = dict(params)
            params.update(
                self._check_costs(tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, caller='predict')
            )
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
        else:
            loss_._require(Capability.CLASS_COSTS, requester=self.__class__.__name__)
            fp_cost, fn_cost, tp_cost, tn_cost = loss_._evaluate_costs(replace_stochastic=True, **loss_params)

        # This model requires scalar (class-dependent) costs, so instance-dependent
        # (array-like) costs are aggregated to their mean value.
        tp_benefit = -float(np.mean(tp_cost))
        tn_benefit = -float(np.mean(tn_cost))
        fp_cost = float(np.mean(fp_cost))
        fn_cost = float(np.mean(fn_cost))
        return tp_benefit, tn_benefit, fp_cost, fn_cost
