from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult
from scipy.special import expit
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import Metric
from ...utils._sklearn_compat import validate_data  # type: ignore[attr-defined]
from ..csclassifier import CostSensitiveClassifier


class OptimizeFnKwargs(Protocol):
    def __call__(
        self, objective: Callable[[FloatNDArray], float], X: FloatNDArray, **kwargs: Any
    ) -> OptimizeResult: ...


class OptimizeFnNoKwargs(Protocol):
    def __call__(self, objective: Callable[[FloatNDArray], float], X: FloatNDArray) -> OptimizeResult: ...


OptimizeFn = OptimizeFnKwargs | OptimizeFnNoKwargs | Callable[..., OptimizeResult]


class LossFnKwargs(Protocol):
    def __call__(self, y_true: IntNDArray, y_proba: FloatNDArray, **kwargs: Any) -> float: ...


class LossFnNoKwargs(Protocol):
    def __call__(self, y_true: IntNDArray, y_proba: FloatNDArray) -> float: ...


LossFn = LossFnKwargs | LossFnNoKwargs | Callable[..., float]


class BaseLogitClassifier(CostSensitiveClassifier, ABC):  # type: ignore[misc]
    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'n_jobs': [None, Integral],
        'C': [Interval(Real, 0, None, closed='right')],
        'fit_intercept': ['boolean'],
        'soft_threshold': ['boolean'],
        'optimize_fn': [callable, None],
        'l1_ratio': [Interval(Real, 0, 1, closed='both')],
        'optimizer_params': [dict, None],
    }

    def __init__(
        self,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        C: float = 1.0,
        fit_intercept: bool = True,
        soft_threshold: bool = True,
        l1_ratio: float = 1.0,
        loss: str | LossFn | Metric | None = None,
        optimize_fn: OptimizeFn | None = None,
        optimizer_params: dict[str, Any] | None = None,
    ):
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.C = C
        self.fit_intercept = fit_intercept
        self.soft_threshold = soft_threshold
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.optimizer_params = optimizer_params
        self.optimize_fn = optimize_fn
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def _fit(self, X: FloatArrayLike, y: ArrayLike, loss: Metric, **loss_params: Any) -> Self:
        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        if self.loss is None:
            loss_params = self._validate_costs(**loss_params)

        return self._fit_estimator(X, y, loss=loss, **loss_params)

    @abstractmethod
    def _fit_estimator(self, X: FloatNDArray, y: NDArray[Any], loss: Metric, **loss_params: Any) -> Self: ...

    def _validate_costs(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        **loss_params: Any,
    ) -> dict[str, Any]:
        if not isinstance(self.loss, Metric):
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )

            if not isinstance(tp_cost, Real) and (tp_cost := np.asarray(tp_cost)).ndim == 1:
                tp_cost = np.expand_dims(tp_cost, axis=1)
            if not isinstance(tn_cost, Real) and (tn_cost := np.asarray(tn_cost)).ndim == 1:
                tn_cost = np.expand_dims(tn_cost, axis=1)
            if not isinstance(fn_cost, Real) and (fn_cost := np.asarray(fn_cost)).ndim == 1:
                fn_cost = np.expand_dims(fn_cost, axis=1)
            if not isinstance(fp_cost, Real) and (fp_cost := np.asarray(fp_cost)).ndim == 1:
                fp_cost = np.expand_dims(fp_cost, axis=1)

            # Assume that the loss function takes the following parameters:
            loss_params['tp_cost'] = tp_cost
            loss_params['tn_cost'] = tn_cost
            loss_params['fn_cost'] = fn_cost
            loss_params['fp_cost'] = fp_cost
        return loss_params

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, 2)
            Predicted probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = self.result_.x
        logits = np.dot(X, theta)
        y_pred = expit(logits)
        # create 2D array with complementary probabilities
        y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred
