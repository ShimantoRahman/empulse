from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Real
from typing import Any, ClassVar, Protocol, Self

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.special import expit
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, LogitObjective
from ...optimizers import Optimizer
from ..csclassifier import CostSensitiveClassifier


class OptimizeFnKwargs(Protocol):
    def __call__(
        self, objective: Callable[[FloatNDArray], float], X: FloatNDArray, **kwargs: Any
    ) -> OptimizeResult: ...


class OptimizeFnNoKwargs(Protocol):
    def __call__(self, objective: Callable[[FloatNDArray], float], X: FloatNDArray) -> OptimizeResult: ...


OptimizeFn = OptimizeFnKwargs | OptimizeFnNoKwargs | Callable[..., OptimizeResult]


class BaseLogitClassifier(CostSensitiveClassifier, ABC):  # type: ignore[misc]
    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'C': [Interval(Real, 0, None, closed='right')],
        'fit_intercept': ['boolean'],
        'soft_threshold': ['boolean'],
        'l1_ratio': [Interval(Real, 0, 1, closed='both')],
        'loss': [BaseMetric, None],
        'optimizer': [Optimizer, None],
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
        loss: BaseMetric | None = None,
        optimizer: Optimizer | None = None,
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.soft_threshold = soft_threshold
        self.l1_ratio = l1_ratio
        self.optimizer = optimizer
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    @abstractmethod
    def _optimize(self, objective: LogitObjective, X: FloatNDArray, **kwargs: Any) -> OptimizeResult:
        """
        Optimize the objective function.

        Subclasses should decide what the default optimizer is.
        If `optimize_fn` is provided, it should be used instead of the default optimizer.
        """

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        if self._get_metric_loss() is None:
            # `fit()` already checked/converted these costs; the logit objective additionally
            # needs instance-dependent costs as a column vector rather than a flat array.
            for key in ('tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'):
                value = loss_params[key]
                if not isinstance(value, Real) and (value := np.asarray(value)).ndim == 1:
                    loss_params[key] = np.expand_dims(value, axis=1)

        return self._fit_estimator(X, y, loss=loss, **loss_params)

    def _fit_estimator(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        objective = loss._logit_objective(
            features=X,
            y_true=y,
            C=self.C,
            l1_ratio=self.l1_ratio,
            soft_threshold=self.soft_threshold,
            fit_intercept=self.fit_intercept,
            **loss_params,
        )
        self.result_ = self._optimize(objective, X)

        if self.fit_intercept:
            self.intercept_ = self.result_.x[0]
            self.coef_ = self.result_.x[1:]
        else:
            self.coef_ = self.result_.x

        self.n_iter_ = self.result_.nit

        return self

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
