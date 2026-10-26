from abc import ABC
from numbers import Integral, Real
from typing import Any, ClassVar, Self

import numpy as np
from joblib import effective_n_jobs
from scipy.special import expit
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ..._common._sklearn_compat import validate_data
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, Capability
from ...optimizers import Optimizer
from .cost_sensitive import CostSensitiveClassifier


class BaseLogitClassifier(CostSensitiveClassifier, ABC):  # type: ignore[misc]
    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'C': [Interval(Real, 0, None, closed='right')],
        'fit_intercept': ['boolean'],
        'l1_ratio': [Interval(Real, 0, 1, closed='both')],
        'loss': [BaseMetric, None],
        'optimizer': [Optimizer, None],
        'n_jobs': [Interval(Integral, 1, None, closed='left'), Interval(Integral, None, -1, closed='right'), None],
    }
    _default_optimizer: ClassVar[type[Optimizer]]

    def __init__(
        self,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        C: float = 1.0,
        fit_intercept: bool = True,
        l1_ratio: float = 1.0,
        loss: BaseMetric | None = None,
        optimizer: Optimizer | None = None,
        n_jobs: int | None = 1,
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.l1_ratio = l1_ratio
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def _resolve_optimizer(self) -> Optimizer:
        """Return `self.optimizer`, or a new `_default_optimizer` if unset."""
        return self._default_optimizer() if self.optimizer is None else self.optimizer

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self._fit_estimator(X, y, loss=loss, **loss_params)

    def _fit_estimator(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        optimizer = self._resolve_optimizer()
        requester = f'{type(self).__name__} with {type(optimizer).__name__}'
        if optimizer.requires_gradient:
            loss._require(
                Capability.LOGIT_OBJECTIVE,
                requester=requester,
                hint=(
                    'Use an optimizer that needs only the loss value, such as GeneticAlgorithmOptimizer.'
                    if Capability.LOGIT_VALUE_OBJECTIVE in loss.capabilities
                    else None
                ),
            )
            build_objective = loss._logit_objective
        else:
            # An optimizer that only compares losses gets an objective that computes the loss alone.
            loss._require(Capability.LOGIT_VALUE_OBJECTIVE, requester=requester)
            build_objective = loss._logit_value_objective
        objective = build_objective(
            features=X,
            y_true=y,
            C=self.C,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            **loss_params,
        )
        # An optimizer that evaluates the objective from several threads of its own, like the genetic
        # algorithm with `n_jobs > 1`, gets one thread per evaluation, so the two do not multiply.
        optimizer_jobs = effective_n_jobs(getattr(optimizer, 'n_jobs', 1))
        objective.n_threads = 1 if optimizer_jobs > 1 else effective_n_jobs(self.n_jobs)
        self.result_ = optimizer(objective=objective, X=X)

        if self.fit_intercept:
            self.intercept_ = self.result_.x[0]
            self.coef_ = self.result_.x[1:]
        else:
            self.coef_ = self.result_.x
            if 'intercept_' in vars(self):
                del self.intercept_

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

        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = self.result_.x
        logits = np.dot(X, theta)
        y_pred = expit(logits)
        y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred
