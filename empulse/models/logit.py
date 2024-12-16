import inspect
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Any, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_X_y, check_array, check_random_state
from scipy.optimize import minimize, OptimizeResult
from numpy.random import default_rng, Generator, RandomState


class BaseLogitClassifier(ABC, ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            C: float = 1.0,
            fit_intercept: bool = True,
            soft_threshold: bool = True,
            l1_ratio: float = 1.0,
            loss_fn: Callable = None,
            optimize_fn: Optional[Callable] = None,
            default_bounds: tuple[float, float] = (-3, 3),
            n_jobs: Optional[int] = None,
            optimizer_params: Optional[dict[str, Any]] = None,
            random_state: Optional[Union[int, RandomState]] = None,
            **kwargs,
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.soft_threshold = soft_threshold
        self.l1_ratio = l1_ratio
        self.loss_fn = loss_fn
        self.n_jobs = n_jobs
        self.default_bounds = default_bounds
        self.random_state = check_random_state(random_state)
        self.result = None

        # necessary to have optimizer_params because sklearn.clone does not clone **kwargs
        if optimizer_params is None:
            optimizer_params = {}
        if kwargs:
            optimizer_params.update(kwargs)
        self.optimizer_params = optimizer_params
        if optimize_fn is None:
            optimize_fn = _optimize
        self.optimize_fn = optimize_fn

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
            Features.

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, 2)
            Predicted probabilities.
        """
        X = check_array(X)
        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        theta = self.result.x
        logits = np.dot(X, theta)
        y_pred = 1 / (1 + np.exp(-logits))

        y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predicted labels.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
            Features.

        Returns
        -------
        y_pred : 1D numpy.ndarray, shape=(n_samples,)
            Predicted labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)


def _optimize(objective, bounds, max_iter=10000, tolerance=1e-4, rng: Generator = default_rng(42),
              **kwargs) -> OptimizeResult:
    # initial_weights = np.zeros(len(bounds), order="F")
    initial_weights = rng.random(len(bounds))

    print(max_iter, tolerance, kwargs)
    result = minimize(
        objective,
        initial_weights,
        method="L-BFGS-B",
        # jac=True,  TODO: allow loss function to return a tuple of loss and gradient
        options={
            "maxiter": max_iter,
            "maxls": 50,
            "gtol": tolerance,
            "ftol": 64 * np.finfo(float).eps,
        },
        **kwargs,
    )
    _check_optimize_result(result)

    return result


def _check_optimize_result(result):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    result : OptimizeResult
       Result of the scipy.optimize.minimize function.

    max_iter : int, default=None
       Expected maximum number of iterations.
    """
    # handle both scipy and scikit-learn solver names
    if result.status != 0:
        try:
            # The message is already decoded in scipy>=1.6.0
            result_message = result.message.decode("latin1")
        except AttributeError:
            result_message = result.message
        warning_msg = (
            "L-BFGS failed to converge (status={}):\n{}.\n\n"
            "Increase the number of iterations (max_iter) "
            "or scale the data as shown in:\n"
            "    https://scikit-learn.org/stable/modules/"
            "preprocessing.html"
        ).format(result.status, result_message)
        warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
