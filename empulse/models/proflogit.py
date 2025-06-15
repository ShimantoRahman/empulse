import sys
from collections.abc import Callable
from functools import partial
from itertools import islice
from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult
from scipy.special import expit

from empulse.optimizers import Generation

from .._types import FloatNDArray, IntNDArray, ParameterConstraint
from ..metrics import Metric, empc_score
from ..metrics.metric.common import Direction
from ._base import BaseLogitClassifier, LossFn, OptimizeFn

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class ProfLogitClassifier(BaseLogitClassifier):
    """
    Logistic classifier to optimize profit-driven score.

    Maximizing empirical EMP for churn by optimizing
    the regression coefficients of the logistic model through
    a Real-coded Genetic Algorithm (RGA).

    Read more in the :ref:`User Guide <proflogit>`.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive ``float``.
        Like in support vector machines, smaller values specify stronger regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    soft_threshold : bool, default=False
        If ``True``, apply soft-thresholding to the regression coefficients.

    l1_ratio : float, default=1.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is a L2 penalty.
        For ``l1_ratio = 1`` it is a L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    loss : Callable or :class:`empulse.metrics.Metric`, default= :func:`empulse.metrics.empc_score`
        Loss function to optimize.

        - If ``Callable`` it should have a signature ``loss(y_true, y_score)``.

        - If :class`~empulse.metrics.Metric`, metric parameters are passed as ``loss_params``
          to the :Meth:`~empulse.models.ProfLogitClassifier.fit` method.

        By default, loss function is maximized, customize behaviour in `optimize_fn`.
        If the loss function in an instance of :class:`~empulse.metrics.Metric` then the optimization direction is
        automatically determined.

    optimize_fn : Callable, optional
        Optimization algorithm. Should be a Callable with signature ``optimize(objective, X)``.
        See :ref:`proflogit` for more information.

    optimizer_params : dict[str, Any], optional
        Additional keyword arguments passed to `optimize_fn`.

        By default, the optimizer is a Real-coded Genetic Algorithm (RGA) with the following parameters:

        - ``max_iter`` : int, default=1000
            Maximum number of iterations.
        - ``patience`` : int, default=250
            Number of iterations with no improvement to wait before stopping the optimization.
        - ``tolerance`` : float, default=1e-4
            Relative tolerance to declare convergence.
        - ``bounds`` : tuple[float, float], default=(-5, 5)
            Lower and upper bounds for the regression coefficients.
        - all other parameters are passed to the :class:`~empulse.optimizers.Generation` initializer.

    n_jobs : int, optional
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    coef_ : numpy.ndarray
        Coefficients of the logit model.

    intercept_ : float
        Intercept of the logit model.
        Only available when ``fit_intercept=True``.

    References
    ----------
    .. [1] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2017). Profit Maximizing Logistic Model for
        Customer Churn Prediction Using Genetic Algorithms.
        Swarm and Evolutionary Computation.
    .. [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2015). Profit Maximizing Logistic Regression Modeling for
        Customer Churn Prediction. IEEE International Conference on
        Data Science and Advanced Analytics (DSAA) (pp. 1–10). Paris, France.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **BaseLogitClassifier._parameter_constraints,
        'loss': [callable, None],
        'n_jobs': [None, Integral],
    }

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        soft_threshold: bool = False,
        l1_ratio: float = 1.0,
        loss: LossFn | Metric = empc_score,
        optimize_fn: OptimizeFn | None = None,
        optimizer_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            C=C,
            fit_intercept=fit_intercept,
            soft_threshold=soft_threshold,
            l1_ratio=l1_ratio,
            loss=loss,
            optimize_fn=optimize_fn,
            optimizer_params=optimizer_params,
        )
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike, **loss_params: Any) -> Self:
        """
        Fit ProfLogit model.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Training data.
        y : 1D array-like, shape=(n_samples,)
            Target values.
        loss_params : dict
            Additional keyword arguments passed to `loss`.

        Returns
        -------
        self : ProfLogitClassifier
            Fitted ProfLogit model.
        """
        super().fit(X, y, **loss_params)
        return self

    def _fit(self, X: FloatNDArray, y: IntNDArray, **loss_params: Any) -> Self:
        optimizer_params = {} if self.optimizer_params is None else self.optimizer_params.copy()
        optimize_fn: OptimizeFn = _optimize if self.optimize_fn is None else self.optimize_fn
        optimize_fn = partial(optimize_fn, **optimizer_params)

        if isinstance(self.loss, str) or self.loss is None:
            raise ValueError('Loss function must be a Callable or an instance of the Metric class.')
        if isinstance(self.loss, Metric) and self.loss.direction == Direction.MINIMIZE:
            loss_fn = lambda *args, **kwargs: -self.loss(*args, **kwargs)
        else:
            loss_fn = self.loss
        objective = partial(
            _objective,
            X=X,
            y=y,
            loss_fn=partial(loss_fn, **loss_params),
            C=self.C,
            l1_ratio=self.l1_ratio,
            soft_threshold=self.soft_threshold,
            fit_intercept=self.fit_intercept,
        )
        self.result_ = optimize_fn(objective, X)

        if self.fit_intercept:
            self.intercept_ = self.result_.x[0]
            self.coef_ = self.result_.x[1:]
        else:
            self.coef_ = self.result_.x

        return self


def _objective(
    weights: FloatNDArray,
    X: FloatNDArray,
    y: IntNDArray,
    loss_fn: LossFn,
    C: float,
    l1_ratio: float,
    soft_threshold: bool,
    fit_intercept: bool,
) -> float:
    """ProfLogit's objective function (maximization problem)."""
    # b is the vector holding the regression coefficients (no intercept)
    b = weights.copy()[1:] if fit_intercept else weights

    if soft_threshold:
        bool_nonzero = (np.abs(b) - C) > 0
        if np.sum(bool_nonzero) > 0:
            b[bool_nonzero] = np.sign(b[bool_nonzero]) * (np.abs(b[bool_nonzero]) - C)
        if np.sum(~bool_nonzero) > 0:
            b[~bool_nonzero] = 0

    logits = np.dot(X, weights)
    y_pred = expit(logits)  # Invert logit transformation
    loss = loss_fn(y, y_pred)
    regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b**2) + l1_ratio * np.sum(np.abs(b))
    penalty = regularization_term / C
    return float(loss - penalty)


def _optimize(
    objective: Callable[[FloatNDArray], float],
    X: FloatNDArray,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 250,
    bounds: tuple[float | int, float | int] = (-5, 5),
    **kwargs: Any,
) -> OptimizeResult:
    rga = Generation(**kwargs)
    previous_score = np.inf
    iter_stagnant = 0
    bounds_per_instance = [bounds] * X.shape[1]

    for _ in islice(rga.optimize(objective, bounds_per_instance), max_iter):
        score = rga.result.fun
        relative_improvement = (score - previous_score) / previous_score if previous_score != np.inf else np.inf
        previous_score = score
        if relative_improvement < tolerance:
            if (iter_stagnant := iter_stagnant + 1) >= patience:
                rga.result.message = 'Converged.'
                rga.result.success = True
                break
        else:
            iter_stagnant = 0
    else:
        rga.result.message = 'Maximum number of iterations reached.'
        rga.result.success = False
    return rga.result
