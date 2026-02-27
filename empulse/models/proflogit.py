from collections.abc import Callable
from functools import partial
from itertools import islice
from numbers import Integral, Real
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult
from scipy.special import expit

from empulse.optimizers import Generation

from .._common import Parameter
from .._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ..metrics import Metric
from ..metrics.metric.common import Direction
from ..metrics.metric.prebuilt_metrics import make_generic_max_profit_metric
from ._base import BaseLogitClassifier, LossFn, OptimizeFn


class ProfLogitClassifier(BaseLogitClassifier):
    """
    Profit-driven logistic regression classifier.

    Maximizing empirical cost-sensitive/value-driven metric
    by optimizing the regression coefficients of the logistic model through a Real-coded Genetic Algorithm (RGA).

    Read more in the :ref:`User Guide <proflogit>`.

    Parameters
    ----------
    tp_benefit : float or array-like, shape=(n_samples,), default=0.0
        Benefit of true positives. If ``float``, then all true positives have the same benefit.
        If array-like, then it is the benefit of each true positive classification.
        Is overwritten if another `tp_benefit` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent benefits to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_benefit : float or array-like, shape=(n_samples,), default=0.0
        Benefit of true negatives. If ``float``, then all true negatives have the same benefit.
        If array-like, then it is the benefit of each true negative classification.
        Is overwritten if another `tn_benefit` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent benefits to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    loss : :class:`empulse.metrics.Metric` or None, default=None
        Loss function to optimize.

        If :class`~empulse.metrics.Metric`, metric parameters are passed as ``loss_params``
        to the :Meth:`~empulse.models.ProfLogitClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

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

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfLogitClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()

        model = ProfLogitClassifier(C=0.1, l1_ratio=0.5)
        model.fit(X, y, tp_benefit=200, incentive_cost=10)

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
        'loss': [Metric, None],
        'n_jobs': [None, Integral],
    }

    def __init__(
        self,
        *,
        tp_benefit: FloatArrayLike | float = 0.0,
        tn_benefit: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
        C: float = 1.0,
        fit_intercept: bool = True,
        soft_threshold: bool = False,
        l1_ratio: float = 1.0,
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
        self.tp_benefit = tp_benefit
        self.tn_benefit = tn_benefit
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.n_jobs = n_jobs

    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        tp_benefit: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_benefit: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> Self:
        """
        Fit ProfLogit model.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Training data.

        y : 1D array-like, shape=(n_samples,)
            Target values.

        tp_benefit : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Benefit of true positives. If ``float``, then all true positives have the same benefit.
            If array-like, then it is the benefit of each true positive classification.

        tn_benefit : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Benefit of true negatives. If ``float``, then all true negatives have the same benefit.
            If array-like, then it is the benefit of each true negative classification.

        fp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        fn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        loss_params : dict
            Additional parameters passed to the loss function.

        Returns
        -------
        self : ProfLogitClassifier
            Fitted ProfLogit model.
        """
        super().fit(X, y, tp_cost=tp_benefit, tn_cost=tn_benefit, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params)
        return self

    def _fit(self, X: FloatNDArray, y: IntNDArray, **loss_params: Any) -> Self:
        optimizer_params = {} if self.optimizer_params is None else self.optimizer_params.copy()
        optimize_fn: OptimizeFn = _optimize if self.optimize_fn is None else self.optimize_fn
        optimize_fn = partial(optimize_fn, **optimizer_params)

        loss = self.loss if isinstance(self.loss, Metric) else make_generic_max_profit_metric()
        if isinstance(loss, Metric) and loss.direction == Direction.MINIMIZE:
            loss = lambda *args, **kwargs: -loss(*args, **kwargs)

        objective = partial(
            _objective,
            X=X,
            y=y,
            loss_fn=partial(loss, **loss_params),
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

    def _validate_costs(
        self,
        tp_cost: FloatArrayLike | float | Parameter,
        tn_cost: FloatArrayLike | float | Parameter,
        fn_cost: FloatArrayLike | float | Parameter,
        fp_cost: FloatArrayLike | float | Parameter,
        **loss_params: Any,
    ) -> dict[str, Any]:
        if not isinstance(self.loss, Metric):
            tp_benefit, tn_benefit, fn_cost, fp_cost = self._check_cost_benefits(
                tp_benefit=tp_cost, tn_benefit=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )

            if not isinstance(tp_benefit, Real) and (tp_benefit := np.asarray(tp_benefit)).ndim == 1:
                tp_benefit = np.expand_dims(tp_benefit, axis=1)
            if not isinstance(tn_benefit, Real) and (tn_cost := np.asarray(tn_benefit)).ndim == 1:
                tn_benefit = np.expand_dims(tn_cost, axis=1)
            if not isinstance(fn_cost, Real) and (fn_cost := np.asarray(fn_cost)).ndim == 1:
                fn_cost = np.expand_dims(fn_cost, axis=1)
            if not isinstance(fp_cost, Real) and (fp_cost := np.asarray(fp_cost)).ndim == 1:
                fp_cost = np.expand_dims(fp_cost, axis=1)

            # Assume that the loss function takes the following parameters:
            loss_params['tp_cost'] = -tp_benefit
            loss_params['tn_cost'] = -tn_benefit
            loss_params['fn_cost'] = fn_cost
            loss_params['fp_cost'] = fp_cost
        return loss_params


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
        threshold = l1_ratio / C
        b = np.sign(b) * np.maximum(np.abs(b) - threshold, 0)

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
                rga.result.message = 'Converged.'  # type: ignore[attr-defined]
                rga.result.success = True  # type: ignore[attr-defined]
                break
        else:
            iter_stagnant = 0
    else:
        rga.result.message = 'Maximum number of iterations reached.'  # type: ignore[attr-defined]
        rga.result.success = False  # type: ignore[attr-defined]
    return rga.result
