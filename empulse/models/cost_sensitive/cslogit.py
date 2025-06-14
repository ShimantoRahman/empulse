import sys
import warnings
from collections.abc import Callable
from functools import partial
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import HasMethods, StrOptions

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import Metric, make_objective_aec
from .._base import BaseLogitClassifier, LossFn, OptimizeFn
from ._cs_mixin import CostSensitiveMixin

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
Loss = Literal['average expected cost']
GradientLossFn = Callable[[FloatNDArray, FloatNDArray, FloatNDArray], tuple[float, FloatNDArray]]
ObjectiveFn = Callable[..., float | tuple[float, FloatNDArray] | tuple[float, FloatNDArray, FloatNDArray]]


class CSLogitClassifier(BaseLogitClassifier, CostSensitiveMixin):
    """
    Logistic classifier to optimize instance-dependent cost loss.

    Read more in the :ref:`User Guide <cslogit>`.

    .. seealso::

        :func:`~empulse.metrics.make_objective_aec` : Creates the instance-dependent cost function.

        :class:`~empulse.models.CSBoostClassifier` : Cost-sensitive XGBoost classifier.

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

            - For ``l1_ratio = 0`` the penalty is a L2 penalty.
            - For ``l1_ratio = 1`` it is a L1 penalty.
            - For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    loss : {'average expected cost'}, Callable or :class:`empulse.metrics.Metric`, default='average expected cost'
        Loss function which should be minimized.

        - If ``str``, then it should be one of the following:

            - 'average expected cost' : Average Expected Cost loss function,
              see :func:`~empulse.metrics.expected_cost_loss`.

        - If ``Callable`` it should have a signature ``loss(y_true, y_score)``.

        - If :class`~empulse.metrics.Metric`, metric parameters are passed as ``loss_params``
          to the :Meth:`~empulse.models.CSLogitClassifier.fit` method.

        By default, loss function is minimized, customize behaviour in `optimize_fn`.

    optimize_fn : Callable, optional
        Optimization algorithm. Should be a Callable with signature ``optimize(objective, X)``.
        See :ref:`proflogit` for more information.

    optimizer_params : dict[str, Any], optional
        Additional keyword arguments passed to `optimize_fn`.

        tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

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
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

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

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    coef_ : numpy.ndarray, shape=(n_features,)
        Coefficients of the logit model.

    intercept_ : float
        Intercept of the logit model.
        Only available when ``fit_intercept=True``.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.models import CSLogitClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()
        fn_cost = np.random.rand(y.size)  # instance-dependent cost
        fp_cost = 5  # constant cost

        model = CSLogitClassifier(C=0.1)
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
        y_proba = model.predict_proba(X)

    Example with passing instance-dependent costs through cross-validation:

    .. code-block:: python

        import numpy as np
        from empulse.models import CSLogitClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        set_config(enable_metadata_routing=True)

        X, y = make_classification()
        fn_cost = np.random.rand(y.size)
        fp_cost = 5

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CSLogitClassifier(C=0.1).set_fit_request(fn_cost=True, fp_cost=True)),
        ])

        cross_val_score(pipeline, X, y, params={'fn_cost': fn_cost, 'fp_cost': fp_cost})

    Example with passing instance-dependent costs through a grid search:

    .. code-block:: python

        import numpy as np
        from empulse.metrics import expected_cost_loss
        from empulse.models import CSLogitClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        set_config(enable_metadata_routing=True)

        X, y = make_classification(n_samples=50)
        fn_cost = np.random.rand(y.size)
        fp_cost = 5

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CSLogitClassifier().set_fit_request(fn_cost=True, fp_cost=True)),
        ])
        param_grid = {'model__C': np.logspace(-5, 2, 5)}
        scorer = make_scorer(
            expected_cost_loss,
            response_method='predict_proba',
            greater_is_better=False,
            normalize=True,
        )
        scorer = scorer.set_score_request(fn_cost=True, fp_cost=True)

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer)
        grid_search.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **BaseLogitClassifier._parameter_constraints,
        'loss': [StrOptions({'average expected cost'}), HasMethods('_logit_objective'), callable, None],
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
    }

    def __init__(
        self,
        *,
        C: float = 1.0,
        fit_intercept: bool = True,
        soft_threshold: bool = False,
        l1_ratio: float = 1.0,
        loss: Loss | LossFn | Metric = 'average expected cost',
        optimize_fn: OptimizeFn | None = None,
        optimizer_params: dict[str, Any] | None = None,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
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
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None

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
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

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

        loss_params : dict[str, Any]
            Additional parameters passed to the loss function.

        Returns
        -------
        self : CSLogitClassifier
            Fitted model.

        """
        super().fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params)
        return self

    def _fit(
        self,
        X: FloatNDArray,
        y: IntNDArray,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        **loss_params: Any,
    ) -> Self:
        optimizer_params = self.optimizer_params or {}

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

        if self.loss == 'average expected cost':
            loss = make_objective_aec(model='cslogit', **loss_params)
            objective: ObjectiveFn = _objective_jacobian
            optimize_fn: Callable[..., OptimizeResult] = _optimize_jacobian
        elif isinstance(self.loss, Metric):
            loss = partial(self.loss._logit_objective, **loss_params)
            objective = _objective_jacobian
            optimize_fn = _optimize_jacobian
        elif self.loss is not None and not isinstance(self.loss, str):
            loss: Callable[[FloatNDArray, FloatNDArray, FloatNDArray], tuple[float, FloatNDArray]] = partial(  # type: ignore[no-redef]
                self.loss, **loss_params
            )
            objective = _objective_callable
            optimize_fn = self._optimize
        else:
            raise ValueError(f'Invalid loss function: {self.loss}')

        partial_objective: Callable[
            [FloatNDArray],
            float | tuple[float, FloatNDArray] | tuple[float, FloatNDArray, FloatNDArray],
        ] = partial(
            objective,
            X=X,
            y=y,
            loss_fn=loss,
            C=self.C,
            l1_ratio=self.l1_ratio,
            soft_threshold=self.soft_threshold,
            fit_intercept=self.fit_intercept,
        )
        optimize_fn: Callable[..., OptimizeResult] = optimize_fn if self.optimize_fn is None else self.optimize_fn  # type: ignore[no-redef]
        self.result_ = optimize_fn(objective=partial_objective, X=X, **optimizer_params)
        self.coef_ = self.result_.x[1:] if self.fit_intercept else self.result_.x
        if self.fit_intercept:
            self.intercept_ = self.result_.x[0]

        return self

    @staticmethod
    def _optimize(
        objective: Callable[[FloatNDArray], float],
        X: FloatNDArray,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        **kwargs: Any,
    ) -> OptimizeResult:
        initial_weights = np.zeros(X.shape[1], order='F', dtype=np.float64)

        result = minimize(
            objective,
            initial_weights,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'maxls': 50,
                'gtol': tolerance,
                'ftol': 64 * np.finfo(float).eps,
            },
            **kwargs,
        )
        _check_optimize_result(result)

        return result


def _optimize_jacobian(
    objective: Callable[[FloatNDArray], tuple[float, FloatNDArray]],
    X: FloatNDArray,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
    **kwargs: Any,
) -> OptimizeResult:
    initial_weights = np.zeros(X.shape[1], order='F', dtype=X.dtype)

    result = minimize(
        objective,
        initial_weights,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iter,
            'maxls': 50,
            'gtol': tolerance,
            'ftol': 64 * np.finfo(float).eps,
        },
        **kwargs,
    )
    _check_optimize_result(result)

    return result


def _objective_jacobian(
    weights: FloatNDArray,
    X: FloatNDArray,
    y: IntNDArray,
    loss_fn: GradientLossFn,
    C: float,
    l1_ratio: float,
    soft_threshold: bool,
    fit_intercept: bool,
) -> tuple[float, FloatNDArray]:
    """Compute the objective function and its gradient using elastic net regularization."""
    if soft_threshold:
        b = weights.copy()[1:] if fit_intercept else weights.copy()
        bool_nonzero = (np.abs(b) - C) > 0
        if np.sum(bool_nonzero) > 0:
            b[bool_nonzero] = np.sign(b[bool_nonzero]) * (np.abs(b[bool_nonzero]) - C)
        if np.sum(~bool_nonzero) > 0:
            b[~bool_nonzero] = 0
    else:
        b = weights[1:] if fit_intercept else weights

    loss, gradient = loss_fn(X, weights, y)
    if l1_ratio == 0.0:
        regularization_term = 0.5 * np.sum(b**2)
        gradient_penalty = b
    elif l1_ratio == 1.0:
        regularization_term = np.sum(np.abs(b))
        gradient_penalty = np.sign(b)
    else:
        regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b**2) + l1_ratio * np.sum(np.abs(b))
        gradient_penalty = (1 - l1_ratio) * b + l1_ratio * np.sign(b)
    penalty = regularization_term / C
    if fit_intercept:
        gradient_penalty = np.hstack((np.array([0]), gradient_penalty))
    return loss + penalty, gradient + gradient_penalty


def _objective_callable(
    weights: FloatNDArray,
    X: FloatNDArray,
    y: IntNDArray,
    loss_fn: LossFn,
    C: float,
    l1_ratio: float,
    soft_threshold: bool,
    fit_intercept: bool,
) -> float | tuple[float, FloatNDArray] | tuple[float, FloatNDArray, FloatNDArray]:
    """Objective function (minimization problem)."""
    # b is the vector holding the regression coefficients (no intercept)
    b = weights.copy()[1:] if fit_intercept else weights

    if soft_threshold:
        bool_nonzero = (np.abs(b) - C) > 0
        if np.sum(bool_nonzero) > 0:
            b[bool_nonzero] = np.sign(b[bool_nonzero]) * (np.abs(b[bool_nonzero]) - C)
        if np.sum(~bool_nonzero) > 0:
            b[~bool_nonzero] = 0

    logits = np.dot(weights, X.T)
    y_pred = expit(logits)
    loss_output = loss_fn(y, y_pred)
    if isinstance(loss_output, tuple):
        if len(loss_output) == 2:
            loss, gradient = loss_output
            return loss + _compute_penalty(b, C, l1_ratio), gradient
        elif len(loss_output) == 3:
            loss, gradient, hessian = loss_output
            return loss + _compute_penalty(b, C, l1_ratio), gradient, hessian
        else:
            raise ValueError(
                f'Invalid loss function output length: {len(loss_output)}, expected 1, 2 or 3. '
                f'(loss, gradient, hessian)'
            )
    else:
        loss = loss_output
        return loss + _compute_penalty(b, C, l1_ratio)


def _compute_penalty(b: FloatNDArray, C: float, l1_ratio: float) -> float:
    regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b**2) + l1_ratio * np.sum(np.abs(b))
    penalty = float(regularization_term / C)
    return penalty


def _check_optimize_result(result: OptimizeResult) -> None:
    """
    Check the OptimizeResult for successful convergence.

    Parameters
    ----------
    result : OptimizeResult
       Result of the scipy.optimize.minimize function.
    """
    # handle both scipy and scikit-learn solver names
    if result.status != 0:
        try:
            # The message is already decoded in scipy>=1.6.0
            result_message = result.message.decode('latin1')
        except AttributeError:
            result_message = result.message
        warning_msg = (
            f'L-BFGS failed to converge (status={result.status}):\n{result_message}.\n\n'
            'Increase the number of iterations (max_iter) '
            'or scale the data as shown in:\n'
            '    https://scikit-learn.org/stable/modules/'
            'preprocessing.html'
        )
        warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
