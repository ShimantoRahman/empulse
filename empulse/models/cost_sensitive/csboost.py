import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, ClassVar, Literal, Self, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import clone
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.validation import check_is_fitted, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics.metric.common import Direction

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = TypeVar('XGBClassifier')  # type: ignore[misc, assignment]
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = TypeVar('LGBMClassifier')  # type: ignore[misc, assignment]
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = TypeVar('CatBoostClassifier')  # type: ignore[misc, assignment]

from ..._common import Parameter
from ...metrics import BaseMetric
from ...metrics._loss import cy_boost_grad_hess
from ..csclassifier import CostSensitiveClassifier

# Hessian is 0 at score 0.5 because the AEC objective's hessian evaluates to p*(1-p),
# which is exactly 0 when p=0.5. A nudge of 1e-2 is large enough to produce a non-zero
# hessian at initialization (kick-starting the optimizer) yet small enough not to meaningfully
# bias the starting point away from 0.5.
_BASE_SCORE = 0.5 + 1e-2


class LGBMObjective:
    """AEC objective for lightgbm."""

    def __init__(self, gradient_const: FloatNDArray):
        self.gradient_const = gradient_const

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Create an objective function for the AEC measure.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_score : np.ndarray
            Predicted labels

        Returns
        -------
        gradient  : np.ndarray
            Gradient of the objective function.

        hessian : np.ndarray
            Hessian of the objective function.
        """
        gradient: FloatNDArray
        hessian: FloatNDArray
        # cy_boost_grad_hess (Cython) requires float64 memoryviews
        gradient, hessian = cy_boost_grad_hess(
            np.asarray(y_true, dtype=np.float64),
            np.asarray(y_score, dtype=np.float64),
            np.asarray(self.gradient_const, dtype=np.float64),
        )
        return gradient, hessian


class LGBMMetricObjective:
    """Metric objective wrapper for lightgbm using dynamic gradient/hessian evaluation."""

    def __init__(self, metric: BaseMetric, **loss_params: FloatNDArray | float):
        self.metric = metric
        self.loss_params = loss_params

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the metric objective."""
        gradient, hessian = self.metric._gradient_boost_objective(y_true, y_score, **self.loss_params)
        return gradient, hessian


class CSBoostClassifier(CostSensitiveClassifier):
    """
    Cost-sensitive gradient boosting classifier.

    CSBoostClassifier supports :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier`
    and :class:`catboost.CatBoostClassifier` as base estimators.
    By default, it uses XGBoost classifier with default hyperparameters.

    Read more in the :ref:`User Guide <csboost>`.

    .. seealso::

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression classifier.

        :class:`~empulse.models.CSTreeClassifier` : Cost-sensitive decision tree classifier.

        :class:`~empulse.models.CSForestClassifier` : Cost-sensitive random forest classifier.

    Parameters
    ----------
    estimator : :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier` \
    or :class:`catboost.CatBoostClassifier`, optional
        XGBoost or LightGBM classifier to be fit with desired hyperparameters.
        If not provided, a XGBoost classifier with default hyperparameters is used.

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

    loss : :class:`empulse.metrics.BaseMetric`, default=None
        Loss function to optimize. Loss parameters are passed as ``loss_params``
          to the :Meth:`~empulse.models.CSBoostClassifier.fit` method.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : :class:`xgboost:xgboost.XGBClassifier`
        Fitted XGBoost classifier.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.models import CSBoostClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()
        fn_cost = np.random.rand(y.size)  # instance-dependent cost
        fp_cost = 5  # constant cost

        model = CSBoostClassifier()
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
        y_proba = model.predict_proba(X)

    Example with passing instance-dependent costs through cross-validation:

    .. code-block:: python

        import numpy as np
        from empulse.models import CSBoostClassifier
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
            ('model', CSBoostClassifier().set_fit_request(fn_cost=True, fp_cost=True))
        ])

        cross_val_score(pipeline, X, y, params={'fn_cost': fn_cost, 'fp_cost': fp_cost})

    Example with passing instance-dependent costs through a grid search:

    .. code-block:: python

        import numpy as np
        from empulse.metrics import expected_cost_loss
        from empulse.models import CSBoostClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        set_config(enable_metadata_routing=True)

        X, y = make_classification(n_samples=50)
        fn_cost = np.random.rand(y.size)
        fp_cost = 5

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CSBoostClassifier(
                XGBClassifier(n_jobs=2, n_estimators=10)
            ).set_fit_request(fn_cost=True, fp_cost=True))
        ])
        param_grid = {
            'model__estimator__learning_rate': np.logspace(-5, 0, 5),
        }
        scorer = make_scorer(
            expected_cost_loss,
            response_method='predict_proba',
            greater_is_better=False,
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
        'estimator': [HasMethods(['fit', 'predict_proba']), None],
        **CostSensitiveClassifier._parameter_constraints,
    }

    def __init__(
        self,
        estimator: XGBClassifier | LGBMClassifier | CatBoostClassifier | None = None,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: BaseMetric | None = None,
    ) -> None:
        self.estimator = estimator
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fit_params: dict[str, Any] | None = None,
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

        fit_params : dict
            Additional keyword arguments to pass to the estimator's fit method.

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        self : CSBoostClassifier
            Fitted CSBoost model.
        """
        super().fit(
            X,
            y,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            fit_params=fit_params,
            **loss_params,
        )
        return self

    def _fit(
        self,
        X: FloatNDArray,
        y: IntNDArray,
        loss: BaseMetric,
        *,
        fit_params: dict[str, Any] | None = None,
        **loss_params: Any,
    ) -> Self:
        if fit_params is None:
            fit_params = {}
        # allow sample weights still to be passed as kwargs to comply with sklearn interface
        if 'sample_weight' in loss_params:
            fit_params['sample_weight'] = loss_params.pop('sample_weight')

        # CatBoost uses sample_weight internally as an index proxy,
        if (
            'sample_weight' in fit_params
            and not isinstance(CatBoostClassifier, TypeVar)
            and (self.estimator is not None and isinstance(self.estimator, CatBoostClassifier))
        ):
            raise ValueError('Sample weights are not allowed when training CatBoostClassifier.')

        if self.estimator is None:
            self._initialize_default_estimator(y=y, loss=loss, **loss_params)
        else:
            self._initialize_custom_estimator(y=y, loss=loss, **loss_params)

        if not isinstance(XGBClassifier, TypeVar) and isinstance(self.estimator_, XGBClassifier):
            self.estimator_.fit(X, y, **fit_params)
        elif not isinstance(LGBMClassifier, TypeVar) and isinstance(self.estimator_, LGBMClassifier):
            self.estimator_.fit(X, y, init_score=np.full(y.shape, _BASE_SCORE), **fit_params)
        elif not isinstance(CatBoostClassifier, TypeVar) and isinstance(self.estimator_, CatBoostClassifier):
            indices = np.arange(X.shape[0])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='Can\'t optimize method "calc_ders_range" because self argument is used',
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    'ignore',
                    message='Can\'t optimize method "evaluate" because self argument is used',
                    category=UserWarning,
                )
                self.estimator_.fit(X, y, sample_weight=indices, baseline=np.full(y.shape, _BASE_SCORE), **fit_params)
        else:
            raise TypeError('Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier')
        return self

    def _initialize_default_estimator(
        self,
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> None:
        if isinstance(XGBClassifier, TypeVar):
            raise ImportError(  # noqa: TRY004
                f'XGBoost package is required to use {type(self).__name__}. '
                'Install optional dependencies through `pip install empulse[optional]` or '
                '`pip install xgboost`'
            )
        objective = self._get_objective('xgboost', y, loss=loss, **loss_params)
        self.estimator_ = XGBClassifier(objective=objective, base_score=_BASE_SCORE)

    def _initialize_custom_estimator(
        self,
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> None:
        if not isinstance(XGBClassifier, TypeVar) and isinstance(self.estimator, XGBClassifier):
            objective = self._get_objective('xgboost', y=y, loss=loss, **loss_params)
            self.estimator_ = clone(self.estimator).set_params(objective=objective, base_score=_BASE_SCORE)
        elif not isinstance(LGBMClassifier, TypeVar) and isinstance(self.estimator, LGBMClassifier):
            objective = self._get_objective('lightgbm', y=y, loss=loss, **loss_params)
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        elif not isinstance(CatBoostClassifier, TypeVar) and isinstance(self.estimator, CatBoostClassifier):
            loss_function, eval_metric = self._get_objective('catboost', y=y, loss=loss, **loss_params)
            self.estimator_ = clone(self.estimator).set_params(loss_function=loss_function, eval_metric=eval_metric)
        else:
            raise TypeError('Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier')

    @overload
    def _get_objective(
        self,
        framework: Literal['xgboost'],
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> Callable[..., Any]: ...

    @overload
    def _get_objective(
        self,
        framework: Literal['lightgbm'],
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> LGBMObjective | LGBMMetricObjective: ...

    @overload
    def _get_objective(
        self,
        framework: Literal['catboost'],
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> tuple['CatBoostObjective', 'CatBoostMetric']: ...

    def _get_objective(
        self,
        framework: Literal['xgboost', 'lightgbm', 'catboost'],
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> Callable[..., Any] | LGBMObjective | LGBMMetricObjective | tuple['CatBoostObjective', 'CatBoostMetric']:
        # MaxProfit requires dynamic thresholding from current round predictions, and LogCost's
        # per-sample loss is non-linear in the predicted probability (unlike Cost/Savings), so both
        # evaluate gradients/hessians directly from the metric each iteration instead of going through
        # a precomputed constant.
        if loss.strategy.requires_dynamic_boost_objective:
            if framework == 'xgboost':
                return partial(loss._gradient_boost_objective, **loss_params)
            if framework == 'lightgbm':
                return LGBMMetricObjective(loss, **loss_params)

            loss_params = {
                name: np.full(y.shape, param) if np.isscalar(param) else param.reshape(-1)
                for name, param in loss_params.items()
            }
            return CatBoostObjective(loss, **loss_params), CatBoostMetric(loss, **loss_params)

        if framework == 'xgboost':
            grad_const = loss._prepare_boost_objective(y, **loss_params).reshape(-1)
            # cy_boost_grad_hess (Cython) requires a float64 memoryview.
            return partial(cy_boost_grad_hess, grad_const=np.asarray(grad_const, dtype=np.float64))
        elif framework == 'lightgbm':
            grad_const = loss._prepare_boost_objective(y, **loss_params).reshape(-1)
            return LGBMObjective(grad_const)
        else:
            grad_const = loss._prepare_boost_objective(y, **loss_params).reshape(-1)
            # normalize the shape of all loss params to be (n_samples,)
            loss_params = {
                name: np.full(y.shape, param) if np.isscalar(param) else param.reshape(-1)
                for name, param in loss_params.items()
            }
            return CatBoostObjective(grad_const), CatBoostMetric(loss, **loss_params)

    def predict_proba(self, X: ArrayLike) -> FloatNDArray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_features)

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if not isinstance(LGBMClassifier, TypeVar) and isinstance(self.estimator_, LGBMClassifier):
            y_proba: FloatNDArray = self.estimator_.predict_proba(X, raw_score=True)
            y_proba: FloatNDArray = expit(y_proba)
            return np.column_stack([1 - y_proba, y_proba])

        y_proba: FloatNDArray = self.estimator_.predict_proba(X)  # type: ignore[no-redef]
        return y_proba


class CatBoostObjective:
    """AEC objective for catboost."""

    def __init__(self, metric_or_gradient_const: BaseMetric | FloatNDArray, **loss_params: FloatNDArray | float):
        self.metric = metric_or_gradient_const if isinstance(metric_or_gradient_const, BaseMetric) else None
        self.gradient_const = metric_or_gradient_const if isinstance(metric_or_gradient_const, np.ndarray) else None
        self.loss_params = loss_params

    def calc_ders_range(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: FloatNDArray
    ) -> list[tuple[float, float]]:
        """
        Compute first and second derivative of the loss function with respect to the predicted value for each object.

        Parameters
        ----------
        predictions : indexed container of floats
            Current predictions for each object.

        targets : indexed container of floats
            Target values you provided with the dataset.

        weights : float, optional (default=None)
            Instance weight. Here instance weights are used to pass the indices of the instances, not actual weights.

        Returns
        -------
            der1 : list-like object of float
            der2 : list-like object of float

        """
        weights = weights.astype(int)
        predictions = np.array(predictions, dtype=np.float64)

        if self.metric is not None:
            # Use weights as a proxy to index instance-dependent parameters.
            loss_params = {
                name: value[weights] if isinstance(value, np.ndarray) else value
                for (name, value) in self.loss_params.items()
            }
            gradient, hessian = self.metric._gradient_boost_objective(targets, predictions, **loss_params)
        else:
            gradient_const = self.gradient_const[weights]  # type: ignore[index]
            # cy_boost_grad_hess (Cython) requires a float64 memoryview; targets is typed as the
            # broader FloatNDArray to match catboost's own callback convention.
            gradient, hessian = cy_boost_grad_hess(np.asarray(targets, dtype=np.float64), predictions, gradient_const)
        # convert from two arrays to one list of tuples
        gradient_f = np.asarray(gradient, dtype=np.float32)
        hessian_f = np.asarray(hessian, dtype=np.float32)
        return list(zip(-gradient_f, -hessian_f, strict=False))


class CatBoostMetric:
    """AEC metric for catboost."""

    def __init__(self, metric: Callable[..., float], **loss_params: FloatNDArray | float):
        self.metric = metric
        self.loss_params = loss_params

    def is_max_optimal(self) -> bool:
        """Return whether greater values of metric are better."""
        if isinstance(self.metric, BaseMetric):
            return self.metric.strategy.direction == Direction.MAXIMIZE
        return False

    def evaluate(
        self, predictions: Sequence[float], targets: Sequence[float], weights: FloatNDArray
    ) -> tuple[float, float]:
        """
        Evaluate metric value.

        Parameters
        ----------
        approxes : list of indexed containers (containers with only __len__ and __getitem__ defined) of float
            Vectors of approx labels.

        targets : one dimensional indexed container of float
            Vectors of true labels.

        weights : one dimensional indexed container of float, optional (default=None)
            Weight for each instance.
            Here instance weights are used to pass the indices of the instances, not actual weights.

        Returns
        -------
            weighted error : float
            total weight : float

        """
        weights = weights.astype(int)
        # Use weights as a proxy to index the costs
        loss_params = {
            name: value[weights] if isinstance(value, np.ndarray) else value
            for (name, value) in self.loss_params.items()
        }

        y_proba = expit(predictions)
        return self.metric(targets, y_proba, **loss_params), 1

    def get_final_error(self, error: float, weight: float) -> float:
        """
        Return final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        metric value : float

        """
        return error
