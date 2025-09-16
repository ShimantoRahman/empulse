import sys
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.base import clone
from sklearn.utils._param_validation import HasMethods

from ..._types import FloatArrayLike, FloatNDArray, ParameterConstraint

if TYPE_CHECKING:
    from ...metrics.savings import AECMetric, AECObjective

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

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
from ...metrics import Metric, make_objective_aec
from ...metrics._loss import cy_boost_grad_hess
from .._base import BaseBoostClassifier
from ._cs_mixin import CostSensitiveMixin

# Hessian is 0 at score 0.5
# which means that at initialization the model optimization doesn't do anything
# therefore we add a small nudge which kickstarts the optimization algorithm (so hessian is not 0)
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
        gradient, hessian = cy_boost_grad_hess(y_true, y_score, self.gradient_const)
        return gradient, hessian


class CSBoostClassifier(BaseBoostClassifier, CostSensitiveMixin):
    """
    Cost-sensitive gradient boosting classifier.

    CSBoostClassifier supports :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier`
    and :class:`catboost.CatBoostClassifier` as base estimators.
    By default, it uses XGBoost classifier with default hyperparameters.

    Read more in the :ref:`User Guide <csboost>`.

    .. seealso::

        :func:`~empulse.metrics.make_objective_aec` : Creates the instance-dependent cost function.

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression.

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

    loss : :class:`empulse.metrics.Metric`, default=None
        Loss function to optimize. Metric parameters are passed as ``loss_params``
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
            normalize=True
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
        **BaseBoostClassifier._parameter_constraints,
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'loss': [HasMethods('_gradient_boost_objective'), None],
    }

    def __init__(
        self,
        estimator: XGBClassifier | LGBMClassifier | CatBoostClassifier | None = None,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
    ) -> None:
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
        super().__init__(estimator=estimator)

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
        X: NDArray[Any],
        y: NDArray[Any],
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        fit_params: dict[str, Any] | None = None,
        **loss_params: Any,
    ) -> Self:
        if fit_params is None:
            fit_params = {}
        # allow sample weights still to be passed as kwargs to comply with sklearn interface
        if 'sample_weight' in loss_params:
            fit_params['sample_weight'] = loss_params.pop('sample_weight')

        if self.loss is None:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )

        if self.estimator is None:
            self._initialize_default_estimator(
                y=y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params
            )
        else:
            self._initialize_custom_estimator(
                y=y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params
            )

        if isinstance(self.estimator_, XGBClassifier):
            self.estimator_.fit(X, y, **fit_params)
        elif isinstance(self.estimator_, LGBMClassifier):
            self.estimator_.fit(X, y, init_score=np.full(y.shape, _BASE_SCORE), **fit_params)
        else:
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
                if 'sample_weight' in fit_params:
                    raise ValueError('Sample weights are not allowed when training CatboostClassifier.')
                self.estimator_.fit(X, y, sample_weight=indices, baseline=np.full(y.shape, _BASE_SCORE), **fit_params)
        return self

    def _initialize_default_estimator(
        self,
        y: FloatNDArray,
        tp_cost: FloatNDArray | FloatArrayLike | float,
        tn_cost: FloatNDArray | FloatArrayLike | float,
        fn_cost: FloatNDArray | FloatArrayLike | float,
        fp_cost: FloatNDArray | FloatArrayLike | float,
        **loss_params: Any,
    ) -> None:
        if XGBClassifier is None:
            raise ImportError(
                f'XGBoost package is required to use {type(self).__name__}. '
                'Install optional dependencies through `pip install empulse[optional]` or '
                '`pip install xgboost`'
            )
        objective = self._get_objective(
            'xgboost', y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params
        )
        self.estimator_ = XGBClassifier(objective=objective, base_score=_BASE_SCORE)

    def _initialize_custom_estimator(
        self,
        y: FloatNDArray,
        tp_cost: FloatNDArray | FloatArrayLike | float,
        tn_cost: FloatNDArray | FloatArrayLike | float,
        fn_cost: FloatNDArray | FloatArrayLike | float,
        fp_cost: FloatNDArray | FloatArrayLike | float,
        **loss_params: Any,
    ) -> None:
        if isinstance(self.estimator, XGBClassifier):
            objective = self._get_objective(
                'xgboost', y=y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params
            )
            self.estimator_ = clone(self.estimator).set_params(objective=objective, base_score=_BASE_SCORE)
        elif isinstance(self.estimator, LGBMClassifier):
            objective = self._get_objective(
                'lightgbm', y=y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params
            )
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        elif isinstance(self.estimator, CatBoostClassifier):
            # self._initialize_catboost_estimator(tp_cost, tn_cost, fn_cost, fp_cost, **loss_params)
            loss_function, eval_metric = self._get_objective(
                'catboost', y=y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params
            )
            self.estimator_ = clone(self.estimator).set_params(loss_function=loss_function, eval_metric=eval_metric)
        else:
            raise ValueError('Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier')

    @overload
    def _get_objective(
        self,
        framework: Literal['xgboost'],
        y: FloatNDArray,
        tp_cost: FloatNDArray | FloatArrayLike | float,
        tn_cost: FloatNDArray | FloatArrayLike | float,
        fn_cost: FloatNDArray | FloatArrayLike | float,
        fp_cost: FloatNDArray | FloatArrayLike | float,
        **loss_params: Any,
    ) -> Callable[..., Any]: ...

    @overload
    def _get_objective(
        self,
        framework: Literal['lightgbm'],
        y: FloatNDArray,
        tp_cost: FloatNDArray | FloatArrayLike | float,
        tn_cost: FloatNDArray | FloatArrayLike | float,
        fn_cost: FloatNDArray | FloatArrayLike | float,
        fp_cost: FloatNDArray | FloatArrayLike | float,
        **loss_params: Any,
    ) -> LGBMObjective: ...

    @overload
    def _get_objective(
        self,
        framework: Literal['catboost'],
        y: FloatNDArray,
        tp_cost: FloatNDArray | FloatArrayLike | float,
        tn_cost: FloatNDArray | FloatArrayLike | float,
        fn_cost: FloatNDArray | FloatArrayLike | float,
        fp_cost: FloatNDArray | FloatArrayLike | float,
        **loss_params: Any,
    ) -> tuple['AECObjective', 'AECMetric'] | tuple['CatboostObjective', 'CatboostMetric']: ...

    def _get_objective(
        self,
        framework: Literal['xgboost', 'lightgbm', 'catboost'],
        y: FloatNDArray,
        tp_cost: FloatNDArray | FloatArrayLike | float,
        tn_cost: FloatNDArray | FloatArrayLike | float,
        fn_cost: FloatNDArray | FloatArrayLike | float,
        fp_cost: FloatNDArray | FloatArrayLike | float,
        **loss_params: Any,
    ) -> (
        Callable[..., Any]
        | LGBMObjective
        | tuple['AECObjective', 'AECMetric']
        | tuple['CatboostObjective', 'CatboostMetric']
    ):
        if self.loss is None:
            return make_objective_aec(framework, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)  # type: ignore[arg-type, misc]
        if framework == 'xgboost':
            # return partial(self.loss._gradient_boost_objective, **loss_params)
            grad_const = self.loss._prepare_boost_objective(y, **loss_params).reshape(-1)
            return partial(cy_boost_grad_hess, grad_const=grad_const)
        elif framework == 'lightgbm':
            grad_const = self.loss._prepare_boost_objective(y, **loss_params).reshape(-1)
            return LGBMObjective(grad_const)
        else:
            grad_const = self.loss._prepare_boost_objective(y, **loss_params).reshape(-1)
            # normalize the shape of all loss params to be (n_samples,)
            loss_params = {
                name: np.full(y.shape, param) if np.isscalar(param) else param.reshape(-1)
                for name, param in loss_params.items()
            }
            return CatboostObjective(grad_const), CatboostMetric(self.loss, **loss_params)

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None


class CatboostObjective:
    """AEC objective for catboost."""

    def __init__(self, gradient_const: FloatNDArray):
        self.gradient_const = gradient_const

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
        # Use weights as a proxy to index the costs
        gradient_const = self.gradient_const[weights]
        predictions = np.array(predictions, dtype=np.float64)

        gradient, hessian = cy_boost_grad_hess(targets, predictions, gradient_const)
        # convert from two arrays to one list of tuples
        return list(zip(-gradient, -hessian, strict=False))


class CatboostMetric:
    """AEC metric for catboost."""

    def __init__(self, metric: Callable[..., float], **loss_params: FloatNDArray | float):
        self.metric = metric
        self.loss_params = loss_params

    def is_max_optimal(self) -> bool:
        """Return whether great values of metric are better."""
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
