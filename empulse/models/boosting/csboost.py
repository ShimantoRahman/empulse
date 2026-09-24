import warnings
from typing import Any, ClassVar, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import clone
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.validation import check_is_fitted, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint

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
from ...metrics import BaseMetric, Capability
from .._base.cost_sensitive import CostSensitiveClassifier
from ._backends import (  # ruff: ignore[unused-import] (re-exported for tests/models/test_csboost.py)
    _BASE_SCORE_PROBA,
    _BASE_SCORE_RAW,
    BoostingBackend,
    backend_for,
)


def _backend_for_estimator(estimator: Any) -> BoostingBackend | None:
    """:func:`backend_for`, threading through this module's own (patchable) classifier names."""
    return backend_for(estimator, xgb_cls=XGBClassifier, lgbm_cls=LGBMClassifier, catboost_cls=CatBoostClassifier)


class CSBoostClassifier(CostSensitiveClassifier):
    """
    Cost-sensitive gradient boosting classifier.

    CSBoostClassifier supports :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier`
    and `CatBoostClassifier
    <https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier>`__ as base estimators.
    By default, it uses XGBoost classifier with default hyperparameters.

    Read more in the :ref:`User Guide <csboost>`.

    .. seealso::

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression classifier.

        :class:`~empulse.models.CSTreeClassifier` : Cost-sensitive decision tree classifier.

        :class:`~empulse.models.CSForestClassifier` : Cost-sensitive random forest classifier.

    Parameters
    ----------
    estimator : :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier` \
    or `CatBoostClassifier <https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier>`__, optional
        XGBoost or LightGBM classifier to be fit with desired hyperparameters.
        If not provided, a XGBoost classifier with default hyperparameters is used.

    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

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

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    loss : :class:`~empulse.metrics.BaseMetric` or None, default=None
        Loss function to optimize. Loss parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.CSBoostClassifier.fit` method.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : :class:`xgboost:xgboost.XGBClassifier`
        Fitted XGBoost classifier.

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.

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
    """

    estimator_: XGBClassifier | LGBMClassifier | CatBoostClassifier

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

        fit_params : dict
            Additional keyword arguments to pass to the estimator's fit method.

        **loss_params : dict
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
        fit_params = {} if fit_params is None else dict(fit_params)
        # allow sample weights still to be passed as kwargs to comply with sklearn interface
        if 'sample_weight' in loss_params:
            fit_params['sample_weight'] = loss_params.pop('sample_weight')

        if self.estimator is None:
            backend = self._initialize_default_estimator(y=y, loss=loss, **loss_params)
        else:
            backend = self._initialize_custom_estimator(y=y, loss=loss, **loss_params)

        y_fit, fit_kwargs = backend.fit_arguments(y, loss, loss_params, fit_params)
        with warnings.catch_warnings():
            for message, category in backend.warning_filters:
                warnings.filterwarnings('ignore', message=message, category=category)
            self.estimator_.fit(X, y_fit, **fit_kwargs)
        return self

    def _initialize_default_estimator(
        self,
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> BoostingBackend:
        xgb_cls = None if isinstance(XGBClassifier, TypeVar) else XGBClassifier
        backend = BoostingBackend(name='xgboost', classifier=xgb_cls)
        if backend.classifier is None:
            raise ImportError(
                f'XGBoost package is required to use {type(self).__name__}. '
                'Install the boosting backends through `pip install empulse[boosting]` or '
                '`pip install xgboost`'
            )
        objective = self._get_objective(backend, y, loss=loss, **loss_params)
        self.estimator_ = backend.build_default(objective)
        return backend

    def _initialize_custom_estimator(
        self,
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> BoostingBackend:
        backend = _backend_for_estimator(self.estimator)
        if backend is None:
            raise TypeError('Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier')
        objective = self._get_objective(backend, y=y, loss=loss, **loss_params)
        self.estimator_ = backend.apply_objective(clone(self.estimator), objective)
        return backend

    def _get_objective(
        self,
        backend: BoostingBackend,
        y: FloatNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> Any:
        # MaxProfit requires dynamic thresholding from current round predictions, and LogCost's
        # per-sample loss is non-linear in the predicted probability (unlike Cost/Savings), so both
        # evaluate gradients/hessians directly from the metric each iteration instead of going through
        # a precomputed constant.
        capabilities = loss.capabilities
        if Capability.BOOST_OBJECTIVE in capabilities:
            return backend.wrap_objective(loss, y, loss_params, precomputed=False)

        if Capability.PRECOMPUTED_BOOST_OBJECTIVE not in capabilities:
            raise ValueError(
                f'{type(self).__name__} requires a loss whose strategy supports gradient boosting '
                f"(neither 'boost_objective' nor 'precomputed_boost_objective'; got the "
                f'{loss.strategy.name!r} strategy).'
            )
        return backend.wrap_objective(loss, y, loss_params, precomputed=True)

    def predict_proba(self, X: ArrayLike) -> FloatNDArray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        y_proba: FloatNDArray
        backend = _backend_for_estimator(self.estimator_)
        raw_score = backend.raw_score(self.estimator_, X) if backend is not None else None
        if raw_score is not None:
            y_proba = expit(raw_score + _BASE_SCORE_RAW)
            return np.column_stack([1 - y_proba, y_proba])

        y_proba = self.estimator_.predict_proba(X)
        return y_proba
