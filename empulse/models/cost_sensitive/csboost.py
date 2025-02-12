import warnings
from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from ..._common import Parameter
from ...metrics import make_objective_aec
from .._base import BaseBoostClassifier
from ._cs_mixin import CostSensitiveMixin


class CSBoostClassifier(BaseBoostClassifier, CostSensitiveMixin):
    """
    Gradient boosting model to optimize instance-dependent cost loss.

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

    _parameter_constraints: ClassVar[dict[str, list]] = {
        **BaseBoostClassifier._parameter_constraints,
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
    }

    def __init__(
        self,
        estimator: XGBClassifier | LGBMClassifier | CatBoostClassifier | None = None,
        *,
        tp_cost: ArrayLike | float = 0.0,
        tn_cost: ArrayLike | float = 0.0,
        fn_cost: ArrayLike | float = 0.0,
        fp_cost: ArrayLike | float = 0.0,
    ) -> None:
        super().__init__(estimator=estimator)
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        tp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        **fit_params: Any,
    ) -> 'CSBoostClassifier':
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

        Returns
        -------
        self : B2BoostClassifier
            Fitted B2Boost model.
        """
        return super().fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **fit_params)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        tp_cost: ArrayLike | float = 0.0,
        tn_cost: ArrayLike | float = 0.0,
        fn_cost: ArrayLike | float = 0.0,
        fp_cost: ArrayLike | float = 0.0,
        **fit_params: Any,
    ) -> 'CSBoostClassifier':
        tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
            tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
        )

        if self.estimator is None:
            if XGBClassifier is None:
                raise ImportError(
                    'XGBoost package is required to use CSBoostClassifier. '
                    'Install optional dependencies through pip install empulse[optional] or '
                    'pip install xgboost'
                )
            objective = make_objective_aec(
                'xgboost',
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            self.estimator_ = XGBClassifier(objective=objective)
        elif isinstance(self.estimator, XGBClassifier):
            objective = make_objective_aec(
                'xgboost',
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        elif isinstance(self.estimator, LGBMClassifier):
            objective = make_objective_aec(
                'lightgbm',
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        elif isinstance(self.estimator, CatBoostClassifier):
            objective, metric = make_objective_aec(
                'catboost',
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            self.estimator_ = clone(self.estimator).set_params(loss_function=objective, eval_metric=metric)
            indices = np.arange(X.shape[0])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='Can\'t optimze method "calc_ders_range" because self argument is used',
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    'ignore',
                    message='Can\'t optimze method "evaluate" because self argument is used',
                    category=UserWarning,
                )
                self.estimator_.fit(X, y, sample_weight=indices, **fit_params)
            return self
        else:
            raise ValueError('estimator must be an instance of XGBClassifier or LGBMClassifier')
        self.estimator_.fit(X, y, **fit_params)
        return self
