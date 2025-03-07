import sys
import warnings
from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone

from ..._types import FloatArrayLike, ParameterConstraint

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # type: ignore[misc, assignment]
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from ..._common import Parameter
from ...metrics import make_objective_churn
from .._base import BaseBoostClassifier


class B2BoostClassifier(BaseBoostClassifier):
    """
    Gradient boosting model to optimize instance-dependent cost loss for customer churn.

    B2BoostClassifier supports :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier`
    and :class:`catboost.CatBoostClassifier`.
    By default, it uses XGBoost classifier with default hyperparameters.

    Read more in the :ref:`User Guide <csboost>`.

    Parameters
    ----------
    estimator : :class:`xgboost:xgboost.XGBClassifier`, :class:`lightgbm:lightgbm.LGBMClassifier` \
    or :class:`catboost.CatBoostClassifier`, optional
        XGBoost or LightGBM classifier to be fit with desired hyperparameters.
        If not provided, a XGBoost classifier with default hyperparameters is used.

    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (0 < `accept_rate` < 1).
        Is overwritten if another `accept_rate` is passed to the ``fit`` method.

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
        If ``array``: individualized customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).
        Is overwritten if another `clv` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).
        Is overwritten if another `incentive_fraction` is passed to the ``fit`` method.

    contact_cost : float, default=1
        Constant cost of contact (``contact_cost > 0``).
        Is overwritten if another `contact_cost` is passed to the ``fit`` method.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : :class:`xgboost:xgboost.XGBClassifier`
        Fitted XGBoost classifier.

    Notes
    -----
    The instance-specific cost function for customer churn is defined as [1]_:

    .. math:: C(s_i) = y_i[s_i(f-\\gamma (1-\\delta )CLV_i] + (1-y_i)[s_i(\\delta CLV_i + f)]

    The measure requires that the churn class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    .. seealso::
        :func:`~empulse.metrics.create_objective_churn` : Creates the instance-dependent cost function
        for customer churn.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.models import B2BoostClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()
        clv = np.random.rand(y.size) * 100

        model = B2BoostClassifier()
        model.fit(X, y, clv=clv, incentive_fraction=0.1)

    .. code-block:: python

        import numpy as np
        from empulse.models import B2BoostClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        set_config(enable_metadata_routing=True)

        X, y = make_classification(n_samples=50)
        clv = np.random.rand(y.size) * 100

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', B2BoostClassifier(contact_cost=10).set_fit_request(clv=True))
        ])

        cross_val_score(pipeline, X, y, params={'clv': clv})

    .. code-block:: python

        import numpy as np
        from empulse.metrics import empb_score
        from empulse.models import B2BoostClassifier
        from sklearn import set_config
        from sklearn.datasets import make_classification
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        set_config(enable_metadata_routing=True)

        X, y = make_classification()
        clv = np.random.rand(y.size) * 100
        contact_cost = 10

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', B2BoostClassifier(
                XGBClassifier(n_jobs=2, n_estimators=10),
                contact_cost=contact_cost
            ).set_fit_request(clv=True))
        ])
        param_grid = {
            'model__estimator__learning_rate': np.logspace(-5, 0, 5),
        }
        scorer = make_scorer(
            empb_score,
            response_method='predict_proba',
            contact_cost=contact_cost
        )
        scorer = scorer.set_score_request(clv=True)

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer)
        grid_search.fit(X, y, clv=clv)

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **BaseBoostClassifier._parameter_constraints,
        'accept_rate': [Real],
        'clv': ['array-like', Real],
        'incentive_fraction': [Real],
        'contact_cost': [Real],
    }

    def __init__(
        self,
        estimator: XGBClassifier | LGBMClassifier | CatBoostClassifier | None = None,
        *,
        accept_rate: float = 0.3,
        clv: float | FloatArrayLike = 200,
        incentive_fraction: float = 0.05,
        contact_cost: float = 15,
    ) -> None:
        super().__init__(estimator=estimator)
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost
        self.accept_rate = accept_rate

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        accept_rate: float | Parameter = Parameter.UNCHANGED,
        clv: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        incentive_fraction: float | Parameter = Parameter.UNCHANGED,
        contact_cost: float | Parameter = Parameter.UNCHANGED,
    ) -> Self:
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

        accept_rate : float, default=0.3
            Probability of a customer responding to the retention offer (``0 < accept_rate < 1``).

        clv : float or 1D array-like, shape=(n_samples), default=200
            If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
            If ``array``: individualized customer lifetime value of each customer when retained
            (``mean(clv) > incentive_cost``).

        incentive_fraction : float, default=10
            Cost of incentive offered to a customer, as a fraction of customer lifetime value
            (``0 < incentive_fraction < 1``).

        contact_cost : float, default=1
            Constant cost of contact (``contact_cost > 0``).

        Returns
        -------
        self : B2BoostClassifier
            Fitted B2Boost model.
        """
        super().fit(
            X,
            y,
            accept_rate=accept_rate,
            clv=clv,
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost,
        )
        return self

    def _fit(  # type: ignore[override]
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        accept_rate: float | Parameter,
        clv: float | FloatArrayLike | Parameter,
        incentive_fraction: float | Parameter,
        contact_cost: float | Parameter,
    ) -> Self:
        if accept_rate is Parameter.UNCHANGED:
            accept_rate = self.accept_rate
        if clv is Parameter.UNCHANGED:
            clv = self.clv
        if incentive_fraction is Parameter.UNCHANGED:
            incentive_fraction = self.incentive_fraction
        if contact_cost is Parameter.UNCHANGED:
            contact_cost = self.contact_cost
        if not isinstance(clv, float | int):
            clv = np.asarray(clv)

        if self.estimator is None:
            if XGBClassifier is None:
                raise ImportError(
                    'XGBoost package is required to use B2BoostClassifier. '
                    'Install optional dependencies through pip install empulse[optional] or '
                    'pip install xgboost'
                )
            objective = make_objective_churn(
                model='xgboost',
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
                accept_rate=accept_rate,
            )
            self.estimator_ = XGBClassifier(objective=objective)
        elif isinstance(self.estimator, XGBClassifier):
            objective = make_objective_churn(
                model='xgboost',
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
                accept_rate=accept_rate,
            )
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        elif isinstance(self.estimator, LGBMClassifier):
            objective = make_objective_churn(
                model='lightgbm',
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
                accept_rate=accept_rate,
            )
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        elif isinstance(self.estimator, CatBoostClassifier):
            objective, metric = make_objective_churn(  # type: ignore[assignment]
                'catboost',
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
                accept_rate=accept_rate,
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
                self.estimator_.fit(X, y, sample_weight=indices)
            return self
        else:
            raise ValueError('estimator must be an instance of XGBClassifier, LGBMClassifier or CatBoostClassifier')

        self.estimator_.fit(X, y)
        return self
