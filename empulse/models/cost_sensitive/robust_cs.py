import sys
from collections.abc import MutableMapping
from numbers import Real
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np
import scipy.stats as st
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, _fit_context, check_is_fitted, clone
from sklearn.linear_model import HuberRegressor
from sklearn.utils._available_if import available_if
from sklearn.utils._metadata_requests import RequestMethod
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, ParameterConstraint
from ...metrics import Metric
from ...utils._sklearn_compat import Tags, _estimator_has, validate_data  # type: ignore[attr-defined]
from ._cs_mixin import CostSensitiveMixin

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
CostStr = Literal['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost']
K = TypeVar('K')
V = TypeVar('V')


class RobustCSClassifier(ClassifierMixin, MetaEstimatorMixin, CostSensitiveMixin, BaseEstimator):  # type: ignore[misc]
    """
    Classifier that fits a cost-sensitive classifier with costs adjusted for outliers.

    The costs are adjusted by fitting an outlier estimator to the costs and imputing the costs for the outliers.
    Outliers are detected by the standardized residuals of the cost and the predicted cost.
    The costs passed to the cost-sensitive classifier are a combination of the original costs (not non-outliers) and
    the imputed predicted costs (for outliers).

    Read more in the :ref:`User Guide <robustcs>`.

    Parameters
    ----------
    estimator : Estimator
        The cost-sensitive classifier to fit.
        The estimator must take tp_cost, tn_cost, fn_cost, and fp_cost as keyword arguments in its fit method
        or should use :class:`~empulse.metrics.Metric` as their loss/criterion.

    outlier_estimator : Estimator, optional
        The outlier estimator to fit to the costs.

        If not provided, a :class:`sklearn:sklearn.linear_model.HuberRegressor` is used with default settings.
    outlier_threshold : float, default=2.5
        The threshold for the standardized residuals to detect outliers.
        If the absolute value of the standardized residual is greater than the threshold,
        the cost is an outlier and will be imputed with the predicted cost.

    detect_outliers_for : {'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', list}, default='all'
        The costs for which to detect outliers.
        By default, all instance-dependent costs are used for outlier detection.
        If a single cost is passed, only that cost is used for outlier detection.
        If a list of costs is passed, only those costs are used for outlier detection.
        tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

        .. note::
            This parameter is ignored if the underlying estimator
            uses :class:`~empulse.metrics.Metric` as its loss/criterion.
            Then all costs that are marked as outlier-sensitive in the metric loss
            are used for outlier detection.
            This can be done through the :meth: `~empulse.metrics.Metric.mark_outlier_sensitive` method.

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
    estimator_ : Estimator
        The fitted cost-sensitive classifier.
    outlier_estimators_ : dict{str, Estimator or None}
        The fitted outlier estimators.
        If no outliers are detected for this cost, the value is None.
        The keys of the directory are 'tp_cost', 'tn_cost', 'fn_cost', and 'fp_cost'.
    costs_ : dict
        The imputed costs for the cost-sensitive classifier.

    Notes
    -----
    Constant costs are not used for outlier detection and imputation.

    Code adapted from [1]_.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from empulse.models import CSLogitClassifier, RobustCSClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()
        fn_cost = np.random.rand(y.size)  # instance-dependent cost
        fp_cost = 5  # constant cost

        model = RobustCSClassifier(CSLogitClassifier(C=0.1))
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

    Example with Metric loss:

    .. code-block:: python

        import numpy as np
        import sympy as sp
        from empulse.metrics import Metric, Cost, CostMatrix
        from empulse.models import CSLogitClassifier, RobustCSClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification()
        a, b = sp.symbols('a b')
        cost_loss = Metric(
            CostMatrix().add_fp_cost(a).add_fn_cost(b).mark_outlier_sensitive(a), Cost()
        )
        fn_cost = np.random.rand(y.size)

        model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
        model.fit(X, y, a=np.random.rand(y.size), b=5)

    Example with passing instance-dependent costs through cross-validation:

    .. code-block:: python

        import numpy as np
        from empulse.models import CSBoostClassifier, RobustCSClassifier
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
            (
                'model',
                RobustCSClassifier(CSBoostClassifier()).set_fit_request(
                    fn_cost=True, fp_cost=True
                ),
            ),
        ])

        cross_val_score(pipeline, X, y, params={'fn_cost': fn_cost, 'fp_cost': fp_cost})

    Example with passing instance-dependent costs through a grid search:

    .. code-block:: python

        import numpy as np
        from empulse.metrics import expected_cost_loss
        from empulse.models import CSLogitClassifier, RobustCSClassifier
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
            (
                'model',
                RobustCSClassifier(CSLogitClassifier()).set_fit_request(
                    fn_cost=True, fp_cost=True
                ),
            ),
        ])
        param_grid = {'model__estimator__C': np.logspace(-5, 2, 5)}
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
    .. [1] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
           Robust instance-dependent cost-sensitive classification.
           Advances in Data Analysis and Classification, 17(4), 1057-1079.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'estimator': [HasMethods(['fit', 'predict_proba']), None],
        'outlier_estimator': [HasMethods(['fit', 'predict']), None],
        'outlier_threshold': [Interval(Real, 0, None, closed='right')],
        'detect_outliers_for': [StrOptions({'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'}), list],
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
    }

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        return self.estimator._get_metric_loss() if isinstance(self.estimator, CostSensitiveMixin) else None

    def __init__(
        self,
        estimator: Any,
        outlier_estimator: Any = None,
        *,
        outlier_threshold: float = 2.5,
        detect_outliers_for: Literal['all'] | CostStr | list[CostStr] = 'all',
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
    ):
        self.estimator = estimator
        self.outlier_estimator = outlier_estimator
        self.outlier_threshold = outlier_threshold
        self.detect_outliers_for = detect_outliers_for
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        super().__init__()

    def __post_init__(self) -> None:
        # Allow passing costs accepted by the metric loss through metadata routing
        if isinstance(self._get_metric_loss(), Metric):
            self.__class__.set_fit_request = RequestMethod(
                'fit',
                sorted(
                    self.__class__._get_default_requests().fit.requests.keys() | self._get_metric_loss()._all_symbols  # type: ignore[union-attr]
                ),
            )

    @_fit_context(prefer_skip_nested_validation=False)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **fit_params: Any,
    ) -> Self:
        """
        Fit the estimator with the adjusted costs.

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
        self : RobustCSLogitClassifier
            Fitted RobustCSLogitClassifier model.
        """
        X, y = validate_data(self, X, y)

        if (
            isinstance(self.estimator, CostSensitiveMixin)
            and (metric_loss := self.estimator._get_metric_loss()) is not None
        ):
            self.costs_: dict[str, int | float | FloatNDArray] = {}
            outlier_symbols = metric_loss.cost_matrix._outlier_sensitive_symbols
            imputed_costs = {}

            self.outlier_estimators_ = {}
            for symbol in outlier_symbols:
                target = fit_params.get(str(symbol))
                if target is None:
                    alias = _invert_dict(metric_loss.cost_matrix._aliases)[str(symbol)]
                    target = fit_params.get(alias)
                    if target is None:
                        raise ValueError(f"Cost '{symbol}' is not provided in fit_params.")
                if not isinstance(target, np.ndarray):
                    raise ValueError(f"Cost '{symbol}' is not an array. Cannot detect outliers for this cost.")
                pos_symbols = metric_loss.tp_cost.free_symbols | metric_loss.fn_cost.free_symbols
                neg_symbols = metric_loss.tn_cost.free_symbols | metric_loss.fp_cost.free_symbols
                if symbol in pos_symbols and symbol not in neg_symbols:
                    X_relevant, target_relevant = X[y > 0], target[y > 0]
                elif symbol in neg_symbols and symbol not in pos_symbols:
                    X_relevant, target_relevant = X[y == 0], target[y == 0]
                else:
                    X_relevant, target_relevant = X.copy(), target.copy()

                if X_relevant.size > 0:
                    outlier_estimator = clone(
                        self.outlier_estimator if self.outlier_estimator is not None else HuberRegressor()
                    ).fit(X_relevant, target_relevant)
                    cost_predictions = outlier_estimator.predict(X)
                    residuals = np.abs(target - cost_predictions)
                    std_residuals = residuals / st.sem(target)
                    outliers = std_residuals > self.outlier_threshold
                    fit_params[str(symbol)] = np.where(outliers, cost_predictions, target)
                    self.costs_[str(symbol)] = fit_params[str(symbol)]
                    self.outlier_estimators_[str(symbol)] = outlier_estimator
        else:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )
            self.costs_ = {
                'tp_cost': tp_cost if isinstance(tp_cost, int | float) else np.array(tp_cost),  # take copy of the array
                'tn_cost': tn_cost if isinstance(tn_cost, int | float) else np.array(tn_cost),
                'fn_cost': fn_cost if isinstance(fn_cost, int | float) else np.array(fn_cost),
                'fp_cost': fp_cost if isinstance(fp_cost, int | float) else np.array(fp_cost),
            }
            should_fit = self._determine_outlier_costs()
            self._fit_outlier_estimators(X, y, should_fit)
            imputed_costs = self.costs_.copy()

        # with the imputed costs fit the estimator
        self.estimator_ = clone(self.estimator).fit(X, y, **imputed_costs, **fit_params)

        if hasattr(self.estimator_, 'n_features_in_'):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, 'feature_names_in_'):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    def _determine_outlier_costs(self) -> list[str]:
        """Determine which costs to fit the outlier estimator on."""
        # only fit on the costs that are arrays and have a standard deviation greater than 0
        should_fit: list[str] = [
            cost_name for cost_name, cost in self.costs_.items() if isinstance(cost, np.ndarray) and np.std(cost) > 0
        ]

        if self.detect_outliers_for != 'all':
            if isinstance(self.detect_outliers_for, str):
                if self.detect_outliers_for in self.costs_:  # single cost
                    if self.detect_outliers_for not in should_fit:
                        raise ValueError(
                            f"Cost '{self.detect_outliers_for}' is not an array or has a standard deviation of 0."
                            ' Cannot detect outliers for this cost.'
                        )
                    should_fit = [self.detect_outliers_for]  # type: ignore[list-item]
                else:
                    raise ValueError(
                        f"Invalid cost name '{self.detect_outliers_for}' in detect_outliers_for."
                        " Must be one of 'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', or a list of these."
                    )
            elif isinstance(self.detect_outliers_for, list):
                for cost_name in self.detect_outliers_for:
                    if cost_name not in self.costs_:
                        raise ValueError(f"Invalid cost name '{cost_name}' in detect_outliers_for.")
                    if cost_name not in should_fit:
                        raise ValueError(
                            f"Cost '{cost_name}' is not an array or has a standard deviation of 0."
                            ' Cannot detect outliers for this cost.'
                        )
                should_fit = [cost_name for cost_name in self.detect_outliers_for if cost_name in should_fit]
            else:
                raise TypeError(
                    f"Invalid type '{type(self.detect_outliers_for)}' for detect_outliers_for."
                    " Must be one of 'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', or a list of these."
                )

        return should_fit

    def _fit_outlier_estimators(self, X: FloatNDArray, y: FloatNDArray, should_fit: list[str]) -> None:
        self.outlier_estimators_ = {}
        for cost_name in self.costs_:
            if cost_name in should_fit:
                target = self.costs_[cost_name]
                if not isinstance(target, np.ndarray):
                    raise ValueError(f"Cost '{cost_name}' is not an array. Cannot detect outliers for this cost.")
                if cost_name in {'tp_cost', 'fn_cost'}:
                    X_relevant, target_relevant = X[y > 0], target[y > 0]
                else:
                    X_relevant, target_relevant = X[y == 0], target[y == 0]

                if X_relevant.size > 0:
                    outlier_estimator = clone(
                        self.outlier_estimator if self.outlier_estimator is not None else HuberRegressor()
                    ).fit(X_relevant, target_relevant)
                    cost_predictions = outlier_estimator.predict(X)
                    residuals = np.abs(target - cost_predictions)
                    std_residuals = residuals / st.sem(target)
                    outliers = std_residuals > self.outlier_threshold
                    self.costs_[cost_name] = np.where(outliers, cost_predictions, target)
                    self.outlier_estimators_[cost_name] = outlier_estimator
                else:
                    self.outlier_estimators_[cost_name] = None
            else:
                self.outlier_estimators_[cost_name] = None

    @available_if(_estimator_has('predict'))  # type: ignore[misc]
    def predict(self, X: FloatArrayLike) -> FloatNDArray:  # noqa: D102
        check_is_fitted(self, 'estimator_')
        y_pred: FloatNDArray = self.estimator_.predict(X)
        return y_pred

    @available_if(_estimator_has('predict_proba'))  # type: ignore[misc]
    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:  # noqa: D102
        check_is_fitted(self, 'estimator_')
        y_proba: FloatNDArray = self.estimator_.predict_proba(X)
        return y_proba

    @available_if(_estimator_has('decision_function'))  # type: ignore[misc]
    def decision_function(self, X: FloatArrayLike) -> FloatNDArray:  # noqa: D102
        check_is_fitted(self, 'estimator_')
        y_score: FloatNDArray = self.estimator_.decision_function(X)
        return y_score

    @property
    def classes_(self) -> NDArray[Any]:  # noqa: D102
        classes: NDArray[Any] = self.estimator_.classes_
        return classes

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags


def _invert_dict(d: MutableMapping[K, V]) -> dict[V, K]:
    """Invert a dictionary, swapping keys and values."""
    return {v: k for k, v in d.items()}
