from collections.abc import Collection
from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, check_is_fitted, clone
from sklearn.linear_model import HuberRegressor
from sklearn.utils import _safe_indexing
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import column_or_1d, indexable

from ..._common import Parameter
from ..._common._cost_routing import RoutesLossParameters
from ..._common._metaestimators import estimator_has
from ..._common._sklearn_compat import Tags, get_tags
from ..._types import FloatArrayLike, FloatNDArray, ParameterConstraint
from ...metrics import BaseMetric
from .._base.cost_sensitive import CostSensitiveClassifier
from .._base.wrapper import (
    BinaryClassifierTags,
    ClassesFromEstimator,
    PassesXThrough,
    final_estimator,
    fit_parameter_name,
)

CostStr = Literal['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost']
# The costs that describe positive samples, whose outliers are detected on those samples only.
POSITIVE_COSTS = frozenset({'tp_cost', 'fn_cost'})
CSCLASSIFIER_PARAMS = CostSensitiveClassifier._parameter_constraints.copy()
CSCLASSIFIER_PARAMS.pop('loss')


class RobustCSClassifier(  # type: ignore[misc]
    RoutesLossParameters, PassesXThrough, ClassesFromEstimator, BinaryClassifierTags, ClassifierMixin, BaseEstimator
):
    """
    Cost-sensitive classifier that is robust to outliers in the instance-dependent costs.

    The costs are adjusted by fitting an outlier estimator to the costs and imputing the costs for the outliers.
    Outliers are detected by the standardized residuals of the cost and the predicted cost.
    The costs passed to the cost-sensitive classifier are a combination of the original costs (for non-outliers) and
    the imputed predicted costs (for outliers).

    Read more in the :ref:`User Guide <robustcs>`.

    .. seealso::

        :class:`~empulse.models.CSLogitClassifier`, :class:`~empulse.models.CSBoostClassifier` :
        typical estimators to wrap, since both accept the four cost keyword arguments this
        meta-estimator imputes.

    Parameters
    ----------
    estimator : Estimator
        The cost-sensitive classifier to fit.
        The estimator must take tp_cost, tn_cost, fn_cost, and fp_cost as keyword arguments in its fit method
        or should use a :class:`~empulse.metrics.BaseMetric` (e.g. :class:`~empulse.metrics.Metric` or
        :class:`~empulse.metrics.MixtureMetric`) as their loss/criterion.

    outlier_estimator : Estimator, optional
        The outlier estimator to fit to the costs.

        If not provided, a :class:`sklearn:sklearn.linear_model.HuberRegressor` is used with default settings.
    outlier_threshold : float, default=2.5
        The threshold for the standardized residuals to detect outliers.
        A residual, the cost minus its predicted cost, is standardized by subtracting the mean of the
        residuals and dividing by their standard deviation. If the absolute standardized residual is
        greater than the threshold, the cost is an outlier and will be imputed with the predicted cost.
        The default therefore flags residuals more than 2.5 standard deviations from the mean.

    detect_outliers_for : {'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', list}, default='all'
        The costs for which to detect outliers.
        By default, all instance-dependent costs are used for outlier detection.
        If a single cost is passed, only that cost is used for outlier detection.
        If a list of costs is passed, only those costs are used for outlier detection.

        .. note::
            This parameter is ignored if the underlying estimator
            uses a :class:`~empulse.metrics.BaseMetric` (e.g. :class:`~empulse.metrics.Metric` or
            :class:`~empulse.metrics.MixtureMetric`) as its loss/criterion.
            Then all parameters marked outlier-sensitive in the metric loss's cost matrix
            (or matrices, for a :class:`~empulse.metrics.MixtureMetric`) are used for outlier detection.
            This can be done through the :meth:`~empulse.metrics.CostMatrix.mark_outlier_sensitive` method.

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

    Attributes
    ----------
    estimator_ : Estimator
        The fitted cost-sensitive classifier.
    outlier_estimators_ : dict{str, Estimator or None}
        The fitted outlier estimators.
        If no outliers are detected for this cost, the value is None.
        The keys of the dictionary are 'tp_cost', 'tn_cost', 'fn_cost', and 'fp_cost'.
    costs_ : dict
        The imputed costs for the cost-sensitive classifier.

    Notes
    -----
    Constant costs are not used for outlier detection and imputation.

    Code adapted from [1]_, with one deviation: [1]_ divides the absolute residuals by their own
    standard deviation, which is about 0.6 times that of the residuals when they are normally
    distributed, so a threshold of 2.5 there flags residuals beyond about 1.5 standard deviations.
    Here the residuals are standardized, so ``outlier_threshold`` is the number of standard deviations
    it says.

    References
    ----------
    .. [1] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
           Robust instance-dependent cost-sensitive classification.
           Advances in Data Analysis and Classification, 17(4), 1057-1079.

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
        )
        scorer = scorer.set_score_request(fn_cost=True, fp_cost=True)

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer)
        grid_search.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    """

    costs_: dict[str, Any]
    outlier_estimators_: dict[str, Any]

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CSCLASSIFIER_PARAMS,
        'estimator': [HasMethods(['fit', 'predict_proba']), None],
        'outlier_estimator': [HasMethods(['fit', 'predict']), None],
        'outlier_threshold': [Interval(Real, 0, None, closed='right')],
        'detect_outliers_for': [StrOptions({'all', 'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'}), list],
    }

    def _allows_nan(self) -> bool:
        """Whether ``X`` may hold missing values: the outlier estimator is fit on ``X`` too, so both must take them."""
        outlier_estimator = self.outlier_estimator if self.outlier_estimator is not None else HuberRegressor()
        return bool(get_tags(self.estimator).input_tags.allow_nan and get_tags(outlier_estimator).input_tags.allow_nan)

    def _more_tags(self) -> dict[str, Any]:
        return {**super()._more_tags(), 'allow_nan': self._allows_nan()}

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self._allows_nan()
        return tags

    def _get_metric_loss(self) -> BaseMetric | None:
        """Return the loss of the cost-sensitive estimator this wraps, the last step of a pipeline, if it has one."""
        classifier = final_estimator(self.estimator)
        return classifier._get_metric_loss() if isinstance(classifier, CostSensitiveClassifier) else None

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

        **fit_params : dict
            Additional keyword arguments to pass to the estimator's fit method.

        Returns
        -------
        self : RobustCSClassifier
            Fitted RobustCSClassifier model.
        """
        # X is the wrapped estimators' to validate: it may be a pipeline that encodes strings or needs the
        # feature names.
        X, y = indexable(X, y)
        y = column_or_1d(y, warn=True)
        if y.size == 0:
            raise ValueError('Found array with 0 sample(s) while a minimum of 1 is required.')
        # Costs refer to the wrapped estimator's positive class: its `pos_label` if it has one, else the
        # greater of the two labels (the wrapped cost-sensitive estimators' `classes_[1]`).
        pos_label = getattr(final_estimator(self.estimator), 'pos_label', None)
        is_positive = y == (np.unique(y)[-1] if pos_label is None else pos_label)

        metric_loss = self._get_metric_loss()

        if metric_loss is not None:
            estimator_params = self._route_costs_to_loss(
                metric_loss, fit_params, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
            )
            self.costs_, self.outlier_estimators_ = self._impute_metric_costs(
                X, is_positive, metric_loss, estimator_params
            )
            estimator_params = self._named_for_estimator(estimator_params, metric_loss.model.accepted_parameters)
        else:
            self.costs_ = self._check_costs(
                **self._costs_with_estimator_fallback(
                    tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
                )
            )
            should_fit = self._determine_outlier_costs()
            self._fit_outlier_estimators(X, is_positive, should_fit)
            estimator_params = {**fit_params, **self._named_for_estimator(self.costs_, self.costs_.keys())}

        self.estimator_ = clone(self.estimator).fit(X, y, **estimator_params)
        self._copy_feature_attributes()
        return self

    def _named_for_estimator(self, params: dict[str, Any], names: Collection[str]) -> dict[str, Any]:
        """Rename the *params* among *names* as the wrapped estimator's ``fit`` takes them, e.g. a pipeline's."""
        return {
            fit_parameter_name(self.estimator, key) if key in names else key: value for key, value in params.items()
        }

    def _costs_with_estimator_fallback(self, **costs: Any) -> dict[str, Any]:
        """
        Fill in the costs not passed to ``fit`` and left at zero here from the wrapped estimator's own.

        The resolved costs are passed to the wrapped estimator explicitly, so taking this
        estimator's default of zero would override costs set on the wrapped estimator instead.
        """
        resolved: dict[str, Any] = {}
        for name, value in costs.items():
            own = getattr(self, name)
            if value is Parameter.UNCHANGED and isinstance(own, Real) and own == 0:
                value = getattr(final_estimator(self.estimator), name, own)
            resolved[name] = value
        return resolved

    def _fit_and_impute(
        self,
        X: FloatNDArray,
        is_positive: NDArray[np.bool_],
        rows: Literal['positive', 'negative', 'both'],
        target: FloatNDArray,
    ) -> tuple[FloatNDArray, Any]:
        """
        Fit the outlier estimator on the *rows* of *target*, and return *target* with their outliers imputed.

        A cost of one class only enters the rows of that class, so the other rows are left as they are:
        their values neither change nor widen the spread an outlier is measured against. Also returns the
        fitted estimator, or ``None``, leaving *target* as it is, when there are no such rows.
        """
        selected = {'positive': is_positive, 'negative': ~is_positive, 'both': np.ones_like(is_positive)}[rows]
        if not selected.any():
            return target, None
        X_fit, target_fit = _safe_indexing(X, selected), target[selected]
        outlier_estimator = self.outlier_estimator if self.outlier_estimator is not None else HuberRegressor()
        outlier_estimator = clone(outlier_estimator).fit(X_fit, target_fit)
        imputed = np.array(target, dtype=np.float64)
        imputed[selected] = self._impute_outliers(outlier_estimator, X_fit, target_fit)
        return imputed, outlier_estimator

    def _impute_outliers(self, outlier_estimator: Any, X: FloatNDArray, target: FloatNDArray) -> FloatNDArray:
        """Predict costs with *outlier_estimator* and replace outlier values in *target*."""
        predictions = outlier_estimator.predict(X)
        residuals = target - predictions
        std = np.std(residuals)
        std_residuals = (residuals - np.mean(residuals)) / std if std > 0 else np.zeros_like(residuals)
        outliers = np.abs(std_residuals) > self.outlier_threshold
        return np.where(outliers, predictions, target)

    def _impute_metric_costs(
        self,
        X: FloatNDArray,
        is_positive: NDArray[np.bool_],
        metric_loss: BaseMetric,
        estimator_params: dict[str, Any],
    ) -> tuple[dict[str, FloatNDArray], dict[str, Any]]:
        """Impute outlier-sensitive metric costs and return (imputed_costs, outlier_estimators).

        *estimator_params* is updated in-place with imputed values for outlier-sensitive symbols.
        """
        imputed_costs: dict[str, FloatNDArray] = {}
        outlier_estimators: dict[str, Any] = {}

        for name, cls in metric_loss.model.outlier_sensitive_parameters().items():
            spellings = metric_loss.model.parameter_spellings(name)
            # The metric takes a parameter under its symbol name or any alias, and so does this.
            supplied = sorted(spellings & estimator_params.keys())
            if not supplied:
                raise ValueError(
                    f"Cost '{name}' is not provided in fit params (accepted as: {', '.join(sorted(spellings))})."
                )
            param_name = supplied[0]
            target = estimator_params[param_name]

            if isinstance(target, Real) or (target := np.asarray(target)).ndim == 0:
                raise TypeError(f"Cost '{param_name}' must be an array for outlier detection.")

            imputed, outlier_est = self._fit_and_impute(X, is_positive, cls, target)
            if outlier_est is not None:
                estimator_params[param_name] = imputed
                imputed_costs[param_name] = imputed
                outlier_estimators[param_name] = outlier_est

        return imputed_costs, outlier_estimators

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

    def _fit_outlier_estimators(self, X: FloatNDArray, is_positive: NDArray[np.bool_], should_fit: list[str]) -> None:
        self.outlier_estimators_ = {}
        for cost_name in self.costs_:
            outlier_est = None
            if cost_name in should_fit:
                rows: Literal['positive', 'negative'] = 'positive' if cost_name in POSITIVE_COSTS else 'negative'
                self.costs_[cost_name], outlier_est = self._fit_and_impute(X, is_positive, rows, self.costs_[cost_name])
            self.outlier_estimators_[cost_name] = outlier_est

    @available_if(estimator_has('predict'))  # type: ignore[misc]
    def predict(self, X: FloatArrayLike) -> FloatNDArray:
        """Predict class labels with the wrapped estimator fitted on cleaned costs."""
        check_is_fitted(self, 'estimator_')
        y_pred: FloatNDArray = self.estimator_.predict(X)
        return y_pred

    @available_if(estimator_has('predict_proba'))  # type: ignore[misc]
    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """Predict class probabilities with the wrapped estimator."""
        check_is_fitted(self, 'estimator_')
        y_proba: FloatNDArray = self.estimator_.predict_proba(X)
        return y_proba

    @available_if(estimator_has('decision_function'))  # type: ignore[misc]
    def decision_function(self, X: FloatArrayLike) -> FloatNDArray:
        """Return the wrapped estimator's decision function values."""
        check_is_fitted(self, 'estimator_')
        y_score: FloatNDArray = self.estimator_.decision_function(X)
        return y_score
