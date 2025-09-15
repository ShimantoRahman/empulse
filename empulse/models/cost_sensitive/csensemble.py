import sys
import threading
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar, Literal

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, issparse
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble._base import _partition_estimators
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._available_if import available_if
from sklearn.utils._mask import indices_to_mask
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import _estimator_has, check_is_fitted, check_random_state

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntArrayLike, IntNDArray, ParameterConstraint
from ...metrics import Metric, expected_cost_loss
from ...utils._sklearn_compat import Tags, type_of_target, validate_data  # type: ignore[attr-defined]
from ._cs_mixin import CostSensitiveMixin
from ._impurity import CostImpurity, EntropyCostImpurity, GiniCostImpurity
from .cstree import CSTreeClassifier

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

RF_PARAM_CONSTRAINTS = RandomForestClassifier._parameter_constraints.copy()
RF_PARAM_CONSTRAINTS.pop('criterion')


class CSForestClassifier(CostSensitiveMixin, ClassifierMixin, BaseEstimator):
    """
    Random Forest classifier to optimize instance-dependent cost loss.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

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

    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    loss : Metric or None, default=None
        The metric to measure the quality of a split.
        If None, the cost impurity is used.

    criterion : {"cost",, "gini", "log_loss" or "entropy"}, default="cost"
        The function to measure the quality of a split.

        How the measure to estimate quality of a split is weighted.

        - If ``"cost"``: The metric is used normally, without extra weighting.
        - If ``"gini"``: The Gini impurity is used to weight the metric.
        - If ``"log_loss"`` or ``"entropy"``: The Shannon information gain is used to weight the metric.

    combination : {"majority_voting', 'weighted_voting'}, default="majority_voting"
        How to combine the predictions of the individual models.

        - "majority_voting": the majority vote of the models.
        - "weighted_voting": the models are weighted by their oob score calculates with the ....

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, :func:`~sklearn.metrics.accuracy_score` is used.
        Provide a callable with signature `metric(y_true, y_pred)` to use a
        custom metric. Only available if `bootstrap=True`.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`Glossary <warm_start>` and
        :ref:`tree_ensemble_warm_start` for details.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details. See
        :ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`
        for an example of such pruning.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(round(n_samples * max_samples), 1)` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multiclass classifications (i.e. when `n_classes > 2`),
          - multioutput classifications (i.e. when `n_outputs_ > 1`),
          - classifications trained on data with missing values.

        The constraints hold over the probability of the positive class.

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.RandomForestClassifier`
        The underlying RandomForestClassifier estimator.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'criterion': [StrOptions({'cost', 'log_loss', 'gini', 'entropy'}), Metric],
        'loss': [Metric, None],
        'combination': [
            StrOptions({'majority_voting', 'weighted_voting'}),
        ],
        **RF_PARAM_CONSTRAINTS,
    }

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
        criterion: Literal['cost', 'gini', 'entropy', 'log_loss'] = 'cost',
        combination: Literal['majority_voting', 'weighted_voting'] = 'majority_voting',
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Literal['sqrt', 'log2'] | int | float = 'sqrt',
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool | Callable[[Any, Any], float] = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: bool | int = 0,
        warm_start: bool = False,
        class_weight: dict[int, float] | Literal['balanced'] | None = None,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None,
        monotonic_cst: IntArrayLike | None = None,
    ):
        self.n_estimators = n_estimators
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
        self.criterion = criterion
        self.combination = combination
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.monotonic_cst = monotonic_cst
        super().__init__()

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

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None

    @property
    def estimators_(self) -> list[DecisionTreeClassifier]:
        """The collection of fitted sub-estimators."""
        check_is_fitted(self)
        estimators: list[DecisionTreeClassifier] = self.estimator_.estimators_
        return estimators

    @property
    def n_classes_(self) -> int | list[int]:
        """The number of classes seen during :term:`fit`."""
        check_is_fitted(self)
        n_classes: int | list[int] = self.estimator_.n_classes_
        return n_classes

    @property
    def feature_importances_(self) -> FloatNDArray:
        """The impurity-based feature importances."""
        check_is_fitted(self)
        importances: FloatNDArray = self.estimator_.feature_importances_
        return importances

    @property
    def oob_score_(self) -> float:
        """Score of the training dataset obtained using an out-of-bag estimate."""
        check_is_fitted(self)
        oob_score: float = self.estimator_.oob_score_
        return oob_score

    @property
    def oob_decision_function_(self) -> FloatNDArray:
        """Decision function computed with out-of-bag estimate on the training set."""
        check_is_fitted(self)
        oob_decision_function: FloatNDArray = self.estimator_.oob_decision_function_
        return oob_decision_function

    @property
    def estimators_samples_(self) -> list[IntNDArray]:
        """The subset of drawn samples (i.e., the in-bag samples) for each base estimator."""
        check_is_fitted(self)
        estimators_samples: list[IntNDArray] = self.estimator_.estimators_samples_
        return estimators_samples

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: IntArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> Self:
        """
        Build an example-dependent cost-sensitive decision tree from the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Ground truth (correct) labels.

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

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                f'Unknown label type: Only binary classification is supported. The type of the target is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        y = np.where(y == self.classes_[1], 1, 0)

        if isinstance(self.loss, Metric):
            fp_cost, fn_cost, tp_cost, tn_cost = self.loss._evaluate_costs(**loss_params)
        else:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )

        n_samples = X.shape[0]
        for name, cost in zip(
            ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=True
        ):
            if isinstance(cost, np.ndarray) and cost.shape[0] != n_samples:
                raise ValueError(f'{name} has shape {cost.shape}, but should have shape ({n_samples},)')

        if self.criterion == 'cost':
            self.criterion_ = CostImpurity(
                n_outputs=1,
                n_classes=np.array([2], dtype=np.intp),
            )
        elif self.criterion == 'gini':
            self.criterion_ = GiniCostImpurity(
                n_outputs=1,
                n_classes=np.array([2], dtype=np.intp),
            )
        elif self.criterion in {'entropy', 'log_loss'}:
            self.criterion_ = EntropyCostImpurity(
                n_outputs=1,
                n_classes=np.array([2], dtype=np.intp),
            )
        else:
            raise ValueError(f'Unknown criterion: {self.criterion}')

        self.criterion_.set_costs(
            tp_cost=tp_cost if not isinstance(tp_cost, np.ndarray) else 0.0,
            tn_cost=tn_cost if not isinstance(tn_cost, np.ndarray) else 0.0,
            fp_cost=fp_cost if not isinstance(fp_cost, np.ndarray) else 0.0,
            fn_cost=fn_cost if not isinstance(fn_cost, np.ndarray) else 0.0,
        )
        self.criterion_.set_array_costs(
            tp_cost=tp_cost.reshape(-1).astype(np.float64)
            if isinstance(tp_cost, np.ndarray)
            else np.array([], dtype=np.float64),
            tn_cost=tn_cost.reshape(-1).astype(np.float64)
            if isinstance(tn_cost, np.ndarray)
            else np.array([], dtype=np.float64),
            fp_cost=fp_cost.reshape(-1).astype(np.float64)
            if isinstance(fp_cost, np.ndarray)
            else np.array([], dtype=np.float64),
            fn_cost=fn_cost.reshape(-1).astype(np.float64)
            if isinstance(fn_cost, np.ndarray)
            else np.array([], dtype=np.float64),
            n_samples=n_samples,
        )

        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion_,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            monotonic_cst=self.monotonic_cst,
        )
        self.estimator_.fit(X, y)

        if self.combination == 'weighted_voting':
            if not self.bootstrap:
                raise ValueError('Weighted voting is only available when bootstrap=True.')
            if self.loss is None:
                self.estimator_weights_ = self._get_oob_weights(
                    X,
                    y,
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                    check_input=False,
                )
            else:
                self.estimator_weights_ = self._get_oob_weights(X, y, **loss_params)
        return self

    def predict(self, X: FloatArrayLike) -> IntNDArray:
        """
        Predict class of X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes,
        """
        check_is_fitted(self)
        y_proba = self.predict_proba(X)
        y_pred: IntNDArray = self.classes_.take(np.argmax(y_proba, axis=1), axis=0)
        return y_pred

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        prob : array of shape = [n_samples, 2]
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X: FloatNDArray = validate_data(self, X, reset=False)

        if self.combination == 'weighted_voting':
            y_proba: FloatNDArray = self._predict_weighted_proba(X)
        else:
            y_proba = self.estimator_.predict_proba(X)

        return y_proba

    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        y_proba = self.predict_proba(X)
        return np.log(y_proba)

    def apply(self, X: FloatArrayLike) -> IntNDArray:
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        check_is_fitted(self)
        X_leaves: IntNDArray = self.estimator_.apply(X)
        return X_leaves

    def decision_path(self, X: FloatArrayLike) -> tuple[csr_matrix, IntNDArray]:
        """
        Return the decision path in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        check_is_fitted(self)
        indicator: csr_matrix
        n_nodes_ptr: IntNDArray
        indicator, n_nodes_ptr = self.estimator_.decision_path(X)
        return indicator, n_nodes_ptr

    def _get_oob_weights(self, X: FloatNDArray, y: IntNDArray, **kwargs: Any) -> FloatNDArray:
        # Prediction requires X to be in CSR format
        if issparse(X):
            X = X.tocsr()  # type: ignore[attr-defined]
        X = X.astype(np.float32)

        n_samples = y.shape[0]
        estimator_weights = np.zeros(self.n_estimators, dtype=np.float64)

        if self.max_samples is None:
            n_samples_bootstrap = n_samples

        if isinstance(self.max_samples, Integral):
            if self.max_samples > n_samples:
                msg = '`max_samples` must be <= n_samples={} but got value {}'
                raise ValueError(msg.format(n_samples, self.max_samples))
            n_samples_bootstrap = self.max_samples

        if isinstance(self.max_samples, Real):
            n_samples_bootstrap = max(round(n_samples * self.max_samples), 1)

        weight_fn = self.loss if isinstance(self.loss, Metric) else expected_cost_loss

        for i, estimator in enumerate(self.estimators_):
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state,
                n_samples,
                n_samples_bootstrap,
            )

            y_pred = self.estimator_._get_oob_predictions(estimator, X[unsampled_indices, :])
            estimator_weights[i] += weight_fn(y[unsampled_indices], y_pred[:, 1, 0], **kwargs)

        estimator_weights /= estimator_weights.sum()

        return estimator_weights

    def _predict_weighted_proba(self, X: FloatArrayLike) -> FloatNDArray:
        X: FloatNDArray = self.estimator_._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)  # type: ignore[arg-type]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require='sharedmem')(
            delayed(_accumulate_weighted_prediction)(e.predict_proba, X, all_proba, weight, lock)
            for e, weight in zip(self.estimators_, self.estimator_weights_, strict=True)
        )

        return all_proba


def _generate_unsampled_indices(random_state: int, n_samples: int, n_samples_bootstrap: int) -> IntNDArray:
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples, n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices: IntNDArray = indices_range[unsampled_mask]

    return unsampled_indices


def _generate_sample_indices(random_state: int, n_samples: int, n_samples_bootstrap: int) -> IntNDArray:
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices: IntNDArray = random_instance.randint(0, n_samples, n_samples_bootstrap, dtype=np.int32)

    return sample_indices


def _accumulate_weighted_prediction(
    predict: Callable[..., FloatNDArray],
    X: FloatArrayLike,
    out: FloatNDArray,
    weight: float,
    lock: threading.Lock,
) -> None:
    """Calculate the weighted prediction."""
    prediction = predict(X, check_input=False)
    with lock:
        out += prediction * weight  # type: ignore[misc]


class CSBaggingClassifier(CostSensitiveMixin, ClassifierMixin, BaseEstimator):
    """
    Bagging classifier to optimize instance-dependent cost loss.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregates their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a :class:`~empulse.models.CSTreeClassifier`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

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

    loss : Metric, default=None
        The loss function to use in order to evaluate the costs.
        If ``None``, then the costs provided to the constructor or to the ``fit``
        method are used directly.
        If a :class:``~empulse.metrics.Metric`` is provided, then the costs are computed using the
        metric, and any costs provided to the
        constructor or to the ``fit`` method are ignored.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see ``bootstrap`` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see ``bootstrap_features`` for more
        details).

        - If int, then draw ``max_features`` features.
        - If float, then draw ``max(1, int(max_features * n_features_in_))`` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimator_: estimator
        The base estimator from which the ensemble is grown.

    estimators_: list of estimators
        The collection of fitted base estimators.

    estimators_samples_: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    estimators_features_: list of arrays
        The subset of drawn features for each base estimator.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    .. [5] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'loss': [Metric, None],
        'combination': [
            StrOptions({'majority_voting', 'weighted_voting'}),
        ],
        **BaggingClassifier._parameter_constraints,
    }

    @property
    def estimators_(self) -> list[DecisionTreeClassifier]:
        """The collection of fitted sub-estimators."""
        check_is_fitted(self)
        estimators: list[DecisionTreeClassifier] = self.estimator_.estimators_
        return estimators

    @property
    def n_classes_(self) -> IntNDArray:
        """The number of classes seen during :term:`fit`."""
        check_is_fitted(self)
        n_classes: int | list[int] = self.estimator_.n_classes_
        n_classes_ = np.array(n_classes)
        return n_classes_

    @property
    def oob_score_(self) -> float:
        """Score of the training dataset obtained using an out-of-bag estimate."""
        check_is_fitted(self)
        oob_score: float = self.estimator_.oob_score_
        return oob_score

    @property
    def oob_decision_function_(self) -> FloatNDArray:
        """Decision function computed with out-of-bag estimate on the training set."""
        check_is_fitted(self)
        oob_decision_function: FloatNDArray = self.estimator_.oob_decision_function_
        return oob_decision_function

    @property
    def estimators_samples_(self) -> list[IntNDArray]:
        """The subset of drawn samples (i.e., the in-bag samples) for each base estimator."""
        check_is_fitted(self)
        estimators_samples: list[IntNDArray] = self.estimator_.estimators_samples_
        return estimators_samples

    @property
    def estimators_features_(self) -> list[IntNDArray]:
        """The subset of drawn features (i.e., the in-bag samples) for each base estimator."""
        check_is_fitted(self)
        estimators_samples: list[IntNDArray] = self.estimator_.estimators_features_
        return estimators_samples

    def __init__(
        self,
        estimator: Any = None,
        n_estimators: int = 10,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
        combination: Literal['majority_voting', 'weighted_voting'] = 'majority_voting',
        max_samples: float | int = 1.0,
        max_features: int | float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
        self.combination = combination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

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

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: IntArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> Self:
        """
        Build a cost-sensitive bagging classifier from the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            Ground truth (correct) labels.

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

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        self : CSBaggingClassifier
            Returns self.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                f'Unknown label type: Only binary classification is supported. The type of the target is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        y = np.where(y == self.classes_[1], 1, 0)

        if isinstance(self.loss, Metric):
            fp_cost, fn_cost, tp_cost, tn_cost = self.loss._evaluate_costs(**loss_params)
        else:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )

        n_samples = X.shape[0]
        for name, cost in zip(
            ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=True
        ):
            if isinstance(cost, np.ndarray) and cost.shape[0] != n_samples:
                raise ValueError(f'{name} has shape {cost.shape}, but should have shape ({n_samples},)')

        if self.estimator is None:
            criterion = CostImpurity(
                n_outputs=1,
                n_classes=np.array([2], dtype=np.intp),
            )
            criterion.set_costs(
                tp_cost=tp_cost if not isinstance(tp_cost, np.ndarray) else 0.0,
                tn_cost=tn_cost if not isinstance(tn_cost, np.ndarray) else 0.0,
                fp_cost=fp_cost if not isinstance(fp_cost, np.ndarray) else 0.0,
                fn_cost=fn_cost if not isinstance(fn_cost, np.ndarray) else 0.0,
            )
            criterion.set_array_costs(
                tp_cost=tp_cost.reshape(-1).astype(np.float64)
                if isinstance(tp_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                tn_cost=tn_cost.reshape(-1).astype(np.float64)
                if isinstance(tn_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                fp_cost=fp_cost.reshape(-1).astype(np.float64)
                if isinstance(fp_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                fn_cost=fn_cost.reshape(-1).astype(np.float64)
                if isinstance(fn_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                n_samples=n_samples,
            )
            self.base_estimator_ = CSTreeClassifier(criterion=criterion)
        else:
            self.base_estimator_ = clone(self.estimator)

        with config_context(enable_metadata_routing=True):
            self.estimator_ = BaggingClassifier(
                estimator=self.base_estimator_.set_fit_request(tp_cost=True, fp_cost=True, tn_cost=True, fn_cost=True),
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                max_samples=self.max_samples,
            )
            self.estimator_.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)

        if self.combination == 'weighted_voting':
            if not self.bootstrap:
                raise ValueError('Weighted voting is only available when bootstrap=True.')
            if self.loss is None:
                self.estimator_weights_ = self._get_oob_weights(
                    X,
                    y,
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                    check_input=False,
                )
            else:
                self.estimator_weights_ = self._get_oob_weights(X, y, **loss_params)
        return self

    def predict(self, X: FloatArrayLike) -> IntNDArray:
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        y_proba = self.predict_proba(X)
        y_pred: IntNDArray = self.classes_.take(np.argmax(y_proba, axis=1), axis=0)
        return y_pred

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X: FloatNDArray = validate_data(self, X, reset=False)

        if self.combination == 'weighted_voting':
            y_proba: FloatNDArray = self._predict_weighted_proba(X)
        else:
            y_proba = self.estimator_.predict_proba(X)

        return y_proba

    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        y_proba = self.predict_proba(X)
        return np.log(y_proba)

    @available_if(_estimator_has('decision_function', delegates=('base_estimator_', 'estimator')))
    def decision_function(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : ndarray of shape (n_samples, 1)
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``.
        """
        check_is_fitted(self)
        decisions: FloatNDArray = self.estimator_.decision_function(X)
        return decisions

    def _get_oob_weights(self, X: FloatNDArray, y: IntNDArray, **loss_params: Any) -> FloatNDArray:
        n_samples = y.shape[0]

        estimator_weights = np.zeros(self.n_estimators, dtype=np.float64)
        weight_fn = self.loss if self.loss is not None else expected_cost_loss

        for i, estimator, samples, features in zip(
            range(self.n_estimators), self.estimators_, self.estimators_samples_, self.estimators_features_, strict=True
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba((X[mask, :])[:, features])[:, 1]
            else:
                y_pred = estimator.predict((X[mask, :])[:, features])
            estimator_weights[i] = weight_fn(y[mask], y_pred, **loss_params)

        estimator_weights /= estimator_weights.sum()

        return estimator_weights

    def _predict_weighted_proba(self, X: FloatNDArray) -> FloatNDArray:
        X = validate_data(self, X, reset=False)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require='sharedmem')(
            delayed(_accumulate_weighted_prediction_non_tree)(e.predict_proba, X, all_proba, weight, lock)
            for e, weight in zip(self.estimators_, self.estimator_weights_, strict=True)
        )

        return all_proba


def _accumulate_weighted_prediction_non_tree(
    predict: Callable[[FloatNDArray], FloatNDArray],
    X: FloatNDArray,
    out: FloatNDArray,
    weight: float,
    lock: threading.Lock,
) -> None:
    """Calculate the weighted prediction."""
    prediction = predict(X)
    with lock:
        out += prediction * weight  # type: ignore[misc]
