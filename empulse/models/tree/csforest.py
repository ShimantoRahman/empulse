import numbers
import warnings
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar, Literal, Self, cast

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..._common._sklearn_compat import Tags, validate_data
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric
from .._base.cost_sensitive import CostSensitiveClassifier
from .._base.ensemble_weighting import goodness_weights, subset_loss_params
from ._cstree import CostTree, cost_records
from ._cstree._grow import (
    TREE_PARAM_CONSTRAINTS,
    TreeParams,
    as_float32,
    grow_tree,
    missing_feature_mask,
    resolve_tree_params,
)
from .cstree import CSTreeClassifier

# The largest tree seed, as scikit-learn's forests draw them.
MAX_INT = np.iinfo(np.int32).max


class CSForestClassifier(CostSensitiveClassifier):
    """
    Cost-sensitive random forest classifier.

    A forest of cost-sensitive decision trees. Missing values (``NaN``) in ``X`` are supported,
    as in :class:`~empulse.models.CSTreeClassifier`.

    Read more in the :ref:`User Guide <csforest>`.

    .. seealso::

        :class:`~empulse.models.CSTreeClassifier` : Cost-sensitive decision tree classifier.

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression classifier.

        :class:`~empulse.models.CSBoostClassifier` : Cost-sensitive gradient boosting classifier.

        :class:`~empulse.models.CSBaggingClassifier` : Bags an ensemble of cost-sensitive trees.

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
        The metric to measure the quality of a split.
        If None, the cost impurity is used.

    criterion : {"cost", "gini", "log_loss" or "entropy"}, default="cost"
        The function to measure the quality of a split.

        How the measure to estimate the quality of a split is weighted.

        - If ``"cost"``: The metric is used normally, without extra weighting.
        - If ``"gini"``: The Gini impurity is used to weight the metric.
        - If ``"log_loss"`` or ``"entropy"``: The Shannon information gain is used to weight the metric.

    combination : {"majority_voting", "weighted_voting"}, default="majority_voting"
        How to combine the predictions of the individual models.

        - "majority_voting": the majority vote of the models.
        - "weighted_voting": the models are weighted by their oob score

        :meth:`predict` combines the trees' cost-optimal decisions this way, and
        :meth:`predict_proba` their class probabilities.

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

            N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)

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
        context. ``-1`` means using all processors. See :term:`Glossary <sklearn:n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <sklearn:random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`Glossary <sklearn:warm_start>` and
        :ref:`sklearn:tree_ensemble_warm_start` for details.

    class_weight : {"balanced", "balanced_subsample"} or dict, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, both classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        With ``bootstrap=True``, a dict or "balanced" weight is applied by drawing
        the bootstrap samples with probability proportional to it, as scikit-learn's
        random forests do.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning of every tree.
        The subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`sklearn:minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(int(n_samples * max_samples), 1)` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    Attributes
    ----------
    estimators_ : list of :class:`~empulse.models.CSTreeClassifier`
        The collection of fitted trees. They are fitted on the 0/1-encoded target.

    classes_ : ndarray of shape (2,)
        The class labels.

    n_classes_ : int
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit <sklearn:fit>`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit <sklearn:fit>`. Defined only when `X`
        has feature names that are all strings.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    estimator_weights_ : ndarray of shape (n_estimators,)
        The weight of each tree's vote, from its out-of-bag loss. This attribute
        exists only when ``combination="weighted_voting"``.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
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
        **CostSensitiveClassifier._parameter_constraints,
        **TREE_PARAM_CONSTRAINTS,
        'min_impurity_decrease': [Interval(Real, 0.0, None, closed='left')],
        'combination': [StrOptions({'majority_voting', 'weighted_voting'})],
        'n_estimators': [Interval(Integral, 1, None, closed='left')],
        'bootstrap': ['boolean'],
        'oob_score': ['boolean', callable],
        'n_jobs': [Integral, None],
        'verbose': ['verbose'],
        'warm_start': ['boolean'],
        'class_weight': [StrOptions({'balanced_subsample', 'balanced'}), dict, list, None],
        'max_samples': [
            None,
            Interval(RealNotInt, 0.0, None, closed='neither'),
            Interval(Integral, 1, None, closed='left'),
        ],
    }

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: BaseMetric | None = None,
        criterion: Literal['cost', 'gini', 'entropy', 'log_loss'] = 'cost',
        combination: Literal['majority_voting', 'weighted_voting'] = 'majority_voting',
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Literal['sqrt', 'log2'] | float | None = 'sqrt',
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool | Callable[[Any, Any], float] = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: bool | int = 0,
        warm_start: bool = False,
        class_weight: dict[int, float] | Literal['balanced', 'balanced_subsample'] | None = None,
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
    ):
        self.n_estimators = n_estimators
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
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    @property
    def n_classes_(self) -> int:
        """The number of classes seen during :term:`fit <sklearn:fit>`."""
        check_is_fitted(self)
        return 2

    @property
    def feature_importances_(self) -> FloatNDArray:
        """The impurity-based feature importances."""
        check_is_fitted(self)
        all_importances = [
            tree.tree_.compute_feature_importances() for tree in self.estimators_ if tree.tree_.node_count > 1
        ]
        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)
        importances: FloatNDArray = np.asarray(np.mean(all_importances, axis=0), dtype=np.float64)
        normalized: FloatNDArray = importances / importances.sum()
        return normalized

    @property
    def estimators_samples_(self) -> list[IntNDArray]:
        """The subset of drawn samples (i.e., the in-bag samples) for each base estimator."""
        check_is_fitted(self)
        return [self._drawn_samples(cast('int', tree.random_state)) for tree in self.estimators_]

    def _fit(
        self,
        X: FloatNDArray,
        y: IntNDArray,
        loss: BaseMetric,
        **loss_params: Any,
    ) -> Self:
        """
        Build an example-dependent cost-sensitive random forest from the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Ground truth (correct) labels.

        loss : BaseMetric
            Loss to be optimized.

        **loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.combination == 'weighted_voting' and not self.bootstrap:
            raise ValueError('Weighted voting is only available when bootstrap=True.')
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                '`max_sample` cannot be set if `bootstrap=False`. '
                'Either switch to `bootstrap=True` or set `max_sample=None`.'
            )
        if not self.bootstrap and self.oob_score:
            raise ValueError('Out of bag estimation only available if bootstrap=True')

        fp_cost, fn_cost, tp_cost, tn_cost = loss._evaluate_costs(replace_stochastic=True, **loss_params)
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape
        records = cost_records(y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)

        self._n_samples = n_samples
        self._sample_weight = self._class_sample_weight(y)
        self._n_samples_bootstrap = (
            _get_n_samples_bootstrap(n_samples, self.max_samples, self._sample_weight) if self.bootstrap else None
        )
        params = resolve_tree_params(
            n_samples=n_samples,
            n_features=n_features,
            criterion=self.criterion,
            splitter='best',
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
        )

        random_state = check_random_state(self.random_state)
        if not self.warm_start or not hasattr(self, 'estimators_'):
            self.estimators_: list[CSTreeClassifier] = []
        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators < 0:
            raise ValueError(
                f'n_estimators={self.n_estimators} must be larger or equal to '
                f'len(estimators_)={len(self.estimators_)} when warm_start==True'
            )
        if n_more_estimators == 0:
            warnings.warn('Warm-start fitting without increasing n_estimators does not fit new trees.', stacklevel=2)
        else:
            if self.warm_start and self.estimators_:
                # Draw the seeds the trees grown so far took, so the new trees get the seeds they would
                # have had in a single fit.
                random_state.randint(MAX_INT, size=len(self.estimators_))
            seeds = [random_state.randint(MAX_INT) for _ in range(n_more_estimators)]

            # Shared, read-only, by every tree.
            X_fortran = as_float32(X, fortran=True)
            missing_mask = missing_feature_mask(X_fortran)
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')(
                delayed(self._grow_one)(params, X_fortran, y, records, missing_mask, seed) for seed in seeds
            )
            self.estimators_.extend(
                _fitted_tree(tree, seed, params, n_features, self.criterion)
                for tree, seed in zip(trees, seeds, strict=True)
            )

        if self.oob_score or self.combination == 'weighted_voting':
            self._set_oob_attributes(as_float32(X), y, loss, n_more_estimators, **loss_params)
        return self

    def _class_sample_weight(self, y: IntNDArray) -> FloatNDArray | None:
        """Return the per-sample weight of the ``class_weight``, or ``None`` when each tree weighs classes."""
        if self.class_weight is None:
            return None
        if self.class_weight == 'balanced_subsample':
            if self.bootstrap:
                return None  # computed on each bootstrap sample
            return compute_sample_weight('balanced', y)  # type: ignore[no-any-return]
        return compute_sample_weight(self.class_weight, y)  # type: ignore[no-any-return]

    def _drawn_samples(self, seed: int) -> IntNDArray:
        """Return the bootstrap sample of the tree with ``seed``, drawn as when it was grown."""
        if not self.bootstrap:
            return np.arange(self._n_samples, dtype=np.int32)
        n_samples_bootstrap = cast('int', self._n_samples_bootstrap)
        return _generate_sample_indices(seed, self._n_samples, n_samples_bootstrap, self._sample_weight)

    def _grow_one(
        self,
        params: TreeParams,
        X: FloatNDArray,
        y: IntNDArray,
        records: FloatNDArray,
        missing_mask: NDArray[np.uint8] | None,
        seed: int,
    ) -> CostTree:
        sample_weight: FloatNDArray | None
        if self.bootstrap:
            indices = self._drawn_samples(seed)
            # The bootstrap counts are the sample weights: a sample drawn twice weighs two.
            sample_weight = np.bincount(indices, minlength=self._n_samples).astype(np.float64)
            if self.class_weight == 'balanced_subsample':
                sample_weight = sample_weight * compute_sample_weight('balanced', y, indices=indices)
        else:
            sample_weight = self._sample_weight
        return grow_tree(params, X, records, sample_weight, np.random.RandomState(seed), missing_mask)

    def _set_oob_attributes(
        self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, n_new_trees: int, **loss_params: Any
    ) -> None:
        """Set the out-of-bag score and decision function, and the trees' out-of-bag voting weights."""
        n_samples = y.shape[0]
        oob_proba = np.zeros((n_samples, 2), dtype=np.float64)
        n_oob = np.zeros(n_samples, dtype=np.int64)
        losses = np.empty(len(self.estimators_), dtype=np.float64)
        for i, tree in enumerate(self.estimators_):
            drawn = self._drawn_samples(cast('int', tree.random_state))
            unsampled = np.flatnonzero(np.bincount(drawn, minlength=n_samples) == 0)
            proba = tree.tree_.predict_proba(X[unsampled])
            oob_proba[unsampled] += proba
            n_oob[unsampled] += 1
            if self.combination == 'weighted_voting':
                oob_params = subset_loss_params(loss_params, unsampled, n_samples)
                losses[i] = loss._loss(y[unsampled], proba[:, 1], validate=False, **oob_params)

        if self.combination == 'weighted_voting':
            self.estimator_weights_ = goodness_weights(losses)

        if self.oob_score and (n_new_trees > 0 or not hasattr(self, 'oob_score_')):
            if (n_oob == 0).any():
                warnings.warn(
                    'Some inputs do not have OOB scores. This probably means too few trees were used to '
                    'compute any reliable OOB estimates.',
                    UserWarning,
                    stacklevel=3,
                )
                n_oob[n_oob == 0] = 1
            self.oob_decision_function_ = oob_proba / n_oob[:, None]
            scoring_function = self.oob_score if callable(self.oob_score) else accuracy_score
            self.oob_score_ = scoring_function(y, np.argmax(self.oob_decision_function_, axis=1))

    def _validate_X_predict(self, X: FloatArrayLike) -> FloatNDArray:
        check_is_fitted(self)
        return as_float32(validate_data(self, X, reset=False, ensure_all_finite='allow-nan'))

    def _tree_weights(self) -> FloatNDArray:
        if self.combination == 'weighted_voting':
            return np.asarray(self.estimator_weights_, dtype=np.float64)
        return np.ones(len(self.estimators_), dtype=np.float64)

    def _accumulate(self, X: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
        """Sum every tree's weighted class fractions and decisions, in parallel over chunks of trees."""
        weights = self._tree_weights()
        n_jobs = min(effective_n_jobs(self.n_jobs), len(self.estimators_))
        chunks = np.array_split(np.arange(len(self.estimators_)), n_jobs)

        def accumulate(chunk: IntNDArray) -> tuple[FloatNDArray, FloatNDArray]:
            proba = np.zeros((X.shape[0], 2), dtype=np.float64)
            votes = np.zeros(X.shape[0], dtype=np.float64)
            for i in chunk:
                self.estimators_[i].tree_.accumulate(X, weights[i], proba, votes)
            return proba, votes

        results = Parallel(n_jobs=n_jobs, verbose=self.verbose, prefer='threads')(
            delayed(accumulate)(chunk) for chunk in chunks
        )
        proba = np.zeros((X.shape[0], 2), dtype=np.float64)
        votes = np.zeros(X.shape[0], dtype=np.float64)
        for chunk_proba, chunk_votes in results:
            proba += chunk_proba
            votes += chunk_votes
        return proba, votes

    def predict(self, X: FloatArrayLike) -> NDArray[Any]:
        """
        Predict class labels for samples in X.

        Each tree votes for the class that costs least on the training samples in the leaf the sample
        falls in (see :meth:`CSTreeClassifier.predict <empulse.models.CSTreeClassifier.predict>`).
        With ``combination="majority_voting"`` every tree has one vote, and with
        ``"weighted_voting"`` its out-of-bag weight. A tie goes to the negative class.

        With instance-dependent costs, the votes use the costs of the training samples. To decide
        each sample by its own costs, threshold :meth:`predict_proba` with the loss's optimal
        threshold instead, e.g. with :class:`~empulse.models.CSThresholdClassifier`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels for each sample.
        """
        _, votes = self._accumulate(self._validate_X_predict(X))
        y_pred: NDArray[Any] = self.classes_.take((votes > self._tree_weights().sum() / 2).astype(np.intp))
        return y_pred

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class probabilities of the input samples X.

        The class fractions of the leaves the sample falls in, averaged over the trees, or with
        ``combination="weighted_voting"``, weighted by the trees' out-of-bag weights.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        prob : array of shape = [n_samples, 2]
            The class probabilities of the input samples.
        """
        proba, _ = self._accumulate(self._validate_X_predict(X))
        if self.combination != 'weighted_voting':
            proba = proba / len(self.estimators_)
        return proba

    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to ``dtype=np.float32``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_ <sklearn:classes_>`.
        """
        with np.errstate(divide='ignore'):
            y_log_proba: FloatNDArray = np.log(self.predict_proba(X))
        return y_log_proba

    def apply(self, X: FloatArrayLike) -> IntNDArray:
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to ``dtype=np.float32``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        leaves = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')(
            delayed(tree.tree_.apply)(X) for tree in self.estimators_
        )
        return np.array(leaves).T

    def decision_path(self, X: FloatArrayLike) -> tuple[csr_matrix, IntNDArray]:
        """
        Return the decision path in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to ``dtype=np.float32``.

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
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')(
            delayed(tree.tree_.decision_path)(X) for tree in self.estimators_
        )
        n_nodes_ptr = np.cumsum([0] + [indicator.shape[1] for indicator in indicators])
        return sparse_hstack(indicators).tocsr(), n_nodes_ptr


def _fitted_tree(tree: CostTree, seed: int, params: TreeParams, n_features: int, criterion: str) -> CSTreeClassifier:
    """Wrap a grown tree in a fitted :class:`CSTreeClassifier`, without fitting it again."""
    estimator = CSTreeClassifier(
        criterion=criterion,  # type: ignore[arg-type]
        max_depth=None if params.max_depth == np.iinfo(np.int32).max else params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        min_weight_fraction_leaf=params.min_weight_fraction_leaf,
        max_features=params.max_features,
        max_leaf_nodes=None if params.max_leaf_nodes < 0 else params.max_leaf_nodes,
        min_impurity_decrease=params.min_impurity_decrease,
        ccp_alpha=params.ccp_alpha,
        random_state=seed,
    )
    estimator.tree_ = tree
    estimator.classes_ = np.array([0, 1])
    estimator.n_features_in_ = n_features
    estimator.max_features_ = params.max_features
    estimator.min_impurity_decrease_ = params.min_impurity_decrease
    return estimator


def _generate_sample_indices(
    seed: int, n_samples: int, n_samples_bootstrap: int, sample_weight: FloatNDArray | None
) -> IntNDArray:
    """Draw a bootstrap sample, with probability proportional to ``sample_weight`` when given."""
    random_instance = np.random.RandomState(seed)
    if sample_weight is None:
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    else:
        sample_indices = random_instance.choice(
            n_samples, n_samples_bootstrap, replace=True, p=sample_weight / np.sum(sample_weight)
        )
    return sample_indices.astype(np.int32)  # type: ignore[no-any-return]


def _get_n_samples_bootstrap(n_samples: int, max_samples: float | None, sample_weight: FloatNDArray | None) -> int:
    """Return the number of samples each bootstrap sample draws."""
    if max_samples is None:
        return n_samples
    if isinstance(max_samples, numbers.Integral):
        return int(max_samples)
    weighted_n_samples = n_samples if sample_weight is None else float(np.sum(sample_weight))
    return max(int(max_samples * weighted_n_samples), 1)
