from typing import Any, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.utils import Bunch, check_random_state, compute_sample_weight
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from ..._common._sklearn_compat import validate_data
from ..._types import FloatArrayLike, FloatNDArray, IntArrayLike, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric
from .._base.cost_sensitive import CostSensitiveClassifier
from ._cstree import CostTree, ccp_pruning_path, cost_records
from ._cstree._grow import TREE_PARAM_CONSTRAINTS, TreeParams, as_float32, grow_tree, resolve_tree_params

# The smallest cost decrease per sample, relative to the average cost per sample, that
# `min_impurity_decrease=None` counts as a real decrease. On a million samples, rounding errors
# stayed below 1e-16 while the smallest real decrease was 2e-10, so this sits well clear of both.
_RELATIVE_MIN_COST_DECREASE = 1e-12


class CSTreeClassifier(CostSensitiveClassifier):  # type: ignore[misc]
    """
    Cost-sensitive decision tree classifier.

    Trees are split based on a cost-sensitive impurity measure.

    Read more in the :ref:`User Guide <cstree>`.

    .. seealso::

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression classifier.

        :class:`~empulse.models.CSBoostClassifier` : Cost-sensitive gradient boosting classifier.

        :class:`~empulse.models.CSForestClassifier` : Cost-sensitive random forest classifier.

        :class:`~empulse.models.CSBaggingClassifier` : Bags an ensemble of cost-sensitive trees.

    Parameters
    ----------
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

        How the measure to estimate quality of a split is weighted.

        - If ``"cost"``: The metric is used normally, without extra weighting.
        - If ``"gini"``: The Gini impurity is used to weight the metric.
        - If ``"log_loss"`` or ``"entropy"``: The Shannon information gain is used to weight the metric.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node.
        Supported strategies are "best" to choose the best split and "random" to choose the best random split.

    max_depth : int or None, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.

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

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at
          each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. note::

            The search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Sklearn Glossary <sklearn:random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float or None, default=None
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        ``None`` means that with ``criterion="cost"`` a node is only split if the split lowers the
        cost of the training samples, and otherwise means ``0.0``. The cost impurity is the cost of
        the node's best single decision, so a split whose children both keep their parent's
        decision leaves it unchanged. With ``0.0``, such splits are still made, and the tree keeps
        splitting until every leaf holds one class, often a hundred or more levels deep, which
        overfits the costs of the training samples. The threshold ``None`` resolves to is
        ``1e-12`` times the average cost per sample, to tell a real decrease from rounding.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    class_weight : dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, both classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`sklearn:minimal_cost_complexity_pruning` for details.

        With ``criterion="cost"``, the impurities summed over the leaves are the
        cost of the training samples per unit of training weight, so ``ccp_alpha``
        is the smallest decrease of that cost an extra leaf has to bring.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [1]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit <sklearn:fit>`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit <sklearn:fit>`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed. Always ``1`` for this
        binary classifier; kept for scikit-learn compatibility.

    tree_ : CostTree
        The fitted tree, as parallel arrays indexed by node. Its attributes carry
        the names of scikit-learn's ``Tree`` (``children_left``, ``feature``,
        ``threshold``, ``impurity``, ``value``, ...), so
        :func:`sklearn.tree.plot_tree` and :func:`sklearn.tree.export_graphviz`
        accept the fitted classifier. ``tree_.positive`` holds the decision of
        every node: whether predicting it positive costs least on its training
        samples.

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           "Example-Dependent Cost-Sensitive Decision Trees",
           Expert Systems with Applications, 42(19), 6609–6619, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **TREE_PARAM_CONSTRAINTS,
        **CostSensitiveClassifier._parameter_constraints,
        'splitter': [StrOptions({'best', 'random'})],
        'class_weight': [dict, list, StrOptions({'balanced'}), None],
    }

    def __init__(
        self,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: BaseMetric | None = None,
        criterion: Literal['cost', 'gini', 'entropy', 'log_loss'] = 'cost',
        splitter: Literal['best', 'random'] = 'best',
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Literal['sqrt', 'log2'] | float | None = None,
        random_state: int | np.random.RandomState | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float | None = None,
        class_weight: dict[int, float] | Literal['balanced'] | None = None,
        ccp_alpha: float = 0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    @property
    def feature_importances_(self) -> FloatNDArray:
        """The impurity-based feature importances."""
        check_is_fitted(self)
        importances: FloatNDArray = self.tree_.compute_feature_importances()
        return importances

    @property
    def n_classes_(self) -> int:
        """The number of classes."""
        check_is_fitted(self)
        return 2

    @property
    def n_outputs_(self) -> int:
        """The number of outputs when ``fit`` is performed."""
        check_is_fitted(self)
        return 1

    def get_depth(self) -> int:
        """Return the depth of the decision tree."""
        check_is_fitted(self)
        return int(self.tree_.max_depth)

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the decision tree."""
        check_is_fitted(self)
        return int(self.tree_.n_leaves)

    def _fit(
        self,
        X: FloatNDArray,
        y: IntArrayLike,
        loss: BaseMetric,
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

        loss : BaseMetric
            Loss to be optimized.

        **loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        self : object
            Returns self.
        """
        fp_cost, fn_cost, tp_cost, tn_cost = loss._evaluate_costs(replace_stochastic=True, **loss_params)
        y = np.asarray(y).reshape(-1)
        records = cost_records(y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
        sample_weight = None if self.class_weight is None else compute_sample_weight(self.class_weight, y)

        self.min_impurity_decrease_ = self._resolve_min_impurity_decrease(y, tp_cost, tn_cost, fp_cost, fn_cost)
        params = self._tree_params(n_samples=X.shape[0], n_features=X.shape[1])
        self.max_features_ = params.max_features
        self.tree_: CostTree = grow_tree(
            params,
            as_float32(X, fortran=True),
            records,
            sample_weight,
            check_random_state(self.random_state),
        )
        return self

    def _tree_params(self, *, n_samples: int, n_features: int) -> TreeParams:
        """Resolve the hyperparameters against data of the given shape, once ``min_impurity_decrease_`` is set."""
        return resolve_tree_params(
            n_samples=n_samples,
            n_features=n_features,
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease_,
            ccp_alpha=self.ccp_alpha,
        )

    def _resolve_min_impurity_decrease(
        self,
        y: IntArrayLike,
        tp_cost: FloatNDArray | float,
        tn_cost: FloatNDArray | float,
        fp_cost: FloatNDArray | float,
        fn_cost: FloatNDArray | float,
    ) -> float:
        """Return ``min_impurity_decrease``, resolving ``None`` as its docstring describes."""
        if self.min_impurity_decrease is not None:
            return float(self.min_impurity_decrease)
        if self.criterion != 'cost':
            return 0.0
        is_positive = np.asarray(y).reshape(-1) == 1
        # The cost of predicting each sample positive, and negative.
        positive_cost = np.where(is_positive, tp_cost, fp_cost)
        negative_cost = np.where(is_positive, fn_cost, tn_cost)
        cost_scale = float(np.mean(np.abs(positive_cost)) + np.mean(np.abs(negative_cost)))
        return _RELATIVE_MIN_COST_DECREASE * cost_scale

    def _validate_X_predict(self, X: FloatArrayLike, check_input: bool) -> FloatNDArray:
        check_is_fitted(self)
        if check_input:
            X = validate_data(self, X, reset=False)
        return as_float32(X)

    def predict(self, X: FloatArrayLike, check_input: bool = True) -> NDArray[Any]:
        """
        Predict class value for X.

        Each leaf predicts the class that costs least on the training samples that fell in it,
        whatever the split ``criterion``. A leaf where both classes cost the same predicts its more
        frequent class. With instance-dependent costs, these are the costs of the
        training samples, so every sample in a leaf gets the same prediction. To decide each sample
        by its own costs, threshold :meth:`predict_proba` with the loss's optimal threshold instead,
        e.g. with :class:`~empulse.models.CSThresholdClassifier`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        X = self._validate_X_predict(X, check_input)
        encoded = self.tree_.predict_positive(X).astype(np.intp)
        y_pred: NDArray[Any] = self.classes_.take(encoded)
        return y_pred

    def predict_proba(self, X: FloatArrayLike, check_input: bool = True) -> FloatNDArray:
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same class in a leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_ <sklearn:classes_>`.
        """
        X = self._validate_X_predict(X, check_input)
        y_proba: FloatNDArray = self.tree_.predict_proba(X)
        return y_proba

    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_ <sklearn:classes_>`.
        """
        with np.errstate(divide='ignore'):
            y_log_proba: FloatNDArray = np.log(self.predict_proba(X))
        return y_log_proba

    def apply(self, X: FloatArrayLike, check_input: bool = True) -> IntNDArray:
        """
        Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        X = self._validate_X_predict(X, check_input)
        X_leaves: IntNDArray = self.tree_.apply(X)
        return X_leaves

    def cost_complexity_pruning_path(self, X: FloatArrayLike, y: IntArrayLike, **fit_params: Any) -> Bunch:
        """
        Compute the pruning path during Minimal Cost-Complexity Pruning.

        Fits an unpruned copy of this classifier (``ccp_alpha=0``) on ``X`` and ``y``, and returns
        its pruning path. See :ref:`sklearn:minimal_cost_complexity_pruning` for details on the
        pruning process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        **fit_params : dict
            Passed on to :meth:`fit`, such as instance-dependent costs.

        Returns
        -------
        ccp_path : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            ccp_alphas : ndarray
                Effective alphas of subtree during pruning.

            impurities : ndarray
                Sum of the impurities of the subtree leaves for the
                corresponding alpha value in ``ccp_alphas``.
        """
        unpruned = clone(self).set_params(ccp_alpha=0.0).fit(X, y, **fit_params)
        return Bunch(**ccp_pruning_path(unpruned.tree_))

    def decision_path(self, X: FloatArrayLike, check_input: bool = True) -> csr_matrix:
        """
        Return the decision path in the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        indicator: csr_matrix = self.tree_.decision_path(X)
        return indicator
