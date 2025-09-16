import sys
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
from sklearn.utils import Bunch
from sklearn.utils._param_validation import Hidden, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntArrayLike, IntNDArray, ParameterConstraint
from ...metrics import Metric
from ...utils._sklearn_compat import Tags, type_of_target, validate_data  # type: ignore[attr-defined]
from ._cs_mixin import CostSensitiveMixin
from ._impurity import CostImpurity, EntropyCostImpurity, GiniCostImpurity

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


TREE_PARAM_CONSTRAINTS = DecisionTreeClassifier._parameter_constraints.copy()
TREE_PARAM_CONSTRAINTS.pop('criterion')


class CSTreeClassifier(CostSensitiveMixin, ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """
    Cost-sensitive decision tree classifier.

    Parameters
    ----------
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

    loss : Metric or None, default=None
        The metric to measure the quality of a split.
        If None, the cost impurity is used.

    criterion : {"cost",, "gini", "log_loss" or "entropy"}, default="cost"
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

        .. versionchanged:: 0.18
           Added float values for fractions.

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
        See :term:`Sklearn Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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


    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
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

        Read more in the :ref:`Sklearn User Guide <monotonic_cst_gbdt>`.

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.DecisionTreeClassifier`
        The underlying DecisionTreeClassifier estimator.

    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           "Example-Dependent Cost-Sensitive Decision Trees. Expert Systems with Applications",
           Expert Systems with Applications, 42(19), 6609–6619, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **TREE_PARAM_CONSTRAINTS,
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'criterion': [
            StrOptions({'cost', 'log_loss', 'gini', 'entropy'}),
            Hidden(CostImpurity),
        ],
        'loss': [Metric, None],
    }

    def __init__(
        self,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
        criterion: Literal['cost', 'gini', 'entropy', 'log_loss'] = 'cost',
        splitter: Literal['best', 'random'] = 'best',
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Literal['auto', 'sqrt', 'log2'] | int | float | None = None,
        random_state: int | np.random.RandomState | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        class_weight: dict[int, float] | Literal['balanced'] | None = None,
        ccp_alpha: float = 0.0,
        monotonic_cst: IntArrayLike | None = None,
    ):
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
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
    def feature_importances_(self) -> FloatNDArray:
        """Return the feature importances."""
        check_is_fitted(self)
        importances: FloatNDArray = self.estimator_.feature_importances_
        return importances

    @property
    def max_features_(self) -> int:
        """Return the inferred value of max_features."""
        check_is_fitted(self)
        max_features: int = self.estimator_.max_features_
        return max_features

    @property
    def n_classes_(self) -> int:
        """Return the number of classes."""
        check_is_fitted(self)
        n_classes: int = self.estimator_.n_classes_
        return n_classes

    @property
    def n_outputs_(self) -> int:
        """The number of outputs when ``fit`` is performed."""
        check_is_fitted(self)
        n_outputs: int = self.estimator_.n_outputs_
        return n_outputs

    @property
    def tree_(self) -> Tree:
        """The underlying Tree object."""
        check_is_fitted(self)
        return self.estimator_.tree_

    def get_depth(self) -> int:
        """Return the depth of the decision tree."""
        check_is_fitted(self)
        depth: int = self.estimator_.get_depth()
        return depth

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the decision tree."""
        check_is_fitted(self)
        n_leaves: int = self.estimator_.get_n_leaves()
        return n_leaves

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
        elif isinstance(self.criterion, CostImpurity):
            self.criterion_ = self.criterion
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

        self.estimator_ = DecisionTreeClassifier(
            criterion=self.criterion_,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            monotonic_cst=self.monotonic_cst,
        )
        self.estimator_.fit(X, y)

        return self

    def predict(self, X: FloatArrayLike, check_input: bool = True) -> IntNDArray:
        """
        Predict class value for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        y_pred: IntNDArray = self.estimator_.predict(X, check_input=check_input)
        return y_pred

    def predict_proba(self, X: FloatArrayLike, check_input: bool = True) -> FloatNDArray:
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        y_proba: FloatNDArray = self.estimator_.predict_proba(X, check_input=check_input)
        return y_proba

    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        y_log_proba: FloatNDArray = self.estimator_.predict_log_proba(X)
        return y_log_proba

    def apply(self, X: FloatArrayLike, check_input: bool = True) -> IntNDArray:
        """
        Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

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
        check_is_fitted(self)
        X_leaves: IntNDArray = self.estimator_.apply(X, check_input=check_input)
        return X_leaves

    def cost_complexity_pruning_path(
        self, X: FloatArrayLike, y: IntArrayLike, sample_weight: FloatArrayLike | None = None
    ) -> Bunch:
        """
        Compute the pruning path during Minimal Cost-Complexity Pruning.

        See :ref:`minimal_cost_complexity_pruning` for details on the pruning process.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

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
        return self.estimator_.cost_complexity_pruning_path(X, y, sample_weight=sample_weight)

    def decision_path(self, X: FloatArrayLike, check_input: bool = True) -> csr_matrix:
        """
        Return the decision path in the tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        return self.estimator_.decision_path(X, check_input=check_input)
