import copy
from math import ceil
from numbers import Integral, Real
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from ..._common import Parameter
from ...metrics import cost_loss
from ._cs_mixin import CostSensitiveMixin


class CSTreeClassifier(CostSensitiveMixin, ClassifierMixin, BaseEstimator):
    """
    Decision tree classifier to optimize instance-dependent cost loss.

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

    criterion : {"direct_cost", "pi_cost", "gini_cost" or "entropy_cost"}, default="direct_cost"
        The function to measure the quality of a split. Supported criteria are
        "direct_cost" for the Direct Cost impurity measure, "pi_cost", "gini_cost",
        and "entropy_cost".

    criterion_weight : bool, default=False
        Whenever or not to weight the gain according to the population distribution.

    num_pct : int, default=100
        Number of percentiles to evaluate the splits for each feature.

    max_features : {"auto", "sqrt", "log2" or None }, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider ``max_features`` features at each split.
        - If float, then ``max_features`` is a percentage and
          ``int(max_features * n_features)`` features are considered at each split.
        - If "auto", then ``max_features=sqrt(n_features)``.
        - If "sqrt", then ``max_features=sqrt(n_features)``.
        - If "log2", then ``max_features=log2(n_features)``.
        - If None, then ``max_features=n_features``.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider ``min_samples_split`` as the minimum number.
        - If float, then ``min_samples_split`` is a fraction and ``ceil(min_samples_split * n_samples)``
          are the minimum number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples
        in each of the left and right branches.
        This may have the effect of smoothing the model, especially in regression.

        - If int, then consider ``min_samples_leaf`` as the minimum number.
        - If float, then ``min_samples_leaf`` is a fraction and ``ceil(min_samples_leaf * n_samples)``
          are the minimum number of samples for each node.

    min_gain : float, default=0.001
        The minimum gain that a split must produce in order to be taken into account.

    pruned : bool, default=True
        Whenever to prune the decision tree using cost-based pruning.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always randomly permuted at each split. When
        `random_state` is None, the random number generator is the :class:`numpy:numpy.random.RandomState` instance
        used by :func:`numpy:numpy.random`.

    Attributes
    ----------
    tree_ : Tree object
        The underlying Tree object.

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           "Example-Dependent Cost-Sensitive Decision Trees. Expert Systems with Applications",
           Expert Systems with Applications, 42(19), 6609–6619, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042
    """

    _parameter_constraints: ClassVar[dict[str, list]] = {
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'criterion': [StrOptions({'direct_cost', 'pi_cost', 'gini_cost', 'entropy_cost'})],
        'criterion_weight': ['boolean'],
        'num_pct': [Interval(Real, left=0, right=100, closed='both')],
        'max_features': [
            StrOptions({'auto', 'sqrt', 'log2'}),
            Interval(Integral, 1, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='right'),
            None,
        ],
        'max_depth': [Interval(Integral, 1, None, closed='left'), None],
        'min_samples_split': [
            Interval(Integral, 2, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='right'),
        ],
        'min_samples_leaf': [
            Interval(Integral, 1, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='neither'),
        ],
        'min_gain': [Interval(Real, left=0, right=None, closed='neither')],
        'pruned': ['boolean'],
        'random_state': ['random_state'],
    }

    def __init__(
        self,
        *,
        tp_cost: ArrayLike | float = 0.0,
        tn_cost: ArrayLike | float = 0.0,
        fn_cost: ArrayLike | float = 0.0,
        fp_cost: ArrayLike | float = 0.0,
        criterion: Literal['direct_cost', 'pi_cost', 'gini_cost', 'entropy_cost'] = 'direct_cost',
        criterion_weight: bool = False,
        num_pct: int = 100,
        max_features: Literal['auto', 'sqrt', 'log2'] | int | float | None = None,
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_gain: float = 0.001,
        pruned: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.pruned = pruned
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def _node_cost(self, y_true, cost_mat):
        """
        Private function to calculate the cost of a node.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represent the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(cost_loss : float, node prediction : int, node predicted probability : float)

        """
        n_samples = len(y_true)

        # Evaluates the cost by predicting the node as positive and negative
        costs = np.zeros(2)
        fp_cost = cost_mat[:, 0]
        fn_cost = cost_mat[:, 1]
        tp_cost = cost_mat[:, 2]
        tn_cost = cost_mat[:, 3]
        costs[0] = cost_loss(
            y_true,
            np.zeros(y_true.shape),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            check_input=False,
        )
        costs[1] = cost_loss(
            y_true,
            np.ones(y_true.shape),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            check_input=False,
        )

        pi = np.array([1 - y_true.mean(), y_true.mean()])

        if self.criterion == 'pi_cost':
            costs *= pi
        elif self.criterion == 'gini_cost':
            costs *= pi**2
        elif self.criterion in 'entropy_cost':
            if pi[0] == 0 or pi[1] == 0:
                costs *= 0
            else:
                costs *= -np.log(pi)

        y_pred = np.argmin(costs)

        # Calculate the predicted probability of a node using laplace correction.
        n_positives = y_true.sum()
        y_prob = (n_positives + 1.0) / (n_samples + 2.0)

        return costs[y_pred], y_pred, y_prob

    def _calculate_gain(self, cost_base, y_true, X, cost_mat, split):
        """
        Private function to calculate the gain in cost of using split in the current node.

        Parameters
        ----------
        cost_base : float
            Cost of the naive prediction

        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represent the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        split : tuple(feature, split_value)
            The feature and value to split the node.

        Returns
        -------
        tuple(gain : float, left node prediction : int)

        """
        # TODO: This must be check in _best_split
        if cost_base == 0.0:  # no gain is possible
            # In case cost_b==0 and pi_1!=(0,1)
            return 0.0, int(np.sign(y_true.mean() - 0.5) == 1)

        feature, split_value = split
        filter_Xl = X[:, feature] <= split_value
        filter_Xr = ~filter_Xl
        n_samples, _ = X.shape

        # Check if one of the leafs is empty
        # TODO: This must be check in _best_split
        if np.nonzero(filter_Xl)[0].shape[0] in {0, n_samples}:  # One leaft is empty
            return 0.0, 0.0

        # Split X in Xl and Xr according to rule split
        Xl_cost, Xl_pred, _ = self._node_cost(y_true[filter_Xl], cost_mat[filter_Xl, :])
        Xr_cost, _, _ = self._node_cost(y_true[filter_Xr], cost_mat[filter_Xr, :])

        if self.criterion_weight:
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            Xl_w = n_samples_Xl * 1.0 / n_samples
            Xr_w = 1 - Xl_w
            gain = round((cost_base - (Xl_w * Xl_cost + Xr_w * Xr_cost)) / cost_base, 6)
        else:
            gain = round((cost_base - (Xl_cost + Xr_cost)) / cost_base, 6)

        return gain, Xl_pred

    def _best_split(self, y_true, X, cost_mat):
        """Private function to calculate the split that gives the best gain.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(split : tuple(j, l), gain : float, left node prediction : int,
              y_pred : int, y_prob : float)

        """
        _, n_features = X.shape
        num_pct = self.num_pct

        cost_base, y_pred, y_prob = self._node_cost(y_true, cost_mat)

        # Calculate the gain of all features each split in num_pct
        gains = np.zeros((n_features, num_pct))
        pred = np.zeros((n_features, num_pct))
        splits = np.zeros((n_features, num_pct))

        # Selected features
        selected_features = np.arange(0, self.n_features_)
        self._rng.shuffle(selected_features)
        selected_features = selected_features[: self.max_features_]
        selected_features.sort()

        # TODO:  # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.

        # For each feature test all possible splits
        for feature in selected_features:
            splits[feature, :] = np.percentile(X[:, feature], np.arange(0, 100, 100.0 / num_pct).tolist())

            for split_val in range(num_pct):
                # Avoid repeated values, since np.percentile may return repeated values
                if split_val == 0 or (split_val > 0 and splits[feature, split_val] != splits[feature, split_val - 1]):
                    split = (feature, splits[feature, split_val])
                    gains[feature, split_val], pred[feature, split_val] = self._calculate_gain(
                        cost_base, y_true, X, cost_mat, split
                    )

        best_split = np.unravel_index(gains.argmax(), gains.shape)

        return (
            (best_split[0], splits[best_split]),
            gains.max(),
            pred[best_split],
            y_pred,
            y_prob,
        )

    def _tree_grow(self, y_true, X, cost_mat, level=0):
        """Private recursive function to grow the decision tree.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        Tree : Object
            Container of the decision tree
            NOTE: it is not the same structure as the sklearn.tree.tree object

        """
        if len(X.shape) == 1:
            tree = {'y_pred': y_true, 'y_prob': 0.5, 'level': level, 'split': -1, 'n_samples': 1, 'gain': 0}
            return tree

        # Calculate the best split of the current node
        split, gain, _, y_pred, y_prob = self._best_split(y_true, X, cost_mat)

        n_samples, _ = X.shape

        # Construct the tree object as a dictionary

        # TODO: Convert tree to be equal to sklearn.tree.tree object
        tree = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'level': level,
            'split': -1,
            'n_samples': n_samples,
            'gain': gain,
        }

        # Check the stopping criteria
        if gain < self.min_gain:
            return tree
        if self.max_depth is not None and level >= self.max_depth:
            return tree
        if 0 < self.min_samples_split < 1:
            min_samples_split = ceil(self.min_samples_split * n_samples)
        else:
            min_samples_split = self.min_samples_split
        if n_samples <= min_samples_split:
            return tree

        feature, split_value = split
        filter_Xl = X[:, feature] <= split_value
        filter_Xr = ~filter_Xl
        n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
        n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

        if 0 < self.min_samples_leaf < 1:
            min_samples_leaf = ceil(self.min_samples_leaf * n_samples)
        else:
            min_samples_leaf = self.min_samples_leaf
        if min(n_samples_Xl, n_samples_Xr) <= min_samples_leaf:
            return tree

        # No stooping criteria is met
        tree['split'] = split
        tree['node'] = self.tree_.n_nodes
        self.tree_.n_nodes += 1

        tree['sl'] = self._tree_grow(y_true[filter_Xl], X[filter_Xl], cost_mat[filter_Xl], level + 1)
        tree['sr'] = self._tree_grow(y_true[filter_Xr], X[filter_Xr], cost_mat[filter_Xr], level + 1)

        return tree

    class _Tree:
        def __init__(self):
            self.n_nodes = 0
            self.tree = {}
            self.tree_pruned = {}
            self.nodes = []
            self.n_nodes_pruned = 0

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        tp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
    ) -> 'CSTreeClassifier':
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

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(f'Only binary classification is supported. The type of the target is {y_type}.')
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        y = np.where(y == self.classes_[1], 1, 0)
        tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            force_array=True,
            n_samples=len(y),
        )
        cost_mat = np.column_stack((fp_cost, fn_cost, tp_cost, tn_cost))
        self._rng = check_random_state(self.random_state)
        _, self.n_features_ = X.shape

        self.tree_ = self._Tree()

        # Maximum number of features to be taken into account per split
        if isinstance(self.max_features, str):
            if self.max_features in {'auto', 'sqrt'}:
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == 'log2':
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError('Invalid value for max_features. Allowed string values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, Integral | np.integer):
            max_features = self.max_features
        else:  # float
            max_features = max(1, int(self.max_features * self.n_features_)) if self.max_features > 0.0 else 1
        self.max_features_ = max_features

        self.tree_.tree = self._tree_grow(y, X, cost_mat)

        if self.pruned:
            self.pruning(X, y, cost_mat)

        return self

    def _nodes(self, tree):
        """Private function that find the number of nodes in a tree.

        Parameters
        ----------
        tree : object

        Returns
        -------
        nodes : array like of shape [n_nodes]
        """

        def recourse(temp_tree_, nodes):
            if isinstance(temp_tree_, dict) and temp_tree_['split'] != -1:
                nodes.append(temp_tree_['node'])
                if temp_tree_['split'] != -1:
                    for k in ['sl', 'sr']:
                        recourse(temp_tree_[k], nodes)
            return None

        nodes_ = []
        recourse(tree, nodes_)
        return nodes_

    def _classify(self, X, tree, proba=False):
        """Private function that classify a dataset using tree.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        tree : object

        proba : bool, optional (default=False)
            If True then return probabilities else return class

        Returns
        -------
        prediction : array of shape = [n_samples]
            If proba then return the predicted positive probabilities, else return
            the predicted class for each example in X
        """
        n_samples, _ = X.shape
        predicted = np.ones(n_samples)

        # Check if final node
        if tree['split'] == -1:
            predicted = predicted * tree['y_pred'] if not proba else predicted * tree['y_prob']
        else:
            feature, split_value = tree['split']
            filter_Xl = X[:, feature] <= split_value
            filter_Xr = ~filter_Xl
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

            if n_samples_Xl == 0:  # If left node is empty only continue with right
                predicted[filter_Xr] = self._classify(X[filter_Xr, :], tree['sr'], proba)
            elif n_samples_Xr == 0:  # If right node is empty only continue with left
                predicted[filter_Xl] = self._classify(X[filter_Xl, :], tree['sl'], proba)
            else:
                predicted[filter_Xl] = self._classify(X[filter_Xl, :], tree['sl'], proba)
                predicted[filter_Xr] = self._classify(X[filter_Xr, :], tree['sr'], proba)

        return predicted

    def predict(self, X):
        """Predict class of X.

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
        X = validate_data(self, X, reset=False)
        tree_ = self.tree_.tree_pruned if self.pruned else self.tree_.tree

        y_pred = self._classify(X, tree_, proba=False)
        # map to original classes
        y_pred = np.where(y_pred == 1, self.classes_[1], self.classes_[0])
        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

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
        X = validate_data(self, X, reset=False)
        n_samples, _ = X.shape
        prob = np.zeros((n_samples, 2))

        tree_ = self.tree_.tree_pruned if self.pruned else self.tree_.tree
        prob[:, 1] = self._classify(X, tree_, proba=True)
        prob[:, 0] = 1 - prob[:, 1]

        return prob

    def _delete_node(self, tree, node):
        """Private function that eliminate node from tree.

        Parameters
        ----------
        tree : object

        node : int
            node to be eliminated from tree

        Returns
        -------
        pruned_tree : object
        """
        # Calculate gains
        temp_tree = copy.deepcopy(tree)

        def recourse(temp_tree_, del_node):
            if isinstance(temp_tree_, dict) and temp_tree_['split'] != -1:
                if temp_tree_['node'] == del_node:
                    del temp_tree_['sr']
                    del temp_tree_['sl']
                    del temp_tree_['node']
                    temp_tree_['split'] = -1
                else:
                    for k in ['sl', 'sr']:
                        recourse(temp_tree_[k], del_node)
            return None

        recourse(temp_tree, node)
        return temp_tree

    def _pruning(self, X, y_true, cost_mat):
        """Private function that prune the decision tree.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        # Calculate gains
        nodes = self._nodes(self.tree_.tree_pruned)
        n_nodes = len(nodes)
        gains = np.zeros(n_nodes)

        y_pred = self._classify(X, self.tree_.tree_pruned)
        fp_cost = cost_mat[:, 0]
        fn_cost = cost_mat[:, 1]
        tp_cost = cost_mat[:, 2]
        tn_cost = cost_mat[:, 3]
        cost_base = cost_loss(
            y_true,
            y_pred,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
        )
        if cost_base == 0:
            cost_base = np.finfo(float).eps

        for m, node in enumerate(nodes):
            # Create temporal tree by eliminating node from tree_pruned
            temp_tree = self._delete_node(self.tree_.tree_pruned, node)
            y_pred = self._classify(X, temp_tree)

            nodes_pruned = self._nodes(temp_tree)

            # Calculate %gain
            gain = (
                cost_base
                - cost_loss(
                    y_true,
                    y_pred,
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    check_input=False,
                )
            ) / cost_base

            # Calculate %gain_size
            gain_size = (len(nodes) - len(nodes_pruned)) * 1.0 / len(nodes)

            # Calculate weighted gain
            gains[m] = gain * gain_size

        best_gain = np.max(gains)
        best_node = nodes[int(np.argmax(gains))]

        if best_gain > self.min_gain:
            self.tree_.tree_pruned = self._delete_node(self.tree_.tree_pruned, best_node)

            # If best tree is not root node, then recursively pruning the tree
            if best_node != 0:
                self._pruning(X, y_true, cost_mat)

    def pruning(self, X, y, cost_mat):
        """
        Prune the decision tree.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represent the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        self.tree_.tree_pruned = copy.deepcopy(self.tree_.tree)
        if self.tree_.n_nodes > 0:
            self._pruning(X, y, cost_mat)
            nodes_pruned = self._nodes(self.tree_.tree_pruned)
            self.tree_.n_nodes_pruned = len(nodes_pruned)
