"""Reference CSTree and CSForest: the CostCla implementation, as used in the benchmark.

Extracted from https://github.com/albahnsen/CostSensitiveClassification (CostCla 0.6, commit
92e40edb32c44fbcadcbe7839ee83dcb5e7e131e), BSD 3-clause License, Copyright (c) Alejandro Correa
Bahnsen. CostCla has not been updated since 2016 and no longer imports on current Python.

This file contains, verbatim, everything ``CostSensitiveDecisionTreeClassifier`` and
``CostSensitiveRandomForestClassifier`` execute:

- ``cost_loss`` and ``savings_score`` from ``costcla/metrics/costs.py``;
- ``CostSensitiveDecisionTreeClassifier`` from ``costcla/models/cost_tree.py``;
- the bagging machinery from ``costcla/models/bagging.py``;
- ``CostSensitiveRandomForestClassifier`` from ``costcla/models/cost_ensemble.py``.

Every edit is marked with a ``# CHANGE:`` comment. They are limited to what current Python, NumPy
and scikit-learn require:

- ``six`` (``string_types``, ``with_metaclass``, ``moves.zip``) replaced by its Python 3 equivalent;
- ``inspect.getargspec`` (removed in Python 3.11, and never used) no longer imported;
- ``np.int`` / ``np.float`` (removed in NumPy 1.24) replaced by ``int`` / ``float``;
- ``BaseEnsemble(base_estimator=...)`` renamed to ``BaseEnsemble(estimator=...)`` (scikit-learn 1.4);
- the stacking and Bayes-minimum-risk combination methods of the ensemble need CostCla's
  ``CostSensitiveLogisticRegression`` (which depends on the unmaintained ``pyea`` package) and
  ``BayesMinimumRiskClassifier``. Those are not extracted; they are replaced by stubs that raise if
  one of those combination methods is requested. The default ``'majority_voting'`` combination,
  used in the benchmark, does not touch them.

The algorithms themselves are untouched. Cost matrix layout (``cost_mat`` of shape
``(n_samples, 4)``): columns are the costs of false positives, false negatives, true positives and
true negatives, in that order.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import copy
import itertools
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, column_or_1d
from sklearn.utils.random import sample_without_replacement

# CHANGE: the original imports these two classes for the stacking and Bayes-minimum-risk combination
# methods only. They are not part of this extraction.


def CostSensitiveLogisticRegression(*args, **kwargs):
    raise NotImplementedError('The stacking combination methods are not included in this extraction.')


def BayesMinimumRiskClassifier(*args, **kwargs):
    raise NotImplementedError('The Bayes-minimum-risk combination methods are not included in this extraction.')


# ================================================================================================
# costcla/metrics/costs.py
# ================================================================================================


def cost_loss(y_true, y_pred, cost_mat):
    # TODO: update description
    """Cost classification loss.

    This function calculates the cost of using y_pred on y_true with
    cost-matrix cost-mat. It differ from traditional classification evaluation
    measures since measures such as accuracy asing the same cost to different
    errors, but that is not the real case in several real-world classification
    problems as they are example-dependent cost-sensitive in nature, where the
    costs due to misclassification vary between examples.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    Returns
    -------
    loss : float
        Cost of a using y_pred on y_true with cost-matrix cost-mat

    References
    ----------
    .. [1] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    .. [2] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    See Also
    --------
    savings_score

    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> cost_loss(y_true, y_pred, cost_mat)
    3
    """
    # TODO: Check consistency of cost_mat

    y_true = column_or_1d(y_true)
    y_true = (y_true == 1).astype(float)  # CHANGE: np.float was removed in NumPy 1.24
    y_pred = column_or_1d(y_pred)
    y_pred = (y_pred == 1).astype(float)  # CHANGE: np.float was removed in NumPy 1.24
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2])
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.sum(cost)


def savings_score(y_true, y_pred, cost_mat):
    # TODO: update description
    """Savings score.

    This function calculates the savings cost of using y_pred on y_true with
    cost-matrix cost-mat, as the difference of y_pred and the cost_loss of a naive
    classification model.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    Returns
    -------
    score : float
        Savings of a using y_pred on y_true with cost-matrix cost-mat

        The best performance is 1.

    References
    ----------
    .. [1] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    See Also
    --------
    cost_loss

    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import savings_score, cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> savings_score(y_true, y_pred, cost_mat)
    0.5
    """
    # TODO: Check consistency of cost_mat
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    n_samples = len(y_true)

    # Calculate the cost of naive prediction
    cost_base = min(cost_loss(y_true, np.zeros(n_samples), cost_mat), cost_loss(y_true, np.ones(n_samples), cost_mat))

    cost = cost_loss(y_true, y_pred, cost_mat)
    return 1.0 - cost / cost_base


# ================================================================================================
# costcla/models/cost_tree.py
# ================================================================================================


class CostSensitiveDecisionTreeClassifier(BaseEstimator):
    """A example-dependent cost-sensitive binary decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="direct_cost")
        The function to measure the quality of a split. Supported criteria are
        "direct_cost" for the Direct Cost impurity measure, "pi_cost", "gini_cost",
        and "entropy_cost".

    criterion_weight : bool, optional (default=False)
        Whenever or not to weight the gain according to the population distribution.

    num_pct : int, optional (default=100)
        Number of percentiles to evaluate the splits for each feature.


    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_samples_leaf`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_gain : float, optional (default=0.001)
        The minimum gain that a split must produce in order to be taken into account.

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Example-Dependent Cost-Sensitive Decision Trees. Expert Systems with Applications" <http://albahnsen.com/files/Example-Dependent%20Cost-Sensitive%20Decision%20Trees.pdf>`__,
           Expert Systems with Applications, 42(19), 6609–6619, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveDecisionTreeClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(
    ...     data.data, data.target, data.cost_mat, test_size=0.33, random_state=0
    ... )
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = (
    ...     RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    ... )
    >>> f = CostSensitiveDecisionTreeClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CSDecisionTree
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.481916135529
    """

    def __init__(
        self,
        criterion='direct_cost',
        criterion_weight=False,
        num_pct=100,
        max_features=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_gain=0.001,
        pruned=True,
    ):

        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.pruned = pruned

        self.n_features_ = None
        self.max_features_ = None

        self.tree_ = []

    def set_param(self, attribute, value):
        setattr(self, attribute, value)

    def _node_cost(self, y_true, cost_mat):
        """Private function to calculate the cost of a node.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(cost_loss : float, node prediction : int, node predicted probability : float)

        """
        n_samples = len(y_true)

        # Evaluates the cost by predicting the node as positive and negative
        costs = np.zeros(2)
        costs[0] = cost_loss(y_true, np.zeros(y_true.shape), cost_mat)
        costs[1] = cost_loss(y_true, np.ones(y_true.shape), cost_mat)

        pi = np.array([1 - y_true.mean(), y_true.mean()])

        if self.criterion == 'direct_cost':
            costs = costs
        elif self.criterion == 'pi_cost':
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
        """Private function to calculate the gain in cost of using split in the
         current node.

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
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        split : tuple of len = 2
            split[0] = feature to split = j
            split[1] = where to split = l

        Returns
        -------
        tuple(gain : float, left node prediction : int)

        """
        # Check if cost_base == 0, then no gain is possible
        # TODO: This must be check in _best_split
        if cost_base == 0.0:
            # In case cost_b==0 and pi_1!=(0,1)
            return 0.0, int(np.sign(y_true.mean() - 0.5) == 1)

        j, l = split
        filter_Xl = X[:, j] <= l
        filter_Xr = ~filter_Xl
        n_samples, n_features = X.shape

        # Check if one of the leafs is empty
        # TODO: This must be check in _best_split
        if np.nonzero(filter_Xl)[0].shape[0] in [0, n_samples]:  # One leaft is empty
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
        n_samples, n_features = X.shape
        num_pct = self.num_pct

        cost_base, y_pred, y_prob = self._node_cost(y_true, cost_mat)

        # Calculate the gain of all features each split in num_pct
        gains = np.zeros((n_features, num_pct))
        pred = np.zeros((n_features, num_pct))
        splits = np.zeros((n_features, num_pct))

        # Selected features
        selected_features = np.arange(0, self.n_features_)
        # Add random state
        np.random.shuffle(selected_features)
        selected_features = selected_features[: self.max_features_]
        selected_features.sort()

        # TODO:  # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.

        # For each feature test all possible splits
        for j in selected_features:
            splits[j, :] = np.percentile(X[:, j], np.arange(0, 100, 100.0 / num_pct).tolist())

            for l in range(num_pct):
                # Avoid repeated values, since np.percentile may return repeated values
                if l == 0 or (l > 0 and splits[j, l] != splits[j, l - 1]):
                    split = (j, splits[j, l])
                    gains[j, l], pred[j, l] = self._calculate_gain(cost_base, y_true, X, cost_mat, split)

        best_split = np.unravel_index(gains.argmax(), gains.shape)

        return (best_split[0], splits[best_split]), gains.max(), pred[best_split], y_pred, y_prob

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
        # TODO: Find error, add min_samples_split
        if len(X.shape) == 1:
            tree = dict(y_pred=y_true, y_prob=0.5, level=level, split=-1, n_samples=1, gain=0)
            return tree

        # Calculate the best split of the current node
        split, gain, Xl_pred, y_pred, y_prob = self._best_split(y_true, X, cost_mat)

        n_samples, n_features = X.shape

        # Construct the tree object as a dictionary

        # TODO: Convert tree to be equal to sklearn.tree.tree object
        tree = dict(y_pred=y_pred, y_prob=y_prob, level=level, split=-1, n_samples=n_samples, gain=gain)

        # Check the stopping criteria
        if gain < self.min_gain:
            return tree
        if self.max_depth is not None and level >= self.max_depth:
            return tree
        if n_samples <= self.min_samples_split:
            return tree

        j, l = split
        filter_Xl = X[:, j] <= l
        filter_Xr = ~filter_Xl
        n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
        n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

        if min(n_samples_Xl, n_samples_Xr) <= self.min_samples_leaf:
            return tree

        # No stooping criteria is met
        tree['split'] = split
        tree['node'] = self.tree_.n_nodes
        self.tree_.n_nodes += 1

        tree['sl'] = self._tree_grow(y_true[filter_Xl], X[filter_Xl], cost_mat[filter_Xl], level + 1)
        tree['sr'] = self._tree_grow(y_true[filter_Xr], X[filter_Xr], cost_mat[filter_Xr], level + 1)

        return tree

    class _tree_class:
        def __init__(self):
            self.n_nodes = 0
            self.tree = dict()
            self.tree_pruned = dict()
            self.nodes = []
            self.n_nodes_pruned = 0

    def fit(self, X, y, cost_mat, check_input=False):
        """Build a example-dependent cost-sensitive decision tree from the training set (X, y, cost_mat)

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

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.


        Returns
        -------
        self : object
            Returns self.
        """
        # TODO: Check input
        # TODO: Add random state
        n_samples, self.n_features_ = X.shape

        self.tree_ = self._tree_class()

        # Maximum number of features to be taken into account per split
        # CHANGE: six.string_types -> str (six is a Python 2 compatibility layer)
        if isinstance(self.max_features, str):
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == 'log2':
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError('Invalid value for max_features. Allowed string values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        elif self.max_features > 0.0:
            max_features = max(1, int(self.max_features * self.n_features_))
        else:
            max_features = 1  # On sklearn is 0.
        self.max_features_ = max_features

        self.tree_.tree = self._tree_grow(y, X, cost_mat)

        if self.pruned:
            self.pruning(X, y, cost_mat)

        self.classes_ = np.array([0, 1])

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
        n_samples, n_features = X.shape
        predicted = np.ones(n_samples)

        # Check if final node
        if tree['split'] == -1:
            if not proba:
                predicted = predicted * tree['y_pred']
            else:
                predicted = predicted * tree['y_prob']
        else:
            j, l = tree['split']
            filter_Xl = X[:, j] <= l
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
        # TODO: Check consistency of X
        if self.pruned:
            tree_ = self.tree_.tree_pruned
        else:
            tree_ = self.tree_.tree

        return self._classify(X, tree_, proba=False)

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
        # TODO: Check consistency of X
        n_samples, n_features = X.shape
        prob = np.zeros((n_samples, 2))

        if self.pruned:
            tree_ = self.tree_.tree_pruned
        else:
            tree_ = self.tree_.tree

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
        cost_base = cost_loss(y_true, y_pred, cost_mat)

        for m, node in enumerate(nodes):
            # Create temporal tree by eliminating node from tree_pruned
            temp_tree = self._delete_node(self.tree_.tree_pruned, node)
            y_pred = self._classify(X, temp_tree)

            nodes_pruned = self._nodes(temp_tree)

            # Calculate %gain
            gain = (cost_base - cost_loss(y_true, y_pred, cost_mat)) / cost_base

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
        """Function that prune the decision tree.

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
        self.tree_.tree_pruned = copy.deepcopy(self.tree_.tree)
        if self.tree_.n_nodes > 0:
            self._pruning(X, y, cost_mat)
            nodes_pruned = self._nodes(self.tree_.tree_pruned)
            self.tree_.n_nodes_pruned = len(nodes_pruned)


# ================================================================================================
# costcla/models/bagging.py
# ================================================================================================

MAX_INT = np.iinfo(np.int32).max


# Is different in 0.15, copy version from 0.16-git
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    if n_jobs == -1:
        n_jobs = min(cpu_count(), n_estimators)

    else:
        n_jobs = min(n_jobs, n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=int)  # CHANGE: np.int removed in NumPy 1.24
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _parallel_build_estimators(n_estimators, ensemble, X, y, cost_mat, seeds, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features

    if not isinstance(max_samples, (numbers.Integral, np.integer)) and (0.0 < max_samples <= 1.0):
        max_samples = int(max_samples * n_samples)

    if not isinstance(max_features, (numbers.Integral, np.integer)) and (0.0 < max_features <= 1.0):
        max_features = int(max_features * n_features)

    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features

    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print('building estimator %d of %d' % (i + 1, n_estimators))

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=seed)
        except ValueError:
            pass

        # Draw features
        if bootstrap_features:
            features = random_state.randint(0, n_features, max_features)
        else:
            features = sample_without_replacement(n_features, max_features, random_state=random_state)

        # Draw samples, using a mask, and then fit
        if bootstrap:
            indices = random_state.randint(0, n_samples, max_samples)
        else:
            indices = sample_without_replacement(n_samples, max_samples, random_state=random_state)

        sample_counts = np.bincount(indices, minlength=n_samples)

        estimator.fit((X[indices])[:, features], y[indices], cost_mat[indices, :])
        samples = sample_counts > 0.0

        estimators.append(estimator)
        estimators_samples.append(samples)
        estimators_features.append(features)

    return estimators, estimators_samples, estimators_features


def _parallel_predict_proba(estimators, estimators_features, X, n_classes, combination, estimators_weight):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features, weight in zip(estimators, estimators_features, estimators_weight):
        proba_estimator = estimator.predict_proba(X[:, features])
        if combination in ['weighted_voting', 'weighted_bmr']:
            proba += proba_estimator * weight
        else:
            proba += proba_estimator

    return proba


def _parallel_predict(estimators, estimators_features, X, n_classes, combination, estimators_weight):
    """Private function used to compute predictions within a job."""
    n_samples = X.shape[0]
    pred = np.zeros((n_samples, n_classes))
    n_estimators = len(estimators)

    for estimator, features, weight in zip(estimators, estimators_features, estimators_weight):
        # Resort to voting
        predictions = estimator.predict(X[:, features])

        for i in range(n_samples):
            if combination == 'weighted_voting':
                pred[i, int(predictions[i])] += 1 * weight
            else:
                pred[i, int(predictions[i])] += 1

    return pred


# TODO: Create stacking set in parallel


def _create_stacking_set(estimators, estimators_features, estimators_weight, X, combination):
    """Private function used to create the stacking training set."""
    n_samples = X.shape[0]

    valid_estimators = np.nonzero(estimators_weight)[0]
    n_valid_estimators = valid_estimators.shape[0]
    X_stacking = np.zeros((n_samples, n_valid_estimators))

    for e in range(n_valid_estimators):
        if combination in ['stacking', 'stacking_bmr']:
            X_stacking[:, e] = estimators[valid_estimators[e]].predict(X[:, estimators_features[valid_estimators[e]]])
        elif combination in ['stacking_proba', 'stacking_proba_bmr']:
            X_stacking[:, e] = estimators[valid_estimators[e]].predict_proba(
                X[:, estimators_features[valid_estimators[e]]]
            )[:, 1]

    return X_stacking


# CHANGE: six.with_metaclass(ABCMeta, BaseEnsemble) -> Python 3 metaclass syntax
class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        combination='majority_voting',
        n_jobs=1,
        random_state=None,
        verbose=0,
    ):
        # CHANGE: scikit-learn 1.4 renamed BaseEnsemble's `base_estimator` argument to `estimator`
        super().__init__(estimator=base_estimator, n_estimators=n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.combination = combination
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, cost_mat, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Convert data
        # X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])  # Not in sklearn verion 0.15

        # Remap output
        n_samples, self.n_features_ = X.shape
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError('max_samples must be in (0, n_samples]')

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError('max_features must be in (0, n_features]')

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators, self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i], self, X, y, cost_mat, seeds[starts[i] : starts[i + 1]], verbose=self.verbose
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(t[2] for t in all_results))

        self._evaluate_oob_savings(X, y, cost_mat)

        if self.combination in ['stacking', 'stacking_proba', 'stacking_bmr', 'stacking_proba_bmr']:
            self._fit_stacking_model(X, y, cost_mat)

        if self.combination in ['majority_bmr', 'weighted_bmr', 'stacking_bmr', 'stacking_proba_bmr']:
            self._fit_bmr_model(X, y)

        return self

    def _fit_bmr_model(self, X, y):
        """Private function used to fit the BayesMinimumRisk model."""
        self.f_bmr = BayesMinimumRiskClassifier()
        X_bmr = self.predict_proba(X)
        self.f_bmr.fit(y, X_bmr)
        return self

    def _fit_stacking_model(self, X, y, cost_mat, max_iter=100):
        """Private function used to fit the stacking model."""
        self.f_staking = CostSensitiveLogisticRegression(verbose=self.verbose, max_iter=max_iter)
        X_stacking = _create_stacking_set(
            self.estimators_, self.estimators_features_, self.estimators_weight_, X, self.combination
        )
        self.f_staking.fit(X_stacking, y, cost_mat)
        return self

    # TODO: _evaluate_oob_savings in parallel
    def _evaluate_oob_savings(self, X, y, cost_mat):
        """Private function used to calculate the OOB Savings of each estimator."""
        estimators_weight = []
        for estimator, samples, features in zip(self.estimators_, self.estimators_samples_, self.estimators_features_):
            # Test if all examples where used for training
            if not np.any(~samples):
                # Then use training
                oob_pred = estimator.predict(X[:, features])
                oob_savings = max(0, savings_score(y, oob_pred, cost_mat))
            else:
                # Then use OOB
                oob_pred = estimator.predict((X[~samples])[:, features])
                oob_savings = max(0, savings_score(y[~samples], oob_pred, cost_mat[~samples]))

            estimators_weight.append(oob_savings)

        # Control in case were all weights are 0
        if sum(estimators_weight) == 0:
            self.estimators_weight_ = np.ones(len(estimators_weight)) / len(estimators_weight)
        else:
            self.estimators_weight_ = (np.array(estimators_weight) / sum(estimators_weight)).tolist()

        return self

    def _validate_y(self, y):
        # Default implementation
        return column_or_1d(y, warn=True)


class BaggingClassifier(BaseBagging, ClassifierMixin):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
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
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    `classes_`: array of shape = [n_classes]
        The classes labels.

    `n_classes_`: int or list
        The number of classes.

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
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        combination='voting',
        n_jobs=1,
        random_state=None,
        verbose=0,
    ):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            combination=combination,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeClassifier())

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X, cost_mat=None):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        cost_mat : optional array-like of shape = [n_samples, 4], (default=None)
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        pred : array of shape = [n_samples]
            The predicted classes.
        """
        # Check data
        # X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])  # Dont in version 0.15

        if self.n_features_ != X.shape[1]:
            raise ValueError(
                'Number of features of the model must '
                f'match the input. Model n_features is {self.n_features_} and '
                f'input n_features is {X.shape[1]}.'
            )

        # TODO: check if combination in possible combinations

        if self.combination in ['stacking', 'stacking_proba']:
            X_stacking = _create_stacking_set(
                self.estimators_, self.estimators_features_, self.estimators_weight_, X, self.combination
            )
            return self.f_staking.predict(X_stacking)

        elif self.combination in ['majority_voting', 'weighted_voting']:
            # Parallel loop
            n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators, self.n_jobs)

            all_pred = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict)(
                    self.estimators_[starts[i] : starts[i + 1]],
                    self.estimators_features_[starts[i] : starts[i + 1]],
                    X,
                    self.n_classes_,
                    self.combination,
                    self.estimators_weight_[starts[i] : starts[i + 1]],
                )
                for i in range(n_jobs)
            )

            # Reduce
            pred = sum(all_pred) / self.n_estimators

            return self.classes_.take(np.argmax(pred, axis=1), axis=0)

        elif self.combination in ['majority_bmr', 'weighted_bmr', 'stacking_bmr', 'stacking_proba_bmr']:
            # TODO: Add check if cost_mat == None
            X_bmr = self.predict_proba(X)
            return self.f_bmr.predict(X_bmr, cost_mat)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of a an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Check data
        # X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])  # Dont in version 0.15

        if self.n_features_ != X.shape[1]:
            raise ValueError(
                'Number of features of the model must '
                f'match the input. Model n_features is {self.n_features_} and '
                f'input n_features is {X.shape[1]}.'
            )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
                self.n_classes_,
                self.combination,
                self.estimators_weight_[starts[i] : starts[i + 1]],
            )
            for i in range(n_jobs)
        )

        # Reduce
        if self.combination in ['majority_voting', 'majority_bmr']:
            proba = sum(all_proba) / self.n_estimators
        elif self.combination in ['weighted_voting', 'weighted_bmr']:
            proba = sum(all_proba)
        elif self.combination in ['stacking', 'stacking_proba', 'stacking_bmr', 'stacking_proba_bmr']:
            X_stacking = _create_stacking_set(
                self.estimators_, self.estimators_features_, self.estimators_weight_, X, self.combination
            )
            proba = self.f_staking.predict_proba(X_stacking)

        return proba


# ================================================================================================
# costcla/models/cost_ensemble.py
# ================================================================================================


class CostSensitiveRandomForestClassifier(BaggingClassifier):
    """A example-dependent cost-sensitive random forest  classifier.

    Parameters
    ----------
    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split in each tree:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    See Also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveRandomForestClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(
    ...     data.data, data.target, data.cost_mat, test_size=0.33, random_state=0
    ... )
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = (
    ...     RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    ... )
    >>> f = CostSensitiveRandomForestClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CostSensitiveRandomForestClassifier
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.499390945808
    """

    def __init__(
        self, n_estimators=10, combination='majority_voting', max_features='auto', n_jobs=1, verbose=False, pruned=False
    ):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(max_features=max_features, pruned=pruned),
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination=combination,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose,
        )
        self.pruned = pruned
