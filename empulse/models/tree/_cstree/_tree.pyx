# distutils: language = c++
"""
The array-based cost-sensitive decision tree, the builders that grow it, and cost-complexity pruning.

The builders and the pruning are ported from scikit-learn's ``DepthFirstTreeBuilder``,
``BestFirstTreeBuilder``, ``_cost_complexity_prune`` and ``_build_pruned_tree``
(sklearn/tree/_tree.pyx), Copyright (c) the scikit-learn developers, BSD-3-Clause license. Unlike
scikit-learn's ``Tree``, a ``CostTree`` is binary and single-output, keeps the fields that traversal
reads apart from the rest, and records at build time which class each node's training samples cost
least to predict, so leaves need no second pass over the training data to make their decisions.
"""

from libc.math cimport INFINITY, isnan
from libc.stdlib cimport free, realloc
from libcpp.algorithm cimport pop_heap, push_heap
from libcpp.stack cimport stack
from libcpp.vector cimport vector

import numpy as np
from scipy.sparse import csr_matrix

from ._splitter cimport Splitter, SplitRecord
from ._utils cimport float32_t, int32_t

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = -1
cdef intp_t _TREE_UNDEFINED = -2

cdef enum:
    LEAF = -1  # the child of a leaf, as a compile-time constant for the traversal loops

cdef float64_t EPSILON = np.finfo('double').eps

cdef Node _dummy_node
NODE_DTYPE = np.asarray(<Node[:1]>(&_dummy_node)).dtype


cdef class CostTree:
    """
    A fitted binary cost-sensitive decision tree, stored as parallel arrays indexed by node.

    Node 0 is the root. A leaf has ``children_left == children_right == -1``. The attribute names
    follow :class:`sklearn.tree._tree.Tree`, so scikit-learn's tree plotting and export functions
    accept it, with ``value[i, 0]`` holding the weight fraction of each class in node ``i``.
    ``positive[i]`` says whether predicting node ``i`` positive costs least on its training samples.
    """

    def __cinit__(self, intp_t n_features):
        self.n_features = n_features
        self.node_count = 0
        self.max_depth = 0
        self.capacity = 0
        self.nodes = NULL
        self.impurity_ = NULL
        self.n_node_samples_ = NULL
        self.weighted_n_node_samples_ = NULL
        self.value_ = NULL
        self.positive_ = NULL

    def __dealloc__(self):
        free(self.nodes)
        free(self.impurity_)
        free(self.n_node_samples_)
        free(self.weighted_n_node_samples_)
        free(self.value_)
        free(self.positive_)

    # --------------------------------------------------------------------------------------------
    # Storage
    # --------------------------------------------------------------------------------------------

    cdef int _resize(self, intp_t capacity) except -1 nogil:
        """Make room for ``capacity`` nodes (at least ``node_count``, and at least one)."""
        if capacity < self.node_count:
            capacity = self.node_count
        if capacity < 1:
            capacity = 1
        _realloc(<void**> &self.nodes, capacity, sizeof(Node))
        _realloc(<void**> &self.impurity_, capacity, sizeof(float64_t))
        _realloc(<void**> &self.n_node_samples_, capacity, sizeof(intp_t))
        _realloc(<void**> &self.weighted_n_node_samples_, capacity, sizeof(float64_t))
        _realloc(<void**> &self.value_, 2 * capacity, sizeof(float64_t))
        _realloc(<void**> &self.positive_, capacity, sizeof(uint8_t))
        self.capacity = capacity
        return 0

    cdef intp_t _add_node(
        self,
        intp_t parent,
        bint is_left,
        bint is_leaf,
        intp_t feature,
        float64_t threshold,
        bint missing_go_to_left,
        float64_t impurity,
        intp_t n_node_samples,
        float64_t weighted_n_node_samples,
        const ClassSums* sums,
    ) except -1 nogil:
        """Add a node, register it with its parent, and return its id."""
        cdef float64_t pos_cost = sums.cp[0] + sums.cp[1]
        cdef float64_t neg_cost = sums.cn[0] + sums.cn[1]
        # The cheaper decision on the node's training samples; a tie goes to the more frequent
        # class, and a tie in that too to negative.
        cdef bint positive = pos_cost < neg_cost or (pos_cost == neg_cost and sums.cw[1] > sums.cw[0])
        return _append_node(
            self,
            parent,
            is_left,
            is_leaf,
            feature,
            threshold,
            missing_go_to_left,
            impurity,
            n_node_samples,
            weighted_n_node_samples,
            sums.cw[0] / weighted_n_node_samples,
            sums.cw[1] / weighted_n_node_samples,
            positive,
        )

    # --------------------------------------------------------------------------------------------
    # Inspection, with the attribute names of scikit-learn's Tree
    # --------------------------------------------------------------------------------------------

    @property
    def n_outputs(self):
        return 1

    @property
    def n_classes(self):
        return np.array([2], dtype=np.intp)

    @property
    def max_n_classes(self):
        return 2

    @property
    def n_leaves(self):
        return int(np.count_nonzero(self.children_left == TREE_LEAF))

    @property
    def nodes_array(self):
        """The packed traversal fields of every node, as a structured array (a copy)."""
        if self.node_count == 0:
            return np.empty(0, dtype=NODE_DTYPE)
        return np.asarray(<Node[:self.node_count]> self.nodes).copy()

    @property
    def children_left(self):
        return self.nodes_array['left_child'].astype(np.intp)

    @property
    def children_right(self):
        return self.nodes_array['right_child'].astype(np.intp)

    @property
    def feature(self):
        return self.nodes_array['feature'].astype(np.intp)

    @property
    def threshold(self):
        return self.nodes_array['threshold']

    @property
    def missing_go_to_left(self):
        """Whether a sample missing the node's feature goes to the left child, of shape ``(node_count,)``."""
        return self.nodes_array['missing_go_to_left'].astype(bool)

    @property
    def impurity(self):
        return _copy_float64(self.impurity_, self.node_count)

    @property
    def n_node_samples(self):
        if self.node_count == 0:
            return np.empty(0, dtype=np.intp)
        return np.asarray(<intp_t[:self.node_count]> self.n_node_samples_).copy()

    @property
    def weighted_n_node_samples(self):
        return _copy_float64(self.weighted_n_node_samples_, self.node_count)

    @property
    def value(self):
        """The weight fraction of each class per node, of shape ``(node_count, 1, 2)``."""
        return _copy_float64(self.value_, 2 * self.node_count).reshape(self.node_count, 1, 2)

    @property
    def positive(self):
        """Whether each node decides positive, of shape ``(node_count,)``."""
        if self.node_count == 0:
            return np.empty(0, dtype=bool)
        return np.asarray(<uint8_t[:self.node_count]> self.positive_).astype(bool)

    # --------------------------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------------------------

    def apply(self, X):
        """Return the index of the leaf each row of ``X`` ends up in."""
        cdef const float32_t[:, ::1] X_view = _as_rows(X)
        cdef intp_t n_samples = X_view.shape[0]
        cdef intp_t[::1] out = np.empty(n_samples, dtype=np.intp)
        cdef intp_t i
        with nogil:
            for i in range(n_samples):
                out[i] = _find_leaf(self.nodes, &X_view[i, 0])
        return np.asarray(out)

    def predict_proba(self, X):
        """Return the class fractions of the leaf each row of ``X`` ends up in."""
        cdef const float32_t[:, ::1] X_view = _as_rows(X)
        cdef intp_t n_samples = X_view.shape[0]
        cdef float64_t[:, ::1] out = np.empty((n_samples, 2), dtype=np.float64)
        cdef intp_t i, leaf
        with nogil:
            for i in range(n_samples):
                leaf = _find_leaf(self.nodes, &X_view[i, 0])
                out[i, 0] = self.value_[2 * leaf]
                out[i, 1] = self.value_[2 * leaf + 1]
        return np.asarray(out)

    def predict_positive(self, X):
        """Return, for each row of ``X``, whether its leaf decides positive."""
        cdef const float32_t[:, ::1] X_view = _as_rows(X)
        cdef intp_t n_samples = X_view.shape[0]
        cdef uint8_t[::1] out = np.empty(n_samples, dtype=np.uint8)
        cdef intp_t i
        with nogil:
            for i in range(n_samples):
                out[i] = self.positive_[_find_leaf(self.nodes, &X_view[i, 0])]
        return np.asarray(out).view(bool)

    def accumulate(
        self,
        const float32_t[:, ::1] X,
        float64_t weight,
        float64_t[:, ::1] proba,
        float64_t[::1] votes,
    ):
        """
        Add ``weight`` times this tree's class fractions to ``proba``, and its decisions to ``votes``.

        ``X`` must be C-ordered float32, as forests convert it once for all their trees.
        """
        cdef intp_t n_samples = X.shape[0]
        cdef intp_t i, leaf
        with nogil:
            for i in range(n_samples):
                leaf = _find_leaf(self.nodes, &X[i, 0])
                proba[i, 0] += weight * self.value_[2 * leaf]
                proba[i, 1] += weight * self.value_[2 * leaf + 1]
                votes[i] += weight * self.positive_[leaf]

    def decision_path(self, X):
        """Return a CSR indicator matrix of the nodes each row of ``X`` passes through."""
        cdef const float32_t[:, ::1] X_view = _as_rows(X)
        cdef intp_t n_samples = X_view.shape[0]
        cdef intp_t[::1] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef intp_t[::1] indices = np.zeros(n_samples * (1 + self.max_depth), dtype=np.intp)
        cdef intp_t i, node_id
        cdef Node* node
        cdef float32_t value
        with nogil:
            for i in range(n_samples):
                indptr[i + 1] = indptr[i]
                node_id = 0
                node = self.nodes
                while node.left_child != _TREE_LEAF:
                    indices[indptr[i + 1]] = node_id
                    indptr[i + 1] += 1
                    value = X_view[i, node.feature]
                    if isnan(value):
                        node_id = node.left_child if node.missing_go_to_left else node.right_child
                    elif value <= node.threshold:
                        node_id = node.left_child
                    else:
                        node_id = node.right_child
                    node = &self.nodes[node_id]
                indices[indptr[i + 1]] = node_id
                indptr[i + 1] += 1
        indices_array = np.asarray(indices)[:indptr[n_samples]]
        data = np.ones(indices_array.shape[0], dtype=np.intp)
        return csr_matrix((data, indices_array, np.asarray(indptr)), shape=(n_samples, self.node_count))

    def compute_feature_importances(self, normalize=True):
        """The total weighted impurity decrease each feature brings, normalized to sum to 1."""
        importances = np.zeros(self.n_features, dtype=np.float64)
        if self.node_count == 0:
            return importances
        left = self.children_left
        right = self.children_right
        weighted_impurity = self.weighted_n_node_samples * self.impurity
        internal = np.flatnonzero(left != TREE_LEAF)
        decrease = (
            weighted_impurity[internal] - weighted_impurity[left[internal]] - weighted_impurity[right[internal]]
        )
        np.add.at(importances, self.feature[internal], decrease)
        importances /= self.weighted_n_node_samples_[0]
        if normalize:
            normalizer = importances.sum()
            if normalizer > 0.0:
                importances /= normalizer
        return importances

    # --------------------------------------------------------------------------------------------
    # Pickling
    # --------------------------------------------------------------------------------------------

    def __reduce__(self):
        return (CostTree, (self.n_features,), self.__getstate__())

    def __getstate__(self):
        return {
            'max_depth': self.max_depth,
            'node_count': self.node_count,
            'nodes': self.nodes_array,
            'impurity': self.impurity,
            'n_node_samples': self.n_node_samples,
            'weighted_n_node_samples': self.weighted_n_node_samples,
            'value': self.value.reshape(-1, 2),
            'positive': self.positive.astype(np.uint8),
        }

    def __setstate__(self, state):
        cdef intp_t node_count = state['node_count']
        cdef const Node[::1] nodes = np.ascontiguousarray(state['nodes'], dtype=NODE_DTYPE)
        cdef const float64_t[::1] impurity = np.ascontiguousarray(state['impurity'], dtype=np.float64)
        cdef const intp_t[::1] n_node_samples = np.ascontiguousarray(state['n_node_samples'], dtype=np.intp)
        cdef const float64_t[::1] weighted = np.ascontiguousarray(state['weighted_n_node_samples'], dtype=np.float64)
        cdef const float64_t[::1] value = np.ascontiguousarray(state['value'], dtype=np.float64).reshape(-1)
        cdef const uint8_t[::1] positive = np.ascontiguousarray(state['positive'], dtype=np.uint8)
        cdef intp_t i
        if not (nodes.shape[0] == impurity.shape[0] == n_node_samples.shape[0] == weighted.shape[0]
                == positive.shape[0] == node_count and value.shape[0] == 2 * node_count):
            raise ValueError('Inconsistent CostTree state.')
        self.node_count = 0
        self._resize(node_count)
        for i in range(node_count):
            self.nodes[i] = nodes[i]
            self.impurity_[i] = impurity[i]
            self.n_node_samples_[i] = n_node_samples[i]
            self.weighted_n_node_samples_[i] = weighted[i]
            self.value_[2 * i] = value[2 * i]
            self.value_[2 * i + 1] = value[2 * i + 1]
            self.positive_[i] = positive[i]
        self.node_count = node_count
        self.max_depth = state['max_depth']


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

cdef int _realloc(void** p, intp_t n_elements, size_t element_size) except -1 nogil:
    cdef void* tmp = realloc(p[0], n_elements * element_size)
    if tmp == NULL:
        raise MemoryError(f'could not allocate {n_elements * element_size} bytes')
    p[0] = tmp
    return 0


cdef object _copy_float64(float64_t* data, intp_t n):
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(<float64_t[:n]> data).copy()


cdef object _as_rows(object X):
    """``X`` as C-ordered float32, so each sample's features are contiguous."""
    return np.ascontiguousarray(X, dtype=np.float32)


cdef inline intp_t _find_leaf(const Node* nodes, const float32_t* row) noexcept nogil:
    """
    The leaf the sample with features ``row`` ends up in.

    Keep the three-way branch: with only the comparison, MSVC turns the choice of child into a
    conditional move, so every level waits for the comparison before the next node loads, where a
    branch lets the processor predict the way and load ahead. On a 300,000-sample tree that made
    traversal 2.8 times slower.
    """
    cdef const Node* node = nodes
    cdef float32_t value
    while node.left_child != LEAF:
        value = row[node.feature]
        if isnan(value):
            if node.missing_go_to_left:
                node = &nodes[node.left_child]
            else:
                node = &nodes[node.right_child]
        elif value <= node.threshold:
            node = &nodes[node.left_child]
        else:
            node = &nodes[node.right_child]
    return node - nodes


cdef intp_t _append_node(
    CostTree tree,
    intp_t parent,
    bint is_left,
    bint is_leaf,
    intp_t feature,
    float64_t threshold,
    bint missing_go_to_left,
    float64_t impurity,
    intp_t n_node_samples,
    float64_t weighted_n_node_samples,
    float64_t negative_fraction,
    float64_t positive_fraction,
    bint positive,
) except -1 nogil:
    cdef intp_t node_id = tree.node_count
    cdef Node* node
    if node_id >= tree.capacity:
        tree._resize(2 * tree.capacity if tree.capacity > 0 else 3)

    node = &tree.nodes[node_id]
    if parent != _TREE_UNDEFINED:
        if is_left:
            tree.nodes[parent].left_child = node_id
        else:
            tree.nodes[parent].right_child = node_id
    if is_leaf:
        node.left_child = _TREE_LEAF
        node.right_child = _TREE_LEAF
        node.feature = _TREE_UNDEFINED
        node.threshold = _TREE_UNDEFINED
        node.missing_go_to_left = 0
    else:
        # The children register themselves when they are added.
        node.feature = <int32_t> feature
        node.threshold = threshold
        node.missing_go_to_left = missing_go_to_left

    tree.impurity_[node_id] = impurity
    tree.n_node_samples_[node_id] = n_node_samples
    tree.weighted_n_node_samples_[node_id] = weighted_n_node_samples
    tree.value_[2 * node_id] = negative_fraction
    tree.value_[2 * node_id + 1] = positive_fraction
    tree.positive_[node_id] = positive
    tree.node_count += 1
    return node_id


cdef inline void _make_leaf(CostTree tree, intp_t node_id) noexcept nogil:
    cdef Node* node = &tree.nodes[node_id]
    node.left_child = _TREE_LEAF
    node.right_child = _TREE_LEAF
    node.feature = _TREE_UNDEFINED
    node.threshold = _TREE_UNDEFINED
    node.missing_go_to_left = 0


# ------------------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------------------

cdef struct StackRecord:
    intp_t start
    intp_t end
    intp_t depth
    intp_t parent
    bint is_left
    float64_t impurity
    intp_t n_constant_features


cdef struct FrontierRecord:
    intp_t node_id
    intp_t start
    intp_t end
    intp_t pos
    intp_t depth
    bint is_leaf
    float64_t impurity
    float64_t impurity_left
    float64_t impurity_right
    float64_t improvement


cdef inline bint _compare_records(const FrontierRecord& left, const FrontierRecord& right) noexcept nogil:
    return left.improvement < right.improvement


cdef struct BuildParams:
    intp_t min_samples_split
    intp_t min_samples_leaf
    float64_t min_weight_leaf
    intp_t max_depth
    float64_t min_impurity_decrease
    bint cost_bound
    # EPSILON in the units of the impurities, which scale with the costs.
    float64_t epsilon


def build_tree(
    CostTree tree,
    Splitter splitter,
    intp_t min_samples_split,
    intp_t min_samples_leaf,
    float64_t min_weight_leaf,
    intp_t max_depth,
    intp_t max_leaf_nodes,
    float64_t min_impurity_decrease,
    bint cost_bound,
    float64_t tolerance_scale=1.0,
):
    """
    Grow ``tree`` on the samples ``splitter`` was created with.

    Depth first, or best first (by impurity improvement) when ``max_leaf_nodes`` is positive.
    ``cost_bound`` makes a node a leaf without searching for a split when no split could lower its
    cost by ``min_impurity_decrease`` (see ``Splitter.max_cost_decrease``); it requires the cost
    criterion and a splitter that tracks the oracle. ``tolerance_scale`` is the typical cost of a
    sample, by which the rounding tolerance of the stopping rules is scaled, so that the same costs
    in other units grow the same tree.
    """
    cdef BuildParams params
    params.min_samples_split = min_samples_split
    params.min_samples_leaf = min_samples_leaf
    params.min_weight_leaf = min_weight_leaf
    params.max_depth = max_depth
    params.min_impurity_decrease = min_impurity_decrease
    params.cost_bound = cost_bound
    params.epsilon = EPSILON * tolerance_scale
    if splitter.n_samples == 0:
        raise ValueError('No sample has a positive weight.')
    if max_leaf_nodes > 0:
        _build_best_first(tree, splitter, &params, max_leaf_nodes)
    else:
        _build_depth_first(tree, splitter, &params)


cdef inline bint _is_unsplittable(
    Splitter splitter, const BuildParams* params, intp_t depth, intp_t n_node_samples,
    float64_t weighted_n_node_samples, float64_t impurity,
) noexcept nogil:
    return (
        depth >= params.max_depth
        or n_node_samples < params.min_samples_split
        or n_node_samples < 2 * params.min_samples_leaf
        or weighted_n_node_samples < 2 * params.min_weight_leaf
        # impurity == 0 with tolerance due to rounding errors
        or impurity <= params.epsilon
        or (params.cost_bound and splitter.max_cost_decrease() + params.epsilon < params.min_impurity_decrease)
    )


cdef void _build_depth_first(CostTree tree, Splitter splitter, const BuildParams* params) except *:
    cdef intp_t init_capacity
    if params.max_depth <= 10:
        init_capacity = <intp_t> (2 ** (params.max_depth + 1)) - 1
    else:
        init_capacity = 2047
    tree._resize(init_capacity)

    cdef intp_t start, end, depth, parent, node_id
    cdef bint is_left, is_leaf
    cdef intp_t n_node_samples
    cdef float64_t weighted_n_node_samples
    cdef float64_t impurity = INFINITY
    cdef intp_t n_constant_features
    cdef SplitRecord split
    cdef bint first = True
    cdef intp_t max_depth_seen = -1
    cdef stack[StackRecord] builder_stack
    cdef StackRecord record

    with nogil:
        builder_stack.push({
            'start': 0,
            'end': splitter.n_samples,
            'depth': 0,
            'parent': _TREE_UNDEFINED,
            'is_left': 0,
            'impurity': INFINITY,
            'n_constant_features': 0,
        })

        while not builder_stack.empty():
            record = builder_stack.top()
            builder_stack.pop()

            start = record.start
            end = record.end
            depth = record.depth
            parent = record.parent
            is_left = record.is_left
            impurity = record.impurity
            n_constant_features = record.n_constant_features

            n_node_samples = end - start
            splitter.node_reset(start, end, &weighted_n_node_samples)
            if first:
                impurity = splitter.node_impurity()
                first = False

            is_leaf = _is_unsplittable(splitter, params, depth, n_node_samples, weighted_n_node_samples, impurity)
            if not is_leaf:
                splitter.node_split(impurity, &split, &n_constant_features)
                # A small tolerance keeps splits whose improvement only rounds below the threshold.
                is_leaf = split.pos >= end or split.improvement + params.epsilon < params.min_impurity_decrease

            node_id = tree._add_node(
                parent, is_left, is_leaf, split.feature, split.threshold, split.missing_go_to_left, impurity,
                n_node_samples, weighted_n_node_samples, &splitter.total,
            )

            if not is_leaf:
                builder_stack.push({
                    'start': split.pos,
                    'end': end,
                    'depth': depth + 1,
                    'parent': node_id,
                    'is_left': 0,
                    'impurity': split.impurity_right,
                    'n_constant_features': n_constant_features,
                })
                builder_stack.push({
                    'start': start,
                    'end': split.pos,
                    'depth': depth + 1,
                    'parent': node_id,
                    'is_left': 1,
                    'impurity': split.impurity_left,
                    'n_constant_features': n_constant_features,
                })

            if depth > max_depth_seen:
                max_depth_seen = depth

        tree._resize(tree.node_count)
        tree.max_depth = max_depth_seen


cdef void _build_best_first(
    CostTree tree, Splitter splitter, const BuildParams* params, intp_t max_leaf_nodes
) except *:
    cdef vector[FrontierRecord] frontier
    cdef FrontierRecord record, split_node_left, split_node_right
    cdef intp_t max_split_nodes = max_leaf_nodes - 1
    cdef intp_t max_depth_seen = -1
    cdef bint is_leaf
    cdef intp_t node_id

    tree._resize(max_split_nodes + max_leaf_nodes)

    with nogil:
        _add_split_node(tree, splitter, params, 0, splitter.n_samples, True, True, _TREE_UNDEFINED, 0,
                        INFINITY, &split_node_left)
        frontier.push_back(split_node_left)
        push_heap(frontier.begin(), frontier.end(), &_compare_records)

        while not frontier.empty():
            pop_heap(frontier.begin(), frontier.end(), &_compare_records)
            record = frontier.back()
            frontier.pop_back()

            node_id = record.node_id
            is_leaf = record.is_leaf or max_split_nodes <= 0
            if is_leaf:
                _make_leaf(tree, node_id)
            else:
                max_split_nodes -= 1
                _add_split_node(tree, splitter, params, record.start, record.pos, False, True, node_id,
                                record.depth + 1, record.impurity_left, &split_node_left)
                _add_split_node(tree, splitter, params, record.pos, record.end, False, False, node_id,
                                record.depth + 1, record.impurity_right, &split_node_right)
                frontier.push_back(split_node_left)
                push_heap(frontier.begin(), frontier.end(), &_compare_records)
                frontier.push_back(split_node_right)
                push_heap(frontier.begin(), frontier.end(), &_compare_records)

            if record.depth > max_depth_seen:
                max_depth_seen = record.depth

        tree._resize(tree.node_count)
        tree.max_depth = max_depth_seen


cdef int _add_split_node(
    CostTree tree,
    Splitter splitter,
    const BuildParams* params,
    intp_t start,
    intp_t end,
    bint is_first,
    bint is_left,
    intp_t parent,
    intp_t depth,
    float64_t impurity,
    FrontierRecord* res,
) except -1 nogil:
    """Add the node of samples[start:end] to the tree, and describe its best split in ``res``."""
    cdef SplitRecord split
    cdef intp_t node_id
    cdef intp_t n_node_samples = end - start
    cdef float64_t weighted_n_node_samples
    # The best-first builder searches every node afresh for constant features.
    cdef intp_t n_constant_features = 0
    cdef bint is_leaf

    splitter.node_reset(start, end, &weighted_n_node_samples)
    if is_first:
        impurity = splitter.node_impurity()

    is_leaf = _is_unsplittable(splitter, params, depth, n_node_samples, weighted_n_node_samples, impurity)
    if not is_leaf:
        splitter.node_split(impurity, &split, &n_constant_features)
        is_leaf = split.pos >= end or split.improvement + params.epsilon < params.min_impurity_decrease

    node_id = tree._add_node(
        parent, is_left, is_leaf, split.feature, split.threshold, split.missing_go_to_left, impurity,
        n_node_samples, weighted_n_node_samples, &splitter.total,
    )

    res.node_id = node_id
    res.start = start
    res.end = end
    res.depth = depth
    res.impurity = impurity
    if not is_leaf:
        res.pos = split.pos
        res.is_leaf = False
        res.improvement = split.improvement
        res.impurity_left = split.impurity_left
        res.impurity_right = split.impurity_right
    else:
        res.pos = end
        res.is_leaf = True
        res.improvement = 0.0
        res.impurity_left = impurity
        res.impurity_right = impurity
    return 0


# ------------------------------------------------------------------------------------------------
# Minimal cost-complexity pruning
# ------------------------------------------------------------------------------------------------

cdef struct PruneRecord:
    intp_t node_idx
    intp_t parent


cdef tuple _cost_complexity_prune(CostTree tree, float64_t ccp_alpha):
    """
    Prune weakest links until the next would cost more than ``ccp_alpha``.

    Returns the leaves of the pruned tree as a mask over the nodes of ``tree``, and the effective
    alphas and total leaf impurities along the way.
    """
    cdef intp_t n_nodes = tree.node_count
    cdef float64_t total_sum_weights = tree.weighted_n_node_samples_[0]
    cdef float64_t[::1] r_node = np.empty(n_nodes, dtype=np.float64)
    cdef intp_t[::1] child_l = tree.children_left
    cdef intp_t[::1] child_r = tree.children_right
    cdef intp_t[::1] parent = np.zeros(n_nodes, dtype=np.intp)
    cdef intp_t[::1] n_leaves = np.zeros(n_nodes, dtype=np.intp)
    cdef float64_t[::1] r_branch = np.zeros(n_nodes, dtype=np.float64)
    cdef uint8_t[::1] leaves_in_subtree = np.zeros(n_nodes, dtype=np.uint8)
    cdef uint8_t[::1] candidate_nodes = np.zeros(n_nodes, dtype=np.uint8)
    cdef uint8_t[::1] in_subtree = np.ones(n_nodes, dtype=np.uint8)
    cdef float64_t[::1] ccp_alphas = np.zeros(n_nodes, dtype=np.float64)
    cdef float64_t[::1] impurities = np.zeros(n_nodes, dtype=np.float64)
    cdef intp_t count = 0

    cdef stack[PruneRecord] ccp_stack
    cdef PruneRecord record
    cdef stack[intp_t] node_indices_stack
    cdef intp_t i, node_idx, leaf_idx, parent_idx, pruned_branch_node_idx = 0, n_pruned_leaves
    cdef float64_t current_r, subtree_alpha, effective_alpha, r_diff
    cdef float64_t max_float64 = np.finfo(np.float64).max

    with nogil:
        for i in range(n_nodes):
            r_node[i] = tree.weighted_n_node_samples_[i] * tree.impurity_[i] / total_sum_weights

        ccp_stack.push({'node_idx': 0, 'parent': _TREE_UNDEFINED})
        while not ccp_stack.empty():
            record = ccp_stack.top()
            ccp_stack.pop()
            node_idx = record.node_idx
            parent[node_idx] = record.parent
            if child_l[node_idx] == _TREE_LEAF:
                leaves_in_subtree[node_idx] = 1
            else:
                ccp_stack.push({'node_idx': child_l[node_idx], 'parent': node_idx})
                ccp_stack.push({'node_idx': child_r[node_idx], 'parent': node_idx})

        # The number of leaves under every node, and their summed impurity.
        for leaf_idx in range(n_nodes):
            if not leaves_in_subtree[leaf_idx]:
                continue
            r_branch[leaf_idx] = r_node[leaf_idx]
            current_r = r_node[leaf_idx]
            while leaf_idx != 0:
                parent_idx = parent[leaf_idx]
                r_branch[parent_idx] += current_r
                n_leaves[parent_idx] += 1
                leaf_idx = parent_idx

        for i in range(n_nodes):
            candidate_nodes[i] = not leaves_in_subtree[i]

        ccp_alphas[count] = 0.0
        impurities[count] = r_branch[0]
        count += 1

        while candidate_nodes[0]:
            # The weakest link: the branch whose leaves save the least impurity per extra leaf.
            effective_alpha = max_float64
            for i in range(n_nodes):
                if not candidate_nodes[i]:
                    continue
                subtree_alpha = (r_node[i] - r_branch[i]) / (n_leaves[i] - 1)
                if subtree_alpha < effective_alpha:
                    effective_alpha = subtree_alpha
                    pruned_branch_node_idx = i

            if ccp_alpha < effective_alpha:
                break

            node_indices_stack.push(pruned_branch_node_idx)
            while not node_indices_stack.empty():
                node_idx = node_indices_stack.top()
                node_indices_stack.pop()
                if not in_subtree[node_idx]:
                    continue
                candidate_nodes[node_idx] = 0
                leaves_in_subtree[node_idx] = 0
                in_subtree[node_idx] = 0
                if child_l[node_idx] != _TREE_LEAF:
                    node_indices_stack.push(child_l[node_idx])
                    node_indices_stack.push(child_r[node_idx])
            leaves_in_subtree[pruned_branch_node_idx] = 1
            in_subtree[pruned_branch_node_idx] = 1

            n_pruned_leaves = n_leaves[pruned_branch_node_idx] - 1
            n_leaves[pruned_branch_node_idx] = 0

            r_diff = r_node[pruned_branch_node_idx] - r_branch[pruned_branch_node_idx]
            r_branch[pruned_branch_node_idx] = r_node[pruned_branch_node_idx]

            node_idx = parent[pruned_branch_node_idx]
            while node_idx != _TREE_UNDEFINED:
                n_leaves[node_idx] -= n_pruned_leaves
                r_branch[node_idx] += r_diff
                node_idx = parent[node_idx]

            ccp_alphas[count] = effective_alpha
            impurities[count] = r_branch[0]
            count += 1

    return (
        np.asarray(leaves_in_subtree).astype(bool),
        np.asarray(ccp_alphas)[:count].copy(),
        np.asarray(impurities)[:count].copy(),
    )


def ccp_pruning_path(CostTree tree):
    """The effective alphas of minimal cost-complexity pruning, and the total leaf impurity at each."""
    _, ccp_alphas, impurities = _cost_complexity_prune(tree, INFINITY)
    return {'ccp_alphas': ccp_alphas, 'impurities': impurities}


def prune_tree(CostTree tree, float64_t ccp_alpha):
    """Return the subtree with the largest cost complexity no greater than ``ccp_alpha``."""
    leaves_in_subtree, _, _ = _cost_complexity_prune(tree, ccp_alpha)
    return _build_pruned_tree(tree, leaves_in_subtree)


cdef struct BuildPrunedRecord:
    intp_t start
    intp_t depth
    intp_t parent
    bint is_left


cdef CostTree _build_pruned_tree(CostTree orig, object leaves_in_subtree):
    """Copy ``orig``, turning the nodes in ``leaves_in_subtree`` into leaves."""
    cdef const uint8_t[::1] leaves = np.ascontiguousarray(leaves_in_subtree, dtype=np.uint8)
    cdef CostTree tree = CostTree(orig.n_features)
    cdef stack[BuildPrunedRecord] prune_stack
    cdef BuildPrunedRecord record
    cdef intp_t orig_node_id, new_node_id, depth, max_depth_seen = -1
    cdef bint is_leaf
    cdef Node* node

    tree._resize(int(np.count_nonzero(leaves_in_subtree)) * 2)
    with nogil:
        prune_stack.push({'start': 0, 'depth': 0, 'parent': _TREE_UNDEFINED, 'is_left': 0})
        while not prune_stack.empty():
            record = prune_stack.top()
            prune_stack.pop()
            orig_node_id = record.start
            depth = record.depth
            is_leaf = leaves[orig_node_id]
            node = &orig.nodes[orig_node_id]

            new_node_id = _append_node(
                tree,
                record.parent,
                record.is_left,
                is_leaf,
                node.feature,
                node.threshold,
                node.missing_go_to_left,
                orig.impurity_[orig_node_id],
                orig.n_node_samples_[orig_node_id],
                orig.weighted_n_node_samples_[orig_node_id],
                orig.value_[2 * orig_node_id],
                orig.value_[2 * orig_node_id + 1],
                orig.positive_[orig_node_id],
            )
            if not is_leaf:
                prune_stack.push({'start': node.right_child, 'depth': depth + 1, 'parent': new_node_id, 'is_left': 0})
                prune_stack.push({'start': node.left_child, 'depth': depth + 1, 'parent': new_node_id, 'is_left': 1})
            if depth > max_depth_seen:
                max_depth_seen = depth
        tree._resize(tree.node_count)
        tree.max_depth = max_depth_seen
    return tree
