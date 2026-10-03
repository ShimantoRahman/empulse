# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from .node cimport Node, create_node, copy_node, free_node, is_leaf, node_probability, reset_node
from .random cimport RandState, rand_int, rand_bool

cdef struct Tree:
    Node* root
    float fitness
    int n_nodes
    # Root of the only subtree whose sample counts may be out of date, or NULL if all are current.
    Node* stale

cdef Tree* create_tree(bint with_root = True) noexcept nogil:
    cdef Tree* tree = <Tree*>malloc(sizeof(Tree))
    tree.stale = NULL
    if with_root:
        tree.root = create_node()
        tree.n_nodes = 1
        tree.stale = tree.root  # never fitted
    else:
        tree.n_nodes = 0
    tree.fitness = -1.0
    return tree

cdef Tree* copy_tree(Tree* tree) noexcept nogil:
    cdef Tree* new_tree = <Tree*>malloc(sizeof(Tree))
    new_tree.root = copy_node(tree.root, NULL)
    new_tree.fitness = tree.fitness
    new_tree.n_nodes = tree.n_nodes
    # The copy has its own nodes; if any of the original's counts were stale, refit it all.
    new_tree.stale = NULL
    if tree.stale is not NULL:
        new_tree.stale = new_tree.root
    return new_tree

cdef void free_tree(Tree* tree) noexcept nogil:
    if tree is NULL:
        return
    free_node(tree.root)
    free(tree)

cdef void reset_tree(Tree* tree) noexcept nogil:
    """Recursively reset node statistics in the tree."""
    reset_node(tree.root)

cdef object serialize_node(Node* node):
    """Serialize a node and its children recursively."""
    if node is NULL:
        return None

    return {
        'split_value': node.split_value,
        'feature_index': node.feature_index,
        'n_samples': node.n_samples,
        'n_positive_samples': node.n_positive_samples,
        'left': serialize_node(node.left),
        'right': serialize_node(node.right)
    }

cdef object serialize_tree(Tree* tree):
    """Convert tree structure to a serializable dictionary."""
    if tree is NULL:
        return None

    return {
        'root': serialize_node(tree.root),
        'fitness': tree.fitness,
        'n_nodes': tree.n_nodes
    }

cdef Node* deserialize_node(object node_data, Node* parent = NULL) noexcept:
    """Deserialize a node and its children recursively."""
    if node_data is None:
        return NULL

    cdef Node* node = create_node()
    node.split_value = node_data['split_value']
    node.feature_index = node_data['feature_index']
    node.n_samples = node_data['n_samples']
    node.n_positive_samples = node_data['n_positive_samples']

    node.parent = parent
    node.left = deserialize_node(node_data['left'], node)
    node.right = deserialize_node(node_data['right'], node)

    return node

cdef Tree* deserialize_tree(object tree_data) noexcept:
    """Reconstruct tree from serialized dictionary."""
    if tree_data is None:
        return NULL

    cdef Tree* tree = <Tree*> malloc(sizeof(Tree))
    tree.root = deserialize_node(tree_data['root'], NULL)
    tree.fitness = tree_data['fitness']
    tree.n_nodes = tree_data['n_nodes']
    tree.stale = NULL  # the counts are restored with the nodes

    return tree

cdef Node* get_leaf(Node* start_node, const float* x) noexcept nogil:
    cdef Node* node = start_node
    while not is_leaf(node):
        if x[node.feature_index] <= node.split_value:
            node = node.left
        else:
            node = node.right
    return node

cdef void sum_counts(Node* node) noexcept nogil:
    """Set the counts of every internal node from ``node`` down to the sums of its children's."""
    if is_leaf(node):
        return
    sum_counts(node.left)
    sum_counts(node.right)
    node.n_samples = node.left.n_samples + node.right.n_samples
    node.n_positive_samples = node.left.n_positive_samples + node.right.n_positive_samples

# Samples are routed through a flat copy of the nodes in which every leaf is its own child, so each
# sample takes the same number of steps and picks the next node by indexing rather than branching.
# A branch per node is mispredicted about as often as the data is unpredictable. Stopping early at a
# leaf brings that branch back and was slower, even for deep trees.

cdef enum:
    LANES = 16

cdef struct FlatNode:
    int feature_index
    float split_value
    int children[2]  # left, right

cdef int flatten(
    Node* node, vector[FlatNode]& flat, vector[Node*]& nodes, int depth, int* max_depth
) noexcept nogil:
    cdef int index = <int>flat.size()
    cdef FlatNode flat_node
    flat_node.feature_index = 0
    flat_node.split_value = 0.0
    flat_node.children[0] = index
    flat_node.children[1] = index
    if not is_leaf(node):
        flat_node.feature_index = node.feature_index
        flat_node.split_value = node.split_value
    flat.push_back(flat_node)
    nodes.push_back(node)
    if is_leaf(node):
        if depth > max_depth[0]:
            max_depth[0] = depth
        return index
    cdef int left = flatten(node.left, flat, nodes, depth + 1, max_depth)
    cdef int right = flatten(node.right, flat, nodes, depth + 1, max_depth)
    flat[index].children[0] = left
    flat[index].children[1] = right
    return index

cdef void count_samples(
    Node* subtree, const vector[FlatNode]& path, const float[:, ::1] X, const int[:] y, int n_samples
) noexcept nogil:
    """
    Count the samples that follow ``path`` in the leaves of ``subtree`` they reach.

    ``path`` holds the split rules from the root down to ``subtree``; its first node is a sink that
    collects the samples leaving the path and each step's other child points to it.
    ``sum_counts`` then fills in the internal nodes of ``subtree``.
    """
    cdef vector[FlatNode] flat = path
    cdef vector[Node*] nodes
    nodes.resize(flat.size(), NULL)
    cdef int start = 1 if path.size() > 0 else 0
    cdef int depth = <int>path.size() - start
    cdef int subtree_depth = 0
    flatten(subtree, flat, nodes, 0, &subtree_depth)
    depth += subtree_depth

    # A node's samples in the low 32 bits and its positive samples in the high 32 bits, so that
    # counting a sample is a single addition.
    cdef vector[unsigned long long] packed_counts
    packed_counts.resize(flat.size(), 0)
    cdef unsigned long long* counts = packed_counts.data()
    cdef const FlatNode* nodes_ = flat.data()
    # Samples are routed LANES at a time in lockstep: each step of one sample waits on the previous
    # one, but the steps of different samples are independent and overlap however deep the tree is.
    cdef const float* rows[LANES]
    cdef int lanes[LANES]
    cdef Py_ssize_t i = 0
    cdef int step, lane, n
    while i + LANES <= n_samples:
        for lane in range(LANES):
            rows[lane] = &X[i + lane, 0]
            lanes[lane] = start
        for step in range(depth):
            for lane in range(LANES):
                n = lanes[lane]
                lanes[lane] = nodes_[n].children[not (rows[lane][nodes_[n].feature_index] <= nodes_[n].split_value)]
        for lane in range(LANES):
            counts[lanes[lane]] += 1 + (<unsigned long long>y[i + lane] << 32)
        i += LANES
    cdef const float* x
    while i < n_samples:
        x = &X[i, 0]
        n = start
        for step in range(depth):
            n = nodes_[n].children[not (x[nodes_[n].feature_index] <= nodes_[n].split_value)]
        counts[n] += 1 + (<unsigned long long>y[i] << 32)
        i += 1

    cdef size_t k
    for k in range(path.size(), flat.size()):
        if is_leaf(nodes[k]):
            nodes[k].n_samples = <int>(<unsigned int>counts[k])
            nodes[k].n_positive_samples = <int>(counts[k] >> 32)
    sum_counts(subtree)

cdef void fit_tree(Tree* tree, const float[:, ::1] X, const int[:] y, int n_samples) noexcept nogil:
    cdef vector[FlatNode] no_path
    count_samples(tree.root, no_path, X, y, n_samples)

cdef void refit_tree(
    Tree* tree,
    const float[:, ::1] X,
    const int[:] y,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
) noexcept nogil:
    """
    Bring the tree's sample counts up to date after a variation operator changed it, then prune it.

    An operator only changes the subtree below one node (``tree.stale``): the samples reaching that
    node, and the counts everywhere outside its subtree, stay the same. So only the samples that
    follow the path from the root to the stale node are counted again, and only in its subtree.
    The rest of the tree was already pruned when its counts were last computed, and pruning only
    looks at counts, so it is revisited only below the stale node too.
    """
    cdef Node* stale = tree.stale
    if stale is NULL:
        return
    tree.stale = NULL
    if stale is tree.root:
        fit_tree(tree, X, y, n_samples)
        prune_illegal_nodes(tree, tree.root, min_samples_split, min_samples_leaf)
        return

    # The split rules leading to the stale node, from the root down, after the sink at index 0.
    cdef vector[Node*] ancestors
    cdef Node* node = stale
    while node.parent is not NULL:
        ancestors.push_back(node.parent)
        node = node.parent
    cdef vector[FlatNode] path
    cdef FlatNode step
    step.feature_index = 0
    step.split_value = 0.0
    step.children[0] = 0
    step.children[1] = 0
    path.push_back(step)
    cdef Py_ssize_t k
    cdef Node* on_path
    for k in range(<Py_ssize_t>ancestors.size() - 1, -1, -1):
        node = ancestors[k]
        on_path = stale if k == 0 else ancestors[k - 1]
        step.feature_index = node.feature_index
        step.split_value = node.split_value
        step.children[0] = <int>path.size() + 1 if node.left is on_path else 0
        step.children[1] = <int>path.size() + 1 if node.right is on_path else 0
        path.push_back(step)

    count_samples(stale, path, X, y, n_samples)
    prune_illegal_nodes(tree, stale, min_samples_split, min_samples_leaf)

cdef void predict_proba_tree(
    Tree* tree, const float[:, ::1] X, float[:] probabilities, int n_samples
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Node* leaf
    for i in range(n_samples):
        leaf = get_leaf(tree.root, &X[i, 0])
        probabilities[i] = node_probability(leaf)

cdef void predict_labels_tree(Tree* tree, const float[:, ::1] X, float[:] probabilities, int n_samples):
    predict_proba_tree(tree, X, probabilities, n_samples)
    for i in range(n_samples):
        probabilities[i] = 1 if probabilities[i] >= 0.5 else 0

cdef struct SplitValues:
    float **values
    int *lengths
    int n_features
    int *splittable_features  # features with at least two distinct values
    int n_splittable

cdef SplitValues* compute_split_values(cnp.ndarray[cnp.float32_t, ndim=2] X) noexcept:
    cdef int n_features = <int>X.shape[1]
    cdef SplitValues* sv = <SplitValues*>malloc(sizeof(SplitValues))
    sv.n_features = n_features
    sv.values = <float **>malloc(n_features * sizeof(float *))
    sv.lengths = <int *>malloc(n_features * sizeof(int))
    sv.splittable_features = <int *>malloc(n_features * sizeof(int))
    sv.n_splittable = 0

    cdef int j, n_unique, i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vals
    for j in range(n_features):
        vals = np.unique(X[:, j])
        n_unique = <int>vals.shape[0]
        sv.lengths[j] = n_unique
        sv.values[j] = <float *>malloc(n_unique * sizeof(float))
        for i in range(n_unique):
            sv.values[j][i] = <float>vals[i]
        if n_unique >= 2:
            sv.splittable_features[sv.n_splittable] = j
            sv.n_splittable += 1
    return sv

cdef void free_split_values(SplitValues* sv) noexcept nogil:
    cdef int j
    for j in range(sv.n_features):
        free(sv.values[j])
    free(sv.values)
    free(sv.lengths)
    free(sv.splittable_features)
    free(sv)

cdef void split(
    RandState* rng,
    Node* node,
    int n_features,
    SplitValues* split_values,
    int depth,
    int max_depth,
) noexcept nogil:
    """Add a randomly generated split rule to a randomly selected leaf node."""
    if max_depth == -1:
        max_depth = depth

    # A feature with a single distinct value cannot split anything; drawing a split value for it
    # would also take a random integer modulo zero, which kills the process.
    if depth < max_depth and split_values.n_splittable > 0:
        node.feature_index = split_values.splittable_features[rand_int(rng, 0, split_values.n_splittable)]
        # Never the largest value: every sample would go left.
        split_value_index = rand_int(rng, 0, split_values.lengths[node.feature_index] - 1)
        node.split_value = split_values.values[node.feature_index][split_value_index]
        node.left = create_node()
        node.left.parent = node
        node.right = create_node()
        node.right.parent = node

cdef void prune(Node* node) noexcept nogil:
    """Prune an internal node into a leaf node by freeing both children."""
    if node is NULL:
        return

    free_node(node.left)
    free_node(node.right)
    node.left = NULL
    node.right = NULL

cdef void prune_illegal_nodes(Tree* tree, Node* node, int min_samples_split, int min_samples_leaf) noexcept nogil:
    """Prune nodes that violate min_samples_split or min_samples_leaf constraints."""
    if node is NULL or is_leaf(node):
        return

    # Recursively check children first
    if node.left is not NULL:
        prune_illegal_nodes(tree, node.left, min_samples_split, min_samples_leaf)
    if node.right is not NULL:
        prune_illegal_nodes(tree, node.right, min_samples_split, min_samples_leaf)

    if node is tree.root:
        return

    if node.n_samples < min_samples_split:
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
        return

    if node.left is not NULL and node.left.n_samples < min_samples_leaf:
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
        return

    if node.right is not NULL and node.right.n_samples < min_samples_leaf:
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
        return

cdef Node* random_subnode(RandState* rng, Node* root) noexcept nogil:

    cdef Node* node = root
    while True:
        if is_leaf(node):
            return node.parent
        if rand_int(rng, 0, 3) == 0:
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool(rng):
                node = node.left
            else:
                node = node.right
        elif node.left is not NULL:
            node = node.left
        elif node.right is not NULL:
            node = node.right
        else:
            return node

cdef Node* random_subnode_with_depth(RandState* rng, Node* root, int* out_depth) noexcept nogil:

    cdef Node* node = root
    cdef int depth = 0
    while True:
        if is_leaf(node) or rand_int(rng, 0, 3) == 0:
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool(rng):
                node = node.left
            else:
                node = node.right
        elif node.left is not NULL:
            node = node.left
        elif node.right is not NULL:
            node = node.right
        else:
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        depth += 1

cdef Node* random_leaf_node(RandState* rng, Node* root, int* out_depth) noexcept nogil:
    """Select a random leaf node and return its depth."""
    cdef Node* node = root
    cdef int depth = 0
    while True:
        if is_leaf(node):
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool(rng):
                node = node.left
            else:
                node = node.right
        elif node.left is not NULL:
            node = node.left
        elif node.right is not NULL:
            node = node.right
        else:
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        depth += 1

cdef struct CandidateSearch:
    Node* candidate
    int count

cdef void _find_candidate_helper(RandState* rng, Node* n, CandidateSearch* search) noexcept nogil:
    if n is NULL or is_leaf(n):
        return

    if (n.left is not NULL and is_leaf(n.left) and
            n.right is not NULL and is_leaf(n.right)):
        search.count += 1
        # Reservoir sampling: select with probability 1/count
        if rand_int(rng, 0, search.count) == 0:
            search.candidate = n

    if n.left is not NULL:
        _find_candidate_helper(rng, n.left, search)
    if n.right is not NULL:
        _find_candidate_helper(rng, n.right, search)

cdef Node* random_subnode_with_leaf_children(RandState* rng, Node* root) noexcept nogil:
    """Select a random internal node that has two leaf children."""
    cdef CandidateSearch search
    search.candidate = NULL
    search.count = 0

    _find_candidate_helper(rng, root, &search)
    return search.candidate