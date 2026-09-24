# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport reverse
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

cdef Node* visit_leaf(Node* start_node, const float* x, int y) noexcept nogil:
    """Traverse until the relevant leaf node and update stats."""
    cdef Node* node = start_node
    while not is_leaf(node):
        node.n_samples += 1
        node.n_positive_samples += y
        if x[node.feature_index] <= node.split_value:
            node = node.left
        else:
            node = node.right
    node.n_samples += 1
    node.n_positive_samples += y
    return node

cdef void fit_tree(Tree* tree, const float[:, ::1] X, const int[:] y, int n_samples) noexcept nogil:
    # Rows are passed on as pointers: slicing X[i] would create a memoryview per sample, and its
    # reference counting was a sizeable share of the whole fit.
    cdef Py_ssize_t i
    for i in range(n_samples):
        visit_leaf(tree.root, &X[i, 0], y[i])

cdef struct PathStep:
    int feature_index
    float split_value
    bint goes_left

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
    follow the path from the root to the stale node are routed again, and only through its subtree.
    The rest of the tree was already pruned when its counts were last computed, and pruning only
    looks at counts, so it is revisited only below the stale node too.
    """
    cdef Node* stale = tree.stale
    if stale is NULL:
        return
    tree.stale = NULL
    if stale is tree.root:
        reset_node(tree.root)
        fit_tree(tree, X, y, n_samples)
        prune_illegal_nodes(tree, tree.root, min_samples_split, min_samples_leaf)
        return

    # The split rules leading to the stale node, from the root down.
    cdef vector[PathStep] path
    cdef Node* child = stale
    cdef Node* parent = stale.parent
    while parent is not NULL:
        path.push_back(PathStep(parent.feature_index, parent.split_value, parent.left is child))
        child = parent
        parent = parent.parent
    reverse(path.begin(), path.end())

    reset_node(stale)
    cdef Py_ssize_t i
    cdef size_t step
    cdef const float* x
    cdef bint reaches_stale
    for i in range(n_samples):
        x = &X[i, 0]
        reaches_stale = True
        for step in range(path.size()):
            if (x[path[step].feature_index] <= path[step].split_value) != path[step].goes_left:
                reaches_stale = False
                break
        if reaches_stale:
            visit_leaf(stale, x, y[i])
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

    # Simply free both children - don't recurse
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

    # Check if this node violates min_samples_split
    if node.n_samples < min_samples_split:
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
        return

    # Check if children violate min_samples_leaf
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

    # Check if this node has two leaf children
    if (n.left is not NULL and is_leaf(n.left) and
            n.right is not NULL and is_leaf(n.right)):
        search.count += 1
        # Reservoir sampling: select with probability 1/count
        if rand_int(rng, 0, search.count) == 0:
            search.candidate = n

    # Recurse to children
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