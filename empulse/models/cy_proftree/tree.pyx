import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport NAN

from .node cimport Node, create_node, copy_node, free_node, is_leaf, node_probability, update_node_stats
from .random cimport rand_int, rand_bool

cdef struct Tree:
    Node* root
    float fitness

cdef Tree* create_tree(bint with_root = True) noexcept nogil:
    cdef Tree* tree = <Tree*>malloc(sizeof(Tree))
    if with_root:
        tree.root = create_node()
    tree.fitness = -1.0
    return tree

cdef Tree* copy_tree(Tree* tree) noexcept nogil:
    cdef Tree* new_tree = <Tree*>malloc(sizeof(Tree))
    new_tree.root = copy_node(tree.root, NULL)
    new_tree.fitness = tree.fitness
    return new_tree

cdef void free_tree(Tree* tree) noexcept nogil:
    if tree is NULL:
        return
    free_node(tree.root)
    free(tree)

cdef Node* get_leaf(Node* start_node, float[:] x) noexcept nogil:
    cdef Node* node = start_node
    while not is_leaf(node):
        if x[node.feature_index] <= node.split_value:
            node = node.left
        else:
            node = node.right
    return node

cdef Node* visit_leaf(Node* start_node, float[:] x, int y) noexcept nogil:
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

cdef void fit_tree(Tree* tree, float[:, :] X, int[:] y, int n_samples) noexcept nogil:
    cdef float[:] x_i
    cdef int y_i
    cdef Node* leaf
    for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]
        leaf = visit_leaf(tree.root, x_i, y)

cdef void predict_proba_tree(Tree* tree, float[:, :] X, float[:] probabilities, int n_samples) noexcept nogil:
    for i in range(n_samples):
        leaf = get_leaf(tree.root, X[i])
        probabilities[i] = node_probability(leaf)

cdef void predict_labels_tree(Tree* tree, float[:, :] X, float[:] probabilities, int n_samples):
    predict_proba_tree(tree, X, probabilities, n_samples)
    for i in range(n_samples):
        probabilities[i] = 1 if probabilities[i] >= 0.5 else 0

cdef struct SplitValues:
    float **values
    int *lengths
    int n_features

cdef SplitValues* compute_split_values(cnp.ndarray[cnp.float32_t, ndim=2] X) noexcept:
    cdef int n_features = <int>X.shape[1]
    cdef SplitValues* sv = <SplitValues*>malloc(sizeof(SplitValues))
    sv.n_features = n_features
    sv.values = <float **>malloc(n_features * sizeof(float *))
    sv.lengths = <int *>malloc(n_features * sizeof(int))

    cdef int j, n_unique, i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vals
    for j in range(n_features):
        vals = np.unique(X[:, j])
        n_unique = <int>vals.shape[0]
        sv.lengths[j] = n_unique
        sv.values[j] = <float *>malloc(n_unique * sizeof(float))
        for i in range(n_unique):
            sv.values[j][i] = <float>vals[i]
    return sv

cdef void free_split_values(SplitValues* sv) noexcept nogil:
    cdef int j
    for j in range(sv.n_features):
        free(sv.values[j])
    free(sv.values)
    free(sv.lengths)
    free(sv)

cdef void split(
    Node* node,
    int n_features,
    SplitValues* split_values,
    int depth,
    int max_depth,
) noexcept nogil:
    """Add a randomly generated split rule to a randomly selected leaf node."""
    if max_depth == -1:
        max_depth = depth

    if depth < max_depth:
        node.feature_index = rand_int(0, n_features)
        split_value_index = rand_int(0, split_values.lengths[node.feature_index] - 1)
        node.split_value = split_values.values[node.feature_index][split_value_index]
        node.left = create_node()
        node.left.parent = node
        node.right = create_node()
        node.right.parent = node

cdef void prune(Node* node) noexcept nogil:
    """Prune an internal node into a leaf node."""

    if node.left is not NULL:
        prune(node.left)
    if node.right is not NULL:
        prune(node.right)
    if is_leaf(node):  # TODO: this check is probably unnecessary
        if node.parent.left == node:
            node.parent.left = NULL
        else:
            node.parent.right = NULL
        free_node(node)

cdef void prune_illegal_nodes(Node* node, int min_samples_split, int min_samples_leaf) noexcept nogil:
    """Prune nodes that violate min_samples_split or min_samples_leaf constraints."""
    if node is NULL or is_leaf(node):
        return

    # Recursively check children first
    if node.left is not NULL:
        prune_illegal_nodes(node.left, min_samples_split, min_samples_leaf)
    if node.right is not NULL:
        prune_illegal_nodes(node.right, min_samples_split, min_samples_leaf)

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

cdef Node* random_subnode(Node* root) noexcept nogil:

    cdef Node* node = root
    while True:
        if is_leaf(node) or rand_int(0, 3) == 0:
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool():
                node = node.left
            else:
                node = node.right
        elif node.left is not NULL:
            node = node.left
        elif node.right is not NULL:
            node = node.right
        else:
            return node

cdef Node* random_subnode_with_depth(Node* root, int* out_depth) noexcept nogil:

    cdef Node* node = root
    cdef int depth = 0
    while True:
        if is_leaf(node) or rand_int(0, 3) == 0:
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool():
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

cdef Node* random_leaf_node(Node* root, int* out_depth) noexcept nogil:
    """Select a random leaf node and return its depth."""
    cdef Node* node = root
    cdef int depth = 0
    while True:
        if is_leaf(node):
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool():
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

cdef void _find_candidate_helper(Node* n, CandidateSearch* search) noexcept nogil:
    if n is NULL or is_leaf(n):
        return

    # Check if this node has two leaf children
    if (n.left is not NULL and is_leaf(n.left) and
            n.right is not NULL and is_leaf(n.right)):
        search.count += 1
        # Reservoir sampling: select with probability 1/count
        if rand_int(0, search.count) == 0:
            search.candidate = n

    # Recurse to children
    if n.left is not NULL:
        _find_candidate_helper(n.left, search)
    if n.right is not NULL:
        _find_candidate_helper(n.right, search)

cdef Node* random_subnode_with_leaf_children(Node* root) noexcept nogil:
    """Select a random internal node that has two leaf children."""
    cdef CandidateSearch search
    search.candidate = NULL
    search.count = 0

    _find_candidate_helper(root, &search)
    return search.candidate