import numpy as np
from libc.stdlib cimport malloc, rand
from libc.math cimport NAN

from .forest cimport Forest
from .node cimport Node, free_node, is_leaf
from .tree cimport Tree, SplitValues, copy_tree, random_subnode_with_depth, prune, split, random_leaf_node, random_subnode_with_leaf_children
from .random cimport rand_int, rand_fraction

cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil:
    if node is NULL or max_depth < 0:
        return

    if max_depth == 0:
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
    else:
        prune_subtree_at_depth(node.left, max_depth - 1)
        prune_subtree_at_depth(node.right, max_depth - 1)

cdef Tree* crossover(Tree* mother, Tree* father, int max_depth) noexcept nogil:
    cdef Tree* child = copy_tree(mother)

    cdef int mother_depth, father_depth
    cdef Node* mother_node = random_subnode_with_depth(mother.root, &mother_depth)
    cdef Node* father_node = random_subnode_with_depth(father.root, &father_depth)

    cdef Node* parent_node = mother_node.parent

    cdef int remaining_depth = max_depth - father_depth

    if remaining_depth >= 0:
        prune_subtree_at_depth(mother_node, remaining_depth)
    else:
        free_node(mother_node.left)
        free_node(mother_node.right)
        mother_node.left = NULL
        mother_node.right = NULL

    if parent_node is NULL:
        child.root = mother_node
        mother_node.parent = NULL
    else:
        if parent_node.left is mother_node:
            parent_node.left = father_node
        else:
            parent_node.right = father_node
        mother_node.parent = parent_node

    child.fitness = NAN
    return child

cdef inline void grow(Tree* tree, SplitValues* split_values, int n_features, int max_depth):
    """Add a random split rule to a random leaf node."""
    cdef int depth = 0
    cdef Node* leaf = random_leaf_node(tree.root, &depth)
    split(leaf, n_features, split_values, depth, max_depth)

cdef inline void prune_internal(Tree* tree):
    """Prune a random internal node which has two leaf nodes as successors."""
    cdef Node* node = random_subnode_with_leaf_children(tree.root)
    if node is not NULL:
        prune(node)

cdef inline void mutate_split_feature(
    Tree* tree,
    int n_features,
    SplitValues* split_values,
) noexcept nogil:
    cdef int depth = 0
    cdef Node* node = random_subnode_with_depth(tree.root, &depth)
    node.feature_index = rand_int(0, n_features)
    cdef int idx = rand_int(0, split_values.lengths[node.feature_index])
    node.split_value = split_values.values[node.feature_index][idx]

cdef inline void mutate_split_value(Tree* tree, SplitValues* split_values) noexcept nogil:
    cdef int depth = 0
    cdef Node* node = random_subnode_with_depth(tree.root, &depth)
    cdef int idx = rand_int(0, split_values.lengths[node.feature_index])
    node.split_value = split_values.values[node.feature_index][idx]