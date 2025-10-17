from libc.stdlib cimport malloc, free
from libc.math cimport NAN

from .node cimport Node, free_node, is_leaf
from .tree cimport Tree, SplitValues, copy_tree, random_subnode_with_depth, grow_subtree
from .random cimport rand_int, rand_fraction

cdef struct Children:
    Tree* son
    Tree* daughter

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

cdef Children* crossover(Tree* tree1, Tree* tree2, int max_depth) noexcept nogil:
    cdef Tree* tree1_copy = copy_tree(tree1)
    cdef Tree* tree2_copy = copy_tree(tree2)

    cdef int depth1, depth2
    cdef Node* node1 = random_subnode_with_depth(tree1_copy.root, &depth1)
    cdef Node* node2 = random_subnode_with_depth(tree2_copy.root, &depth2)

    cdef Node* parent1 = node1.parent
    cdef Node* parent2 = node2.parent

    cdef int remaining_depth1 = max_depth - depth2
    cdef int remaining_depth2 = max_depth - depth1

    if remaining_depth1 >= 0:
        prune_subtree_at_depth(node2, remaining_depth1)
    else:
        free_node(node2.left)
        free_node(node2.right)
        node2.left = NULL
        node2.right = NULL

    if remaining_depth2 >= 0:
        prune_subtree_at_depth(node1, remaining_depth2)
    else:
        free_node(node1.left)
        free_node(node1.right)
        node1.left = NULL
        node1.right = NULL

    if parent1 is NULL:
        tree1_copy.root = node2
        node2.parent = NULL
    else:
        if parent1.left is node1:
            parent1.left = node2
        else:
            parent1.right = node2
        node2.parent = parent1

    if parent2 is NULL:
        tree2_copy.root = node1
        node1.parent = NULL
    else:
        if parent2.left is node2:
            parent2.left = node1
        else:
            parent2.right = node1
        node1.parent = parent2

    cdef Children* children = <Children*>malloc(sizeof(Children))
    children.son = tree1_copy
    children.daughter = tree2_copy
    children.son.fitness = NAN
    children.daughter.fitness = NAN
    return children

cdef void mutate(
        Tree* tree,
        int n_features,
        SplitValues* split_values,
        int max_depth,
) noexcept nogil:
    tree.fitness = NAN

    cdef int depth
    cdef Node* node = random_subnode_with_depth(tree.root, &depth)
    cdef int choice
    cdef int idx
    if is_leaf(node):
        grow_subtree(
            node=node,
            n_features=n_features,
            split_values=split_values,
            depth=depth,
            max_depth=max_depth,
        )
    else:
        choice = rand_int(0, 3)
        idx
        if choice == 0:
            node.feature_index = rand_int(0, n_features)
            idx = rand_int(0, split_values.lengths[node.feature_index])
            node.split_value = split_values.values[node.feature_index][idx]
        elif choice == 1:
            idx = rand_int(0, split_values.lengths[node.feature_index])
            node.split_value = split_values.values[node.feature_index][idx]
        else:
            grow_subtree(
                node=node,
                n_features=n_features,
                split_values=split_values,
                depth=depth,
                max_depth=max_depth,
            )