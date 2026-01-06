import numpy as np
from libc.stdlib cimport malloc, rand
from libc.math cimport NAN

from .forest cimport Forest
from .node cimport Node, free_node, is_leaf
from .tree cimport Tree, SplitValues, copy_tree, random_subnode_with_depth, prune, split, random_leaf_node
from .random cimport rand_int, rand_fraction


cdef Tree* tournament_selection(Forest* population, int tournament_size) noexcept:
    cdef int n = population.n_trees
    cdef int i, j, index
    cdef int selected
    cdef int[:] indices = np.empty(tournament_size, dtype=np.int32)
    cdef bint is_duplicate

    for i in range(tournament_size):
        while True:
            index = rand() % n
            is_duplicate = False
            for j in range(i):
                if indices[j] == index:
                    is_duplicate = True
                    break
            if not is_duplicate:
                indices[i] = index
                break

    cdef int best_index = indices[0]
    for i in range(1, tournament_size):
        if population.trees[indices[i]].fitness < population.trees[best_index].fitness:
            best_index = indices[i]
    return population.trees[best_index]

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


cdef Tree* evolve(
    Tree* old_tree,
    Forest* population,
    SplitValues* split_values,
    int n_features,
    int max_depth,
    int tournament_size,
):
    """Apply genetic operators to the tree."""
    cdef Tree* tree = copy_tree(old_tree)

    cdef int choice, idx, depth
    cdef Node* node
    cdef Tree* partner
    choice = rand_int(1, 5)
    if choice == 1:
        node = random_subnode_with_depth(tree.root, &depth)
        mutate_split_feature(node, n_features, split_values)
    elif choice == 2:
        node = random_subnode_with_depth(tree.root, &depth)
        mutate_split_value(node, split_values)
    elif choice == 3:
        node = random_subnode_with_depth(tree.root, &depth)
        prune(node)
    elif choice == 4:
        node = random_leaf_node(tree.root, &depth)
        if depth < max_depth:
            split(node, n_features, split_values, depth, max_depth)
    else:
        partner = tournament_selection(population, tournament_size)
        tree = crossover(tree, partner, max_depth)
    return tree

cdef inline void mutate_split_feature(
    Node* node,
    int n_features,
    SplitValues* split_values,
) noexcept nogil:
    node.feature_index = rand_int(0, n_features)
    idx = rand_int(0, split_values.lengths[node.feature_index])
    node.split_value = split_values.values[node.feature_index][idx]

cdef inline void mutate_split_value(Node* node, SplitValues* split_values) noexcept nogil:
    idx = rand_int(0, split_values.lengths[node.feature_index])
    node.split_value = split_values.values[node.feature_index][idx]