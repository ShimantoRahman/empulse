# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

from .tree cimport Tree, free_tree, copy_tree
from .random cimport rand_int

cdef struct Forest:
    Tree** trees
    int n_trees

cdef Forest* create_forest(int n_trees) noexcept nogil:
    cdef Forest* forest = <Forest*>malloc(sizeof(Forest))
    forest.n_trees = n_trees
    forest.trees = <Tree**>malloc(n_trees * sizeof(Tree*))
    return forest

cdef void free_forest(Forest* forest) noexcept nogil:
    if forest is NULL:
        return
    cdef int i
    for i in range(forest.n_trees):
        free_tree(forest.trees[i])
    free(forest.trees)
    free(forest)

cdef Tree* choose_different_tree(Forest* population, int current_index) noexcept nogil:
    """Choose a random tree from population that is different from the current index."""
    cdef int partner_index

    if population.n_trees <= 1:
        return copy_tree(population.trees[0])

    # Keep selecting until we get a different index
    partner_index = rand_int(0, population.n_trees)
    while partner_index == current_index:
        partner_index = rand_int(0, population.n_trees)

    return copy_tree(population.trees[partner_index])
