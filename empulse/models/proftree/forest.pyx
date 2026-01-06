import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

from .tree cimport Tree, free_tree

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

cdef Forest* sort_population(Forest* population, Forest* sorted_population, float[:] fitness_scores) noexcept:
    cdef int i
    for i in range(population.n_trees):
        fitness_scores[i] = population.trees[i].fitness
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sorted_indices = np.argsort(fitness_scores)[::-1]
    for i in range(population.n_trees):
        sorted_population.trees[i] = population.trees[sorted_indices[i]]
    free(population.trees)
    population.trees = sorted_population.trees
    sorted_population.trees = <Tree**>malloc(population.n_trees * sizeof(Tree*))