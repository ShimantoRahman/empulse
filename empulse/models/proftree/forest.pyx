import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, rand

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
        if indices[i] < best_index:
            best_index = indices[i]
    return population.trees[best_index]

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