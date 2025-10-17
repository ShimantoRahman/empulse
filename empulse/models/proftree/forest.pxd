from .tree cimport Tree

cdef struct Forest:
    Tree** trees
    int n_trees

cdef Forest* create_forest(int n_trees) noexcept nogil
cdef void free_forest(Forest* forest) noexcept nogil

cdef Tree* tournament_selection(Forest* population, int tournament_size) noexcept

cdef Forest* sort_population(Forest* population, Forest* sorted_population, float[:] fitness_scores) noexcept
