from .tree cimport Tree
from .random cimport RandState

cdef struct Forest:
    Tree** trees
    int n_trees

cdef Forest* create_forest(int n_trees) noexcept nogil
cdef void free_forest(Forest* forest) noexcept nogil

cdef Tree* choose_different_tree(RandState* rng, Forest* population, int current_index) noexcept nogil