from .tree cimport Tree, SplitValues
from .node cimport Node

cdef struct Children:
    Tree* son
    Tree* daughter

cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil

cdef Children* crossover(Tree* tree1, Tree* tree2, int max_depth) noexcept nogil

cdef void mutate(
    Tree* tree,
    int n_features,
    SplitValues* split_values,
    int max_depth,
) noexcept nogil
