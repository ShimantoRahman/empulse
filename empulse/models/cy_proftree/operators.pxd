from .tree cimport Tree, SplitValues
from .node cimport Node

cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil

cdef int count_nodes(Node* node) noexcept nogil

cdef Tree* crossover(Tree* mother, Tree* father, int max_depth) noexcept nogil

cdef void grow(Tree* tree, SplitValues* split_values, int n_features, int max_depth) noexcept nogil

cdef void prune_internal(Tree* tree) noexcept nogil

cdef void mutate_split_feature(Tree* tree, int n_features, SplitValues* split_values) noexcept nogil

cdef void mutate_split_value(Tree* tree, SplitValues* split_values) noexcept nogil
