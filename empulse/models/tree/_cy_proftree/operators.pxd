from .tree cimport Tree, SplitValues
from .node cimport Node
from .random cimport RandState

cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil

cdef int count_nodes(Node* node) noexcept nogil

cdef Tree* crossover(RandState* rng, Tree* mother, Tree* father, int max_depth) noexcept nogil

cdef void grow(RandState* rng, Tree* tree, SplitValues* split_values, int n_features, int max_depth) noexcept nogil

cdef void prune_internal(RandState* rng, Tree* tree) noexcept nogil

cdef void mutate_split_feature(RandState* rng, Tree* tree, int n_features, SplitValues* split_values) noexcept nogil

cdef void mutate_split_value(RandState* rng, Tree* tree, SplitValues* split_values) noexcept nogil
