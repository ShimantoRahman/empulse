from .forest cimport Forest
from .tree cimport Tree, SplitValues
from .node cimport Node

cdef Tree* tournament_selection(Forest* population, int tournament_size) noexcept

cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil

cdef Tree* crossover(Tree* mother, Tree* father, int max_depth) noexcept nogil

cdef Tree* evolve(
    Tree* old_tree,
    Forest* population,
    SplitValues* split_values,
    int n_features,
    int max_depth,
    int tournament_size,
)

cdef void mutate_split_feature(
    Node* node,
    int n_features,
    SplitValues* split_values,
) noexcept nogil

cdef void mutate_split_value(Node* node, SplitValues* split_values) noexcept nogil
