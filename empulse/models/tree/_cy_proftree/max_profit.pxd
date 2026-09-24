from libcpp.vector cimport vector

from .node cimport Node

cdef struct Leaf:
    float score  # the probability the leaf predicts
    int n_positive
    int n_negative

cdef void collect_leaves(Node* node, vector[Leaf]& leaves) noexcept nogil

cdef float max_profit_score(
    Node* root,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
) noexcept nogil
