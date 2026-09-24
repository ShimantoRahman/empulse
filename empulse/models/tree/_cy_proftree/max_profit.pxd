from .node cimport Node

cdef float max_profit_score(
    Node* root,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
) noexcept nogil
