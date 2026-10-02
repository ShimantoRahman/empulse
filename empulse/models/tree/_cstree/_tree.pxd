from ._criterion cimport ClassSums
from ._utils cimport float64_t, int32_t, intp_t, uint8_t


# What traversing the tree reads, packed together so a path down the tree touches as few cache lines
# as possible. Everything else about a node lives in separate arrays.
cdef struct Node:
    intp_t left_child           # -1 for a leaf
    intp_t right_child
    int32_t feature             # -2 for a leaf
    uint8_t missing_go_to_left  # where a missing (NaN) value goes
    float64_t threshold         # samples with a value <= threshold go left; -2 for a leaf


cdef class CostTree:
    cdef readonly intp_t n_features
    cdef readonly intp_t node_count
    cdef readonly intp_t max_depth
    cdef intp_t capacity

    cdef Node* nodes
    cdef float64_t* impurity_
    cdef intp_t* n_node_samples_
    cdef float64_t* weighted_n_node_samples_
    cdef float64_t* value_              # 2 per node: the weight fraction of each class
    cdef uint8_t* positive_             # whether predicting the node positive costs least

    cdef int _resize(self, intp_t capacity) except -1 nogil
    cdef intp_t _add_node(
        self,
        intp_t parent,
        bint is_left,
        bint is_leaf,
        intp_t feature,
        float64_t threshold,
        bint missing_go_to_left,
        float64_t impurity,
        intp_t n_node_samples,
        float64_t weighted_n_node_samples,
        const ClassSums* sums,
    ) except -1 nogil
