from ._criterion cimport ClassSums, CriterionKind
from ._utils cimport float32_t, float64_t, intp_t, uint8_t, uint32_t


cdef struct SplitRecord:
    intp_t feature          # which feature to split on
    intp_t pos              # samples[start:pos] go left, samples[pos:end] go right
    float64_t threshold     # samples with a value <= threshold go left
    bint missing_go_to_left # samples missing the feature (NaN) go left
    float64_t improvement   # the weighted impurity decrease of the split
    float64_t impurity_left
    float64_t impurity_right


cdef class Splitter:
    # Data, shared read-only between all trees grown on it
    cdef object _X_ref, _cost_ref, _weight_ref, _missing_ref
    cdef const float32_t* X          # Fortran-ordered, so a feature's values are contiguous
    cdef intp_t n_rows
    cdef const float64_t* cost       # one record [a, b, y, 0] per row (see _criterion.pxd)
    cdef const float64_t* weight     # one weight per row, or NULL when every row weighs 1
    cdef const uint8_t* missing_mask # per feature, whether any row misses it; NULL when none does

    # Working buffers, owned by this tree
    cdef intp_t[::1] samples         # the rows with a nonzero weight, partitioned node by node
    cdef float32_t[::1] feature_values
    cdef intp_t[::1] features
    cdef intp_t[::1] constant_features
    cdef uint32_t[::1] sort_keys            # radix sort buffers, unused by random splits
    cdef uint32_t[::1] sort_keys_buffer
    cdef intp_t[::1] sort_samples_buffer

    # Parameters
    cdef CriterionKind kind
    cdef bint random                 # draw one random threshold per feature instead of trying all
    cdef bint track_oracle           # sum the per-sample cheapest costs, for the cost bound
    cdef intp_t max_features
    cdef intp_t min_samples_leaf
    cdef float64_t min_weight_leaf
    cdef uint32_t rand_r_state

    cdef readonly intp_t n_samples
    cdef readonly intp_t n_features
    cdef readonly float64_t weighted_n_samples

    # The node being split
    cdef intp_t start, end, pos
    cdef ClassSums total
    # What _update moves samples from when counting from the far end: end and total, except while
    # a feature with missing values is searched, when the missing samples are left out of both.
    cdef intp_t scan_end
    cdef ClassSums scan_total
    cdef ClassSums left
    cdef float64_t oracle            # sum of w_i * min(a_i, b_i) over the node

    cdef void node_reset(self, intp_t start, intp_t end, float64_t* weighted_n_node_samples) noexcept nogil
    cdef float64_t node_impurity(self) noexcept nogil
    cdef float64_t max_cost_decrease(self) noexcept nogil
    cdef int node_split(self, float64_t impurity, SplitRecord* split, intp_t* n_constant_features) except -1 nogil
