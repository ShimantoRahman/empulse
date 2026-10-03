from libcpp.vector cimport vector


cdef struct Point:
    long long n_negative  # samples ranked at or above a threshold that are negative
    long long n_positive  # ... and positive


cdef struct Group:
    double score
    long long n_positive
    long long n_negative


cdef void _add_groups_to_hull(vector[Group]& groups, vector[Point]& hull) noexcept nogil
