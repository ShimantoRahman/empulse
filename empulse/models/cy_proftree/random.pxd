cdef int rand_int(int low, int high) noexcept nogil
cdef bint rand_bool() noexcept nogil
cdef float rand_fraction() noexcept nogil
cdef void set_seed(int random_state) noexcept nogil
