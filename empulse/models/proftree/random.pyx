from libc.stdlib cimport rand, RAND_MAX

cdef int rand_int(int low, int high) noexcept nogil:
    """Return a random integer in [low, high)."""
    return low + rand() % (high - low)

cdef bint rand_bool() noexcept nogil:
    """Return a random boolean value."""
    return rand() % 2 == 0

cdef float rand_fraction() noexcept nogil:
    return rand() / (<float>RAND_MAX + 1.0)
