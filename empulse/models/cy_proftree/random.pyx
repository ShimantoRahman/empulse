from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

cdef inline int rand_int(int low, int high) noexcept nogil:
    """Return a random integer in [low, high)."""
    return low + rand() % (high - low)

cdef inline bint rand_bool() noexcept nogil:
    """Return a random boolean value."""
    return rand() % 2 == 0

cdef inline float rand_fraction() noexcept nogil:
    return rand() / (<float>RAND_MAX + 1.0)

cdef inline void set_seed(int random_state) noexcept nogil:
    if random_state != -1:
        srand(<unsigned int> random_state)
    else:
        srand(<unsigned int> time(NULL))