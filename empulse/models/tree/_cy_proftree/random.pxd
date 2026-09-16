cdef struct RandState:
    unsigned long long state
    unsigned long long inc

cdef unsigned int rand_uint32(RandState* rng) noexcept nogil
cdef void seed_rand(RandState* rng, unsigned int seed) noexcept nogil
cdef int rand_int(RandState* rng, int low, int high) noexcept nogil
cdef bint rand_bool(RandState* rng) noexcept nogil
cdef float rand_fraction(RandState* rng) noexcept nogil
