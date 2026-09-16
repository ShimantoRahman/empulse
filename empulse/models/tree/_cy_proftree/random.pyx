# distutils: language = c++

# PCG32 (https://www.pcg-random.org): a small, fast, statistically sound generator whose entire
# state is two 64-bit words, so it costs nothing to keep one per fit instead of one per process.

cdef inline unsigned int rand_uint32(RandState* rng) noexcept nogil:
    """Advance the state and return the next 32-bit output."""
    cdef unsigned long long oldstate = rng.state
    rng.state = oldstate * <unsigned long long>6364136223846793005 + rng.inc
    cdef unsigned int xorshifted = <unsigned int>(((oldstate >> 18) ^ oldstate) >> 27)
    cdef unsigned int rot = <unsigned int>(oldstate >> 59)
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))

cdef inline void seed_rand(RandState* rng, unsigned int seed) noexcept nogil:
    """Seed the state. Distinct seeds select distinct streams, not just distinct offsets."""
    rng.state = 0
    rng.inc = (<unsigned long long>seed << 1) | 1
    rand_uint32(rng)
    rng.state = rng.state + <unsigned long long>seed
    rand_uint32(rng)

cdef inline int rand_int(RandState* rng, int low, int high) noexcept nogil:
    """Return a random integer in [low, high)."""
    return low + <int>(rand_uint32(rng) % <unsigned int>(high - low))

cdef inline bint rand_bool(RandState* rng) noexcept nogil:
    """Return a random boolean value."""
    return (rand_uint32(rng) & 1) == 0

cdef inline float rand_fraction(RandState* rng) noexcept nogil:
    """Return a random float in [0, 1)."""
    return <float>(rand_uint32(rng) * (1.0 / 4294967296.0))
