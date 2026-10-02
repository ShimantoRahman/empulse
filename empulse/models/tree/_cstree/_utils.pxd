# Typedefs, the random number generator and the sort used by the cost-sensitive tree builder.
#
# The random number generator and the introsort are ported from scikit-learn
# (sklearn/utils/_random.pxd, sklearn/tree/_utils.pyx and sklearn/utils/_sorting.pyx),
# Copyright (c) the scikit-learn developers, BSD-3-Clause license. They are kept identical so a
# tree grown from a given seed draws the same features in the same order as scikit-learn's.
#
# Everything here is inline, so this file needs no extension module of its own.

from libc.math cimport log2
from libc.string cimport memcpy, memset

ctypedef float float32_t
ctypedef double float64_t
ctypedef Py_ssize_t intp_t
ctypedef signed int int32_t
ctypedef unsigned int uint32_t
ctypedef unsigned long long uint64_t
ctypedef unsigned char uint8_t

cdef enum:
    # The largest value rand_r returns, 2^31 - 1, the same on every platform (RAND_MAX is not).
    RAND_R_MAX = 2147483647

cdef inline uint32_t rand_r(uint32_t* seed) noexcept nogil:
    """Return a pseudo-random uint32 in [0, RAND_R_MAX] and advance ``seed`` (xorshift)."""
    if seed[0] == 0:
        seed[0] = 1
    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)
    return seed[0] % ((<uint32_t>RAND_R_MAX) + 1)


cdef inline intp_t rand_int(intp_t low, intp_t high, uint32_t* random_state) noexcept nogil:
    """Return a random integer in [low, high)."""
    return low + rand_r(random_state) % (high - low)


cdef inline float64_t rand_uniform(float64_t low, float64_t high, uint32_t* random_state) noexcept nogil:
    """Return a random float in [low, high)."""
    return ((high - low) * <float64_t> rand_r(random_state) / <float64_t> RAND_R_MAX) + low


# ------------------------------------------------------------------------------------------------
# Introsort of feature values, carrying the sample indices along
# ------------------------------------------------------------------------------------------------

cdef inline void _swap(float32_t* values, intp_t* indices, intp_t i, intp_t j) noexcept nogil:
    values[i], values[j] = values[j], values[i]
    indices[i], indices[j] = indices[j], indices[i]


cdef inline float32_t _median3(float32_t* values, intp_t n) noexcept nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    cdef float32_t a = values[0], b = values[n // 2], c = values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


cdef inline void _insertion_sort(float32_t* values, intp_t* indices, intp_t n) noexcept nogil:
    cdef intp_t i, j, temp_idx
    cdef float32_t temp_val
    for i in range(1, n):
        temp_val = values[i]
        temp_idx = indices[i]
        j = i
        while j > 0 and values[j - 1] > temp_val:
            values[j] = values[j - 1]
            indices[j] = indices[j - 1]
            j -= 1
        values[j] = temp_val
        indices[j] = temp_idx


cdef inline void _sift_down(float32_t* values, intp_t* indices, intp_t start, intp_t end) noexcept nogil:
    # Restore heap order in values[start:end] by moving the max element to start.
    cdef intp_t child, maxind, root
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and values[maxind] < values[child]:
            maxind = child
        if child + 1 < end and values[maxind] < values[child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        _swap(values, indices, root, maxind)
        root = maxind


cdef inline void _heapsort(float32_t* values, intp_t* indices, intp_t n) noexcept nogil:
    cdef intp_t start, end
    start = (n - 2) // 2
    end = n
    while True:
        _sift_down(values, indices, start, end)
        if start == 0:
            break
        start -= 1
    end = n - 1
    while end > 0:
        _swap(values, indices, 0, end)
        _sift_down(values, indices, 0, end)
        end = end - 1


cdef inline void _introsort(float32_t* values, intp_t* indices, intp_t n, intp_t maxd) noexcept nogil:
    # Introsort with median of 3 pivot selection and a 3-way partition, which is fast when there
    # are many repeated values, as there are in most tabular features.
    cdef float32_t pivot
    cdef intp_t i, l, r
    while n > 15:
        if maxd <= 0:  # gone quadratic
            _heapsort(values, indices, n)
            return
        maxd -= 1
        pivot = _median3(values, n)
        i = l = 0
        r = n
        while i < r:
            if values[i] < pivot:
                _swap(values, indices, i, l)
                i += 1
                l += 1
            elif values[i] > pivot:
                r -= 1
                _swap(values, indices, i, r)
            else:
                i += 1
        # values[:l] < pivot, values[l:r] == pivot, values[r:] > pivot
        _introsort(values, indices, l, maxd)
        values += r
        indices += r
        n -= r
    _insertion_sort(values, indices, n)


cdef inline void sort(float32_t* values, intp_t* indices, intp_t n) noexcept nogil:
    """Sort ``values`` ascending, applying the same permutation to ``indices``."""
    if n == 0:
        return
    _introsort(values, indices, n, 2 * <intp_t>log2(n))


# ------------------------------------------------------------------------------------------------
# Radix sort, for large nodes
# ------------------------------------------------------------------------------------------------

cdef inline uint32_t float_to_key(float32_t value) noexcept nogil:
    """Map a float to an unsigned integer with the same order (for every non-NaN value)."""
    cdef uint32_t bits
    memcpy(&bits, &value, 4)
    # Flip every bit of a negative value and only the sign bit of the others, without a branch on
    # the sign (which mispredicts on centred features).
    return bits ^ ((0u - (bits >> 31)) | 0x80000000u)


cdef inline float32_t key_to_float(uint32_t key) noexcept nogil:
    """Invert ``float_to_key``."""
    cdef uint32_t bits = key ^ (((key >> 31) - 1u) | 0x80000000u)
    cdef float32_t value
    memcpy(&value, &bits, 4)
    return value


cdef inline void radix_sort(uint64_t* items, uint64_t* buffer, intp_t n) noexcept nogil:
    """
    Sort ``items`` ascending by their upper 32 bits, keeping items with equal keys in their order.

    Each item packs its key above a 32-bit payload, so a pass moves a single word per item. A stable
    least-significant-digit radix sort in three passes of 11 bits, linear in ``n``; a pass is skipped
    when every key shares its digit. ``buffer`` holds ``n`` items.
    """
    cdef uint32_t counts[3][2048]
    cdef uint32_t* count
    cdef uint64_t* source = items
    cdef uint64_t* target = buffer
    cdef uint64_t* swap
    cdef uint64_t item
    cdef uint32_t key, total, digit_count, digit
    cdef intp_t i, d, shift
    memset(counts, 0, sizeof(counts))
    for i in range(n):
        key = <uint32_t> (items[i] >> 32)
        counts[0][key & 0x7FF] += 1
        counts[1][(key >> 11) & 0x7FF] += 1
        counts[2][key >> 22] += 1
    for d in range(3):
        shift = 32 + 11 * d
        count = counts[d]
        if count[(items[0] >> shift) & 0x7FF] == <uint32_t> n:
            continue
        total = 0
        for i in range(2048):
            digit_count = count[i]
            count[i] = total
            total += digit_count
        for i in range(n):
            item = source[i]
            digit = (item >> shift) & 0x7FF
            target[count[digit]] = item
            count[digit] += 1
        swap = source
        source = target
        target = swap
    if source != items:
        memcpy(items, source, n * sizeof(uint64_t))
