# The cost-sensitive split criteria, as inline functions over running sums of a node's samples.
#
# Every sample i carries two costs: a_i, what predicting it positive costs (its tp_cost if it is a
# positive, its fp_cost if a negative), and b_i, what predicting it negative costs (fn_cost or
# tn_cost). A node's cost impurity is the cost of its cheapest single decision per unit of weight,
# min(sum w_i a_i, sum w_i b_i) / sum w_i. The gini and entropy criteria weight the two sides by a
# class-purity term (Correa Bahnsen et al., 2015).
#
# The sums are kept per class and added in sample order.

from libc.math cimport log as ln

from ._utils cimport float64_t, intp_t

cdef enum CriterionKind:
    COST = 0
    GINI = 1
    ENTROPY = 2


cdef struct ClassSums:
    float64_t w       # total weight, summed in sample order
    float64_t cw[2]   # weight of each class (index 0 negative, 1 positive)
    float64_t cp[2]   # weighted cost of predicting positive, per class
    float64_t cn[2]   # weighted cost of predicting negative, per class


cdef inline void sums_clear(ClassSums* s) noexcept nogil:
    s.w = 0.0
    s.cw[0] = 0.0
    s.cw[1] = 0.0
    s.cp[0] = 0.0
    s.cp[1] = 0.0
    s.cn[0] = 0.0
    s.cn[1] = 0.0


cdef inline void sums_add(ClassSums* s, const float64_t* cost, float64_t w) noexcept nogil:
    """
    Add one sample: ``cost`` points at its record ``[a, b, y, _]``.

    The class is selected arithmetically rather than by indexing: with y either 0.0 or 1.0, each
    product below is exactly the term or exactly zero, so the sums are those of adding the sample to
    its own class alone. Constant indices let the compiler keep the sums in registers.
    """
    cdef float64_t y = cost[2]
    cdef float64_t wy = w * y
    cdef float64_t wa = w * cost[0]
    cdef float64_t wb = w * cost[1]
    cdef float64_t way = wa * y
    cdef float64_t wby = wb * y
    s.cw[0] += w - wy
    s.cw[1] += wy
    s.cp[0] += wa - way
    s.cp[1] += way
    s.cn[0] += wb - wby
    s.cn[1] += wby
    s.w += w


cdef inline void sums_remove(ClassSums* s, const float64_t* cost, float64_t w) noexcept nogil:
    cdef float64_t y = cost[2]
    cdef float64_t wy = w * y
    cdef float64_t wa = w * cost[0]
    cdef float64_t wb = w * cost[1]
    cdef float64_t way = wa * y
    cdef float64_t wby = wb * y
    s.cw[0] -= w - wy
    s.cw[1] -= wy
    s.cp[0] -= wa - way
    s.cp[1] -= way
    s.cn[0] -= wb - wby
    s.cn[1] -= wby
    s.w -= w


cdef inline void sums_subtract(ClassSums* out, const ClassSums* total, const ClassSums* part) noexcept nogil:
    """``out = total - part``, class by class."""
    out.w = total.w - part.w
    out.cw[0] = total.cw[0] - part.cw[0]
    out.cw[1] = total.cw[1] - part.cw[1]
    out.cp[0] = total.cp[0] - part.cp[0]
    out.cp[1] = total.cp[1] - part.cp[1]
    out.cn[0] = total.cn[0] - part.cn[0]
    out.cn[1] = total.cn[1] - part.cn[1]


cdef inline float64_t _log2(float64_t x) noexcept nogil:
    return ln(x) / ln(2.0)


cdef extern from *:
    """
    #if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    #include <emmintrin.h>
    static __inline double empulse_fast_fmin(double a, double b) {
        return _mm_cvtsd_f64(_mm_min_sd(_mm_set_sd(a), _mm_set_sd(b)));
    }
    #else
    static inline double empulse_fast_fmin(double a, double b) { return a < b ? a : b; }
    #endif
    """
    # ``a if a < b else b``. MSVC calls the CRT for ``fmin`` and branches on the ternary, which
    # mispredicts whenever the cheaper decision varies; minsd computes it without a branch.
    float64_t fast_fmin "empulse_fast_fmin"(float64_t a, float64_t b) noexcept nogil


cdef inline float64_t decision_cost(float64_t cp0, float64_t cp1, float64_t cn0, float64_t cn1) noexcept nogil:
    """The weighted cost of a node's cheapest single decision: its cost impurity times its weight."""
    cdef float64_t pos_cost = 0.0
    cdef float64_t neg_cost = 0.0
    pos_cost += cp0
    pos_cost += cp1
    neg_cost += cn0
    neg_cost += cn1
    return fast_fmin(pos_cost, neg_cost)


cdef inline float64_t impurity_of(
    CriterionKind kind,
    float64_t w,
    float64_t neg_count,
    float64_t pos_count,
    float64_t cp0,
    float64_t cp1,
    float64_t cn0,
    float64_t cn1,
) noexcept nogil:
    """The impurity of a node with total weight ``w``, from its sums as ``ClassSums`` holds them."""
    cdef float64_t pos_cost = 0.0
    cdef float64_t neg_cost = 0.0
    cdef float64_t pos_entropy, neg_entropy
    if kind == COST:
        return decision_cost(cp0, cp1, cn0, cn1) / w
    pos_cost += cp0
    pos_cost += cp1
    neg_cost += cn0
    neg_cost += cn1
    if kind == GINI:
        return fast_fmin(pos_cost * (pos_count * pos_count / w), neg_cost * (neg_count * neg_count / w)) / w
    # ENTROPY
    pos_cost /= w
    neg_cost /= w
    pos_entropy = _log2(pos_count / w) if pos_count > 0.0 else 0.0
    neg_entropy = _log2(neg_count / w) if neg_count > 0.0 else 0.0
    return fast_fmin(pos_cost * -pos_entropy, neg_cost * -neg_entropy)


cdef inline float64_t impurity(CriterionKind kind, const ClassSums* s, float64_t w) noexcept nogil:
    """The impurity of a node with sums ``s`` and total weight ``w``."""
    return impurity_of(kind, w, s.cw[0], s.cw[1], s.cp[0], s.cp[1], s.cn[0], s.cn[1])


cdef inline float64_t proxy_improvement(CriterionKind kind, const ClassSums* left, const ClassSums* right) noexcept nogil:
    """
    A quantity that orders candidate splits like their impurity improvement does.

    It is computed as scikit-learn computes it, ``-w_R * imp_R - w_L * imp_L``, rather than with the
    weights folded in algebraically. Where no split lowers the impurity, every candidate ties but for
    rounding, and rounding is then what spreads the chosen thresholds across the node; a formula
    under which they tie exactly makes the first candidate win everywhere, splitting one sample off
    at a time. Breaking such ties by balance instead keeps trees shallow, but moves the decision
    boundaries enough to cost a cost-sensitive forest much of its savings.
    """
    return -right.w * impurity(kind, right, right.w) - left.w * impurity(kind, left, left.w)


cdef inline float64_t impurity_improvement(
    float64_t weighted_n_samples,
    float64_t weighted_n_node_samples,
    float64_t weighted_n_left,
    float64_t weighted_n_right,
    float64_t impurity_parent,
    float64_t impurity_left,
    float64_t impurity_right,
) noexcept nogil:
    """The weighted impurity decrease of a split, as ``min_impurity_decrease`` measures it."""
    return ((weighted_n_node_samples / weighted_n_samples) *
            (impurity_parent - (weighted_n_right / weighted_n_node_samples * impurity_right)
                             - (weighted_n_left / weighted_n_node_samples * impurity_left)))
