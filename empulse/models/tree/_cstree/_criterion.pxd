# The cost-sensitive split criteria, as inline functions over running sums of a node's samples.
#
# Every sample i carries two costs: a_i, what predicting it positive costs (its tp_cost if it is a
# positive, its fp_cost if a negative), and b_i, what predicting it negative costs (fn_cost or
# tn_cost). A node's cost impurity is the cost of its cheapest single decision per unit of weight,
# min(sum w_i a_i, sum w_i b_i) / sum w_i. The gini and entropy criteria weight the two sides by a
# class-purity term (Correa Bahnsen et al., 2015).
#
# The sums are kept per class and added in sample order.

from libc.math cimport fmin, log as ln

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
    """Add one sample: ``cost`` points at its record ``[a, b, y, _]``."""
    cdef intp_t c = <intp_t> cost[2]
    s.cw[c] += w
    s.cp[c] += w * cost[0]
    s.cn[c] += w * cost[1]
    s.w += w


cdef inline void sums_remove(ClassSums* s, const float64_t* cost, float64_t w) noexcept nogil:
    cdef intp_t c = <intp_t> cost[2]
    s.cw[c] -= w
    s.cp[c] -= w * cost[0]
    s.cn[c] -= w * cost[1]
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


cdef inline float64_t impurity(CriterionKind kind, const ClassSums* s, float64_t w) noexcept nogil:
    """The impurity of a node with sums ``s`` and total weight ``w``."""
    cdef float64_t pos_cost = 0.0
    cdef float64_t neg_cost = 0.0
    cdef float64_t pos_count, neg_count, pos_entropy, neg_entropy
    pos_cost += s.cp[0]
    pos_cost += s.cp[1]
    neg_cost += s.cn[0]
    neg_cost += s.cn[1]
    if kind == COST:
        return fmin(pos_cost, neg_cost) / w
    pos_count = s.cw[1]
    neg_count = s.cw[0]
    if kind == GINI:
        return fmin(pos_cost * (pos_count * pos_count / w), neg_cost * (neg_count * neg_count / w)) / w
    # ENTROPY
    pos_cost /= w
    neg_cost /= w
    pos_entropy = _log2(pos_count / w) if pos_count > 0.0 else 0.0
    neg_entropy = _log2(neg_count / w) if neg_count > 0.0 else 0.0
    return fmin(pos_cost * -pos_entropy, neg_cost * -neg_entropy)


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
