# distutils: language = c++

from libcpp.algorithm cimport sort as cpp_sort
from libcpp.vector cimport vector

from .node cimport Node, is_leaf


cdef struct Leaf:
    float score  # the probability the leaf predicts
    int n_positive
    int n_negative


cdef inline bint _ranks_higher(const Leaf& a, const Leaf& b) noexcept nogil:
    return a.score > b.score


cdef void _collect_leaves(Node* node, vector[Leaf]& leaves) noexcept nogil:
    if node is NULL:
        return
    if not is_leaf(node):
        _collect_leaves(node.left, leaves)
        _collect_leaves(node.right, leaves)
    elif node.n_samples > 0:  # a leaf no sample reaches adds no point to the ROC curve
        leaves.push_back(Leaf(
            <float>node.n_positive_samples / <float>node.n_samples,
            node.n_positive_samples,
            node.n_samples - node.n_positive_samples,
        ))


cdef inline float _profit(
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
    float pi0,
    float pi1,
    float true_positive_rate,
    float false_positive_rate,
) noexcept nogil:
    return (
        (tp_benefit + fn_cost) * pi0 * true_positive_rate
        - (tn_benefit + fp_cost) * pi1 * false_positive_rate
        + tn_benefit * pi1
        - fn_cost * pi0
    )


cdef float max_profit_score(
    Node* root,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
) noexcept nogil:
    """
    Maximum profit of a fitted tree on its training samples, from its leaf counts alone.

    Every sample in a leaf gets the same score, so the ROC curve of the tree's predictions has one
    point per distinct leaf score: ranking the leaves by score and accumulating their positive and
    negative counts gives exactly the curve that ranking every sample would. The profit is linear in
    the true and false positive rates, so its maximum over the curve's convex hull is attained at a
    point of the curve itself, and it suffices to evaluate each point.

    This costs O(n_leaves log n_leaves) rather than predicting and sorting all n_samples samples.
    """
    cdef vector[Leaf] leaves
    _collect_leaves(root, leaves)
    cpp_sort(leaves.begin(), leaves.end(), _ranks_higher)

    cdef long n_positive = 0
    cdef long n_negative = 0
    cdef size_t i
    for i in range(leaves.size()):
        n_positive += leaves[i].n_positive
        n_negative += leaves[i].n_negative
    cdef float pi0 = <float>n_positive / <float>(n_positive + n_negative)
    cdef float pi1 = 1.0 - pi0

    # Predicting every sample negative is the (0, 0) point.
    cdef float maximum_profit = _profit(tp_benefit, tn_benefit, fp_cost, fn_cost, pi0, pi1, 0.0, 0.0)
    cdef float profit
    cdef long true_positives = 0
    cdef long false_positives = 0
    for i in range(leaves.size()):
        true_positives += leaves[i].n_positive
        false_positives += leaves[i].n_negative
        # Leaves with equal scores cannot be separated by a threshold: only the end of a tie is a point.
        if i + 1 < leaves.size() and leaves[i + 1].score == leaves[i].score:
            continue
        profit = _profit(
            tp_benefit, tn_benefit, fp_cost, fn_cost, pi0, pi1,
            <float>true_positives / <float>n_positive,
            <float>(<double>false_positives / <double>n_negative),
        )
        if profit > maximum_profit:
            maximum_profit = profit

    return maximum_profit
