# distutils: language = c++
"""
Find the best split of a node of a cost-sensitive decision tree.

The split search is ported from scikit-learn's ``node_split_best`` and ``node_split_random``
(sklearn/tree/_splitter.pyx and sklearn/tree/_partitioner.pyx), Copyright (c) the scikit-learn
developers, BSD-3-Clause license. It keeps their feature sampling, constant-feature bookkeeping,
tie handling and treatment of missing values: a split sends the samples missing its feature (NaN)
to whichever child scores best, and when the node has none, to the child with more samples. Sparse
input and monotonic constraints are not supported. The criterion's running sums are inline (see
``_criterion.pxd``), and every sample's costs are read from a single record.
"""

from libc.math cimport INFINITY, isnan
from libc.string cimport memcpy

import numpy as np

from ._criterion cimport (
    COST, CriterionKind, ClassSums, decision_cost, fast_fmin, impurity, impurity_of, impurity_improvement,
    proxy_improvement, sums_add, sums_clear, sums_remove, sums_subtract,
)
from ._utils cimport float_to_key, key_to_float, radix_sort, rand_int, rand_uniform, sort

# Two feature values closer than this count as equal: a split between them is never considered.
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Nodes at least this large are sorted by radix sort, smaller ones by introsort.
cdef intp_t RADIX_SORT_MIN_SAMPLES = 512

# The relative rounding error of a sum, per term, assumed by max_cost_decrease. Generous: the bound it
# guards may only ever overestimate.
cdef float64_t ROUNDING = 4.0 * np.finfo(np.float64).eps

# How far, relatively, the cost criterion's split proxy may lie from minus the sum of its children's
# decision costs. Three roundings separate them; the margin is far wider, so that it can only ever
# let through a candidate that loses, never turn one away that wins.
cdef float64_t PROXY_SLACK = 1e-13


cdef inline void _init_split(SplitRecord* split, intp_t start_pos) noexcept nogil:
    split.impurity_left = INFINITY
    split.impurity_right = INFINITY
    split.pos = start_pos
    split.feature = 0
    split.threshold = 0.0
    split.missing_go_to_left = False
    split.improvement = -INFINITY


cdef class Splitter:
    """
    Split search for one tree.

    Parameters
    ----------
    X : ndarray of shape (n_rows, n_features), dtype float32, Fortran-ordered
        The training samples.
    cost : ndarray of shape (n_rows, 4), dtype float64, C-ordered
        Per row: the cost of predicting it positive, of predicting it negative, its class (0.0 or
        1.0), and padding.
    weight : ndarray of shape (n_rows,), dtype float64, or None
        The weight of each row. Rows weighing zero take no part in the tree.
    criterion : int
        A ``CriterionKind``.
    random : bool
        Draw one random threshold per feature (``splitter="random"``) instead of trying them all.
    max_features, min_samples_leaf, min_weight_leaf
        As resolved by the estimator.
    seed : int
        Seed of the feature and threshold draws.
    track_oracle : bool
        Sum the cheapest cost of every sample of a node, which ``max_cost_decrease`` needs.
    missing_mask : ndarray of shape (n_features,), dtype uint8, or None
        Whether any row misses each feature (is NaN). ``None`` when no row misses any.
    """

    def __cinit__(
        self,
        object X,
        object cost,
        object weight,
        int criterion,
        bint random,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        uint32_t seed,
        bint track_oracle,
        object missing_mask=None,
    ):
        cdef const float32_t[::1, :] X_view = X
        cdef const float64_t[:, ::1] cost_view = cost
        cdef const float64_t[::1] weight_view
        cdef const uint8_t[::1] missing_view
        if cost_view.shape[0] != X_view.shape[0] or cost_view.shape[1] != 4:
            raise ValueError('cost must have shape (n_rows, 4)')

        self._X_ref = X
        self._cost_ref = cost
        self._weight_ref = weight
        self.n_rows = X_view.shape[0]
        self.n_features = X_view.shape[1]
        self.X = &X_view[0, 0]
        self.cost = &cost_view[0, 0]

        self._missing_ref = missing_mask
        if missing_mask is None:
            self.missing_mask = NULL
        else:
            missing_view = missing_mask
            if missing_view.shape[0] != self.n_features:
                raise ValueError('missing_mask must have shape (n_features,)')
            self.missing_mask = &missing_view[0]

        if weight is None:
            self.weight = NULL
            self.samples = np.arange(self.n_rows, dtype=np.intp)
            self.weighted_n_samples = <float64_t> self.n_rows
        else:
            weight_view = weight
            if weight_view.shape[0] != self.n_rows:
                raise ValueError('weight must have shape (n_rows,)')
            self.weight = &weight_view[0]
            self.samples = np.flatnonzero(weight).astype(np.intp, copy=False)
            self.weighted_n_samples = _sum_in_order(weight_view)
        self.n_samples = self.samples.shape[0]

        self.feature_values = np.empty(self.n_samples, dtype=np.float32)
        self.features = np.arange(self.n_features, dtype=np.intp)
        self.constant_features = np.empty(self.n_features, dtype=np.intp)
        if not random:
            self.sort_items = np.empty(self.n_samples, dtype=np.uint64)
            self.sort_items_buffer = np.empty(self.n_samples, dtype=np.uint64)
        # The radix sort carries row indices in 32 bits.
        self.radix_sort_min_samples = RADIX_SORT_MIN_SAMPLES if self.n_rows <= 0xFFFFFFFF else self.n_rows + 1

        self.kind = <CriterionKind> criterion
        self.random = random
        self.track_oracle = track_oracle
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.rand_r_state = seed

    cdef void node_reset(self, intp_t start, intp_t end, float64_t* weighted_n_node_samples) noexcept nogil:
        """Take samples[start:end] as the node to split, and sum its costs."""
        cdef intp_t p, i
        cdef float64_t w
        cdef const float64_t* record
        cdef const intp_t* samples = &self.samples[0]
        cdef const float64_t* cost = self.cost
        cdef const float64_t* weight = self.weight
        cdef float64_t oracle = 0.0
        cdef ClassSums total
        sums_clear(&total)
        for p in range(start, end):
            i = samples[p]
            w = weight[i] if weight != NULL else 1.0
            record = cost + 4 * i
            sums_add(&total, record, w)
            if self.track_oracle:
                oracle += w * fast_fmin(record[0], record[1])
        self.start = start
        self.end = end
        self.total = total
        self.scan_end = end
        self.scan_total = total
        self.oracle = oracle
        _reset(self)
        weighted_n_node_samples[0] = total.w

    cdef float64_t node_impurity(self) noexcept nogil:
        return impurity(self.kind, &self.total, self.total.w)

    cdef float64_t max_cost_decrease(self) noexcept nogil:
        """
        An upper bound on how much any split of the node can lower its cost, per training weight.

        Splitting can at best decide every sample by its own cheapest cost, so no split lowers the
        node's cost, min(sum w a, sum w b), below the sum of w min(a, b). The bound is generous by
        the rounding the two sums can carry, so it never falls below what the split search would
        compute. Only meaningful for the cost criterion, with ``track_oracle``.
        """
        cdef float64_t pos_cost = self.total.cp[0] + self.total.cp[1]
        cdef float64_t neg_cost = self.total.cn[0] + self.total.cn[1]
        cdef float64_t node_cost = fast_fmin(pos_cost, neg_cost)
        cdef float64_t slack = ROUNDING * (self.end - self.start + 1) * (
            abs(pos_cost) + abs(neg_cost) + abs(self.oracle)
        )
        return (node_cost - self.oracle + slack) / self.weighted_n_samples

    cdef int node_split(self, float64_t impurity, SplitRecord* split, intp_t* n_constant_features) except -1 nogil:
        """
        Find the best split of the node, and partition its samples by it.

        ``split.pos == end`` when no valid split exists. ``n_constant_features`` holds how many of the
        leading ``features`` are known to be constant in the node, and is updated with those found.
        """
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t* features = &self.features[0]
        cdef intp_t* constant_features = &self.constant_features[0]
        cdef intp_t* samples = &self.samples[0]
        cdef float32_t* feature_values = &self.feature_values[0]
        cdef intp_t n_features = self.n_features
        cdef intp_t max_features = self.max_features
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef uint32_t* random_state = &self.rand_r_state
        cdef CriterionKind kind = self.kind

        cdef SplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement
        cdef float64_t best_proxy_improvement = -INFINITY
        cdef ClassSums right
        cdef const float32_t* Xf
        cdef float32_t value, min_feature_value, max_feature_value
        cdef const uint8_t* missing_mask = self.missing_mask
        cdef intp_t n_missing
        cdef bint missing_go_to_left

        cdef intp_t f_i = n_features
        cdef intp_t f_j, p
        cdef intp_t n_visited_features = 0
        cdef intp_t n_found_constants = 0      # found constant during this search
        cdef intp_t n_drawn_constants = 0      # known constant and drawn without replacement
        cdef intp_t n_known_constants = n_constant_features[0]
        cdef intp_t n_total_constants = n_known_constants

        _init_split(&best_split, end)
        current_split.feature = 0

        # Draw up to max_features features without replacement (Fisher-Yates over `features`),
        # skipping features already known to be constant in this node. Loop invariant on `features`:
        # - [:n_drawn_constants] drawn and known constant;
        # - [n_drawn_constants:n_known_constants] known constant, not drawn yet;
        # - [n_known_constants:n_total_constants] newly found constant;
        # - [n_total_constants:f_i] not drawn yet, not known constant;
        # - [f_i:n_features] drawn and not constant.
        while (f_i > n_total_constants and
               (n_visited_features < max_features or
                # At least one drawn feature must be non constant
                n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
                continue

            f_j += n_found_constants
            current_split.feature = features[f_j]
            Xf = self.X + current_split.feature * self.n_rows

            n_missing = 0
            if not self.random:
                if missing_mask != NULL and missing_mask[current_split.feature]:
                    n_missing = _move_missing_to_end(self, Xf)
                if n_missing < end - start:
                    _sort_node(self, Xf, start, end - n_missing)
                    min_feature_value = feature_values[start]
                    max_feature_value = feature_values[end - n_missing - 1]
            elif missing_mask != NULL and missing_mask[current_split.feature]:
                n_missing = _gather_values(self, Xf, &min_feature_value, &max_feature_value)
            else:
                min_feature_value = Xf[samples[start]]
                max_feature_value = min_feature_value
                for p in range(start, end):
                    value = Xf[samples[p]]
                    feature_values[p] = value
                    if value < min_feature_value:
                        min_feature_value = value
                    elif value > max_feature_value:
                        max_feature_value = value

            # Constant: every value missing, or none missing and the values equal.
            if n_missing == end - start or (
                n_missing == 0 and max_feature_value <= min_feature_value + FEATURE_THRESHOLD
            ):
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]

            if not self.random and n_missing > 0:
                _search_with_missing(self, end - n_missing, &current_split, &best_split, &best_proxy_improvement)
            elif not self.random:
                _search_sorted(self, &current_split, &best_split, &best_proxy_improvement)
            else:
                # Try one random threshold, sending the missing values to a random side.
                current_split.threshold = rand_uniform(min_feature_value, max_feature_value, random_state)
                missing_go_to_left = n_missing > 0 and rand_int(0, 2, random_state)
                if current_split.threshold == max_feature_value:
                    current_split.threshold = min_feature_value
                current_split.pos = _partition_values(self, current_split.threshold, missing_go_to_left)
                if current_split.pos - start < min_samples_leaf or end - current_split.pos < min_samples_leaf:
                    continue

                _reset(self)
                _update(self, current_split.pos)
                sums_subtract(&right, &self.total, &self.left)
                if self.left.w < min_weight_leaf or right.w < min_weight_leaf:
                    continue

                current_proxy_improvement = proxy_improvement(kind, &self.left, &right)
                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    if n_missing > 0:
                        current_split.missing_go_to_left = missing_go_to_left
                    else:
                        current_split.missing_go_to_left = current_split.pos - start > end - current_split.pos
                    best_split = current_split

        if best_split.pos < end:
            # The random search left the samples partitioned by the last feature it drew.
            if not self.random or current_split.feature != best_split.feature:
                _partition(self, best_split.feature, best_split.threshold, best_split.missing_go_to_left)
            _finish_split(self, impurity, &best_split)

        # The leading known constant features must keep their order for sibling and child nodes.
        memcpy(features, constant_features, sizeof(intp_t) * n_known_constants)
        memcpy(constant_features + n_known_constants, features + n_known_constants,
               sizeof(intp_t) * n_found_constants)

        n_constant_features[0] = n_total_constants
        split[0] = best_split
        return 0

cdef inline void _reset(Splitter self) noexcept nogil:
    """Move every sample of the node to the right child."""
    self.pos = self.start
    sums_clear(&self.left)


cdef inline void _update(Splitter self, intp_t new_pos) noexcept nogil:
    """Move samples[pos:new_pos] to the left child, from whichever end is closer."""
    cdef intp_t p
    cdef const intp_t* samples = &self.samples[0]
    cdef const float64_t* cost = self.cost
    cdef const float64_t* weight = self.weight
    cdef ClassSums left
    if (new_pos - self.pos) <= (self.scan_end - new_pos):
        left = self.left
        if weight == NULL:
            for p in range(self.pos, new_pos):
                sums_add(&left, cost + 4 * samples[p], 1.0)
        else:
            for p in range(self.pos, new_pos):
                sums_add(&left, cost + 4 * samples[p], weight[samples[p]])
    else:
        left = self.scan_total
        if weight == NULL:
            for p in range(self.scan_end - 1, new_pos - 1, -1):
                sums_remove(&left, cost + 4 * samples[p], 1.0)
        else:
            for p in range(self.scan_end - 1, new_pos - 1, -1):
                sums_remove(&left, cost + 4 * samples[p], weight[samples[p]])
    self.left = left
    self.pos = new_pos


cdef inline void _sort_node(Splitter self, const float32_t* Xf, intp_t start, intp_t end) noexcept nogil:
    """Sort samples[start:end] by their value of feature ``Xf``, into feature_values[start:end]."""
    cdef intp_t p, i
    cdef intp_t* samples = &self.samples[0]
    cdef float32_t* feature_values = &self.feature_values[0]
    cdef uint64_t* items
    if end - start < self.radix_sort_min_samples:
        for p in range(start, end):
            feature_values[p] = Xf[samples[p]]
        sort(feature_values + start, samples + start, end - start)
    else:
        # Each item packs the sort key above the row index.
        items = &self.sort_items[0]
        for p in range(start, end):
            i = samples[p]
            items[p] = (<uint64_t> float_to_key(Xf[i]) << 32) | <uint64_t> i
        radix_sort(items + start, &self.sort_items_buffer[0], end - start)
        for p in range(start, end):
            samples[p] = <intp_t> (items[p] & 0xFFFFFFFFu)
            feature_values[p] = key_to_float(<uint32_t> (items[p] >> 32))


cdef inline void _partition(
    Splitter self, intp_t feature, float64_t threshold, bint missing_go_to_left
) noexcept nogil:
    """Reorder samples[start:end] so those going left by the given rule come first."""
    cdef intp_t partition_start = self.start
    cdef intp_t partition_end = self.end
    cdef intp_t* samples = &self.samples[0]
    cdef const float32_t* Xf = self.X + feature * self.n_rows
    # A NaN fails every comparison: `<=` sends it right, `not >` left.
    if missing_go_to_left:
        while partition_start < partition_end:
            if not Xf[samples[partition_start]] > threshold:
                partition_start += 1
            else:
                partition_end -= 1
                samples[partition_start], samples[partition_end] = samples[partition_end], samples[partition_start]
    else:
        while partition_start < partition_end:
            if Xf[samples[partition_start]] <= threshold:
                partition_start += 1
            else:
                partition_end -= 1
                samples[partition_start], samples[partition_end] = samples[partition_end], samples[partition_start]


cdef inline void _finish_split(Splitter self, float64_t impurity_parent, SplitRecord* split) noexcept nogil:
    """Fill in the impurities and improvement of a split whose samples are already partitioned."""
    cdef ClassSums right
    _reset(self)
    _update(self, split.pos)
    sums_subtract(&right, &self.total, &self.left)
    split.impurity_left = impurity(self.kind, &self.left, self.left.w)
    split.impurity_right = impurity(self.kind, &right, right.w)
    split.improvement = impurity_improvement(
        self.weighted_n_samples,
        self.total.w,
        self.left.w,
        right.w,
        impurity_parent,
        split.impurity_left,
        split.impurity_right,
    )


cdef inline intp_t _partition_values(Splitter self, float64_t threshold, bint missing_go_to_left) noexcept nogil:
    """Partition samples[start:end] and their feature_values by ``value <= threshold``, NaN as given."""
    cdef intp_t partition_start = self.start
    cdef intp_t partition_end = self.end
    cdef intp_t* samples = &self.samples[0]
    cdef float32_t* feature_values = &self.feature_values[0]
    cdef float32_t value
    while partition_start < partition_end:
        value = feature_values[partition_start]
        if value <= threshold or (missing_go_to_left and isnan(value)):
            partition_start += 1
        else:
            partition_end -= 1
            feature_values[partition_start], feature_values[partition_end] = (
                feature_values[partition_end], feature_values[partition_start])
            samples[partition_start], samples[partition_end] = samples[partition_end], samples[partition_start]
    return partition_end


cdef inline intp_t _move_missing_to_end(Splitter self, const float32_t* Xf) noexcept nogil:
    """Move the samples of the node missing feature ``Xf`` to its end, and return how many there are."""
    cdef intp_t* samples = &self.samples[0]
    cdef intp_t p = self.start
    cdef intp_t end = self.end
    while p < end:
        if isnan(Xf[samples[p]]):
            end -= 1
            samples[p], samples[end] = samples[end], samples[p]
        else:
            p += 1
    return self.end - end


cdef inline intp_t _gather_values(
    Splitter self, const float32_t* Xf, float32_t* min_value, float32_t* max_value
) noexcept nogil:
    """
    Copy the node's values of feature ``Xf`` into feature_values, and return how many are missing.

    ``min_value`` and ``max_value`` receive the extremes of the values that are not.
    """
    cdef const intp_t* samples = &self.samples[0]
    cdef float32_t* feature_values = &self.feature_values[0]
    cdef intp_t p
    cdef intp_t n_missing = 0
    cdef bint seen = False
    cdef float32_t value
    cdef float32_t low = 0.0
    cdef float32_t high = 0.0
    for p in range(self.start, self.end):
        value = Xf[samples[p]]
        feature_values[p] = value
        if isnan(value):
            n_missing += 1
        elif not seen:
            low = value
            high = value
            seen = True
        elif value < low:
            low = value
        elif value > high:
            high = value
    min_value[0] = low
    max_value[0] = high
    return n_missing


cdef inline void _search_sorted(
    Splitter self,
    SplitRecord* current_split,
    SplitRecord* best_split,
    float64_t* best_proxy_improvement,
) noexcept nogil:
    """
    Try every position between two distinct values of a feature that no sample of the node misses.

    samples[start:end] are sorted into feature_values. The sums of the left child are moved from
    whichever end is closer, as ``_update`` moves them, but kept in locals and passed on by value so
    that they can stay in registers.
    """
    cdef intp_t start = self.start
    cdef intp_t end = self.end
    cdef intp_t min_samples_leaf = self.min_samples_leaf
    cdef float64_t min_weight_leaf = self.min_weight_leaf
    cdef CriterionKind kind = self.kind
    cdef const float32_t* feature_values = &self.feature_values[0]
    cdef const intp_t* samples = &self.samples[0]
    cdef const float64_t* cost = self.cost
    cdef const float64_t* weight = self.weight
    cdef ClassSums total = self.total
    cdef ClassSums left, right
    cdef float64_t current_proxy_improvement, left_cost, right_cost
    cdef intp_t p = start
    cdef intp_t pos = start
    cdef intp_t p_prev, q

    sums_clear(&left)
    while p < end:
        p += 1
        while p < end and feature_values[p] <= feature_values[p - 1] + FEATURE_THRESHOLD:
            p += 1
        p_prev = p - 1
        if p == end:
            continue
        if p - start < min_samples_leaf or end - p < min_samples_leaf:
            continue

        if (p - pos) <= (end - p):
            if weight == NULL:
                for q in range(pos, p):
                    sums_add(&left, cost + 4 * samples[q], 1.0)
            else:
                for q in range(pos, p):
                    sums_add(&left, cost + 4 * samples[q], weight[samples[q]])
        else:
            left = total
            if weight == NULL:
                for q in range(end - 1, p - 1, -1):
                    sums_remove(&left, cost + 4 * samples[q], 1.0)
            else:
                for q in range(end - 1, p - 1, -1):
                    sums_remove(&left, cost + 4 * samples[q], weight[samples[q]])
        pos = p

        sums_subtract(&right, &total, &left)
        if left.w < min_weight_leaf or right.w < min_weight_leaf:
            continue

        # proxy_improvement, written out so that no sums escape to memory.
        if kind == COST:
            right_cost = decision_cost(right.cp[0], right.cp[1], right.cn[0], right.cn[1])
            left_cost = decision_cost(left.cp[0], left.cp[1], left.cn[0], left.cn[1])
            # The proxy is -(left_cost + right_cost) up to a few roundings, so a candidate that falls
            # short of the best by more than PROXY_SLACK cannot win and needs no divisions.
            if (left_cost + right_cost) * (1.0 - PROXY_SLACK) > -best_proxy_improvement[0]:
                continue
            current_proxy_improvement = -right.w * (right_cost / right.w) - left.w * (left_cost / left.w)
        else:
            current_proxy_improvement = (
                -right.w * impurity_of(kind, right.w, right.cw[0], right.cw[1],
                                       right.cp[0], right.cp[1], right.cn[0], right.cn[1])
                - left.w * impurity_of(kind, left.w, left.cw[0], left.cw[1],
                                       left.cp[0], left.cp[1], left.cn[0], left.cn[1])
            )
        if current_proxy_improvement > best_proxy_improvement[0]:
            best_proxy_improvement[0] = current_proxy_improvement
            current_split.pos = p
            # The sum of halves avoids overflowing to infinity.
            current_split.threshold = feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
            current_split.missing_go_to_left = p - start > end - p
            best_split[0] = current_split[0]


cdef inline void _search_with_missing(
    Splitter self,
    intp_t end_non_missing,
    SplitRecord* current_split,
    SplitRecord* best_split,
    float64_t* best_proxy_improvement,
) noexcept nogil:
    """
    Try every position between two distinct values of a feature that some samples of the node miss.

    samples[start:end_non_missing] hold the samples that have it, sorted into feature_values, and the
    rest hold those that miss it. Every position is tried with the missing samples on either side,
    and so is splitting the missing samples off from the rest, with an infinite threshold.
    """
    cdef intp_t start = self.start
    cdef intp_t end = self.end
    cdef intp_t n_missing = end - end_non_missing
    cdef intp_t min_samples_leaf = self.min_samples_leaf
    cdef float64_t min_weight_leaf = self.min_weight_leaf
    cdef CriterionKind kind = self.kind
    cdef const float32_t* feature_values = &self.feature_values[0]
    cdef intp_t p = start
    cdef intp_t p_prev
    cdef float64_t current_proxy_improvement
    cdef ClassSums missing, left, right

    _sum_samples(self, end_non_missing, end, &missing)
    self.scan_end = end_non_missing
    sums_subtract(&self.scan_total, &self.total, &missing)
    _reset(self)
    while p < end_non_missing:
        p += 1
        while p < end_non_missing and feature_values[p] <= feature_values[p - 1] + FEATURE_THRESHOLD:
            p += 1
        p_prev = p - 1
        _update(self, p)  # self.left: the samples left of p that have the feature

        # The missing samples go right.
        if p - start >= min_samples_leaf and end - p >= min_samples_leaf:
            sums_subtract(&right, &self.total, &self.left)
            if self.left.w >= min_weight_leaf and right.w >= min_weight_leaf:
                current_proxy_improvement = proxy_improvement(kind, &self.left, &right)
                if current_proxy_improvement > best_proxy_improvement[0]:
                    best_proxy_improvement[0] = current_proxy_improvement
                    current_split.pos = p
                    if p == end_non_missing:
                        current_split.threshold = INFINITY
                    else:
                        current_split.threshold = feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                    current_split.missing_go_to_left = False
                    best_split[0] = current_split[0]

        # The missing samples go left.
        if (p < end_non_missing and p - start + n_missing >= min_samples_leaf
                and end_non_missing - p >= min_samples_leaf):
            sums_subtract(&right, &self.scan_total, &self.left)
            sums_subtract(&left, &self.total, &right)
            if left.w >= min_weight_leaf and right.w >= min_weight_leaf:
                current_proxy_improvement = proxy_improvement(kind, &left, &right)
                if current_proxy_improvement > best_proxy_improvement[0]:
                    best_proxy_improvement[0] = current_proxy_improvement
                    current_split.pos = p + n_missing
                    current_split.threshold = feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                    current_split.missing_go_to_left = True
                    best_split[0] = current_split[0]

    self.scan_end = end
    self.scan_total = self.total


cdef inline void _sum_samples(Splitter self, intp_t start, intp_t end, ClassSums* out) noexcept nogil:
    """Sum the costs of samples[start:end] into ``out``."""
    cdef const intp_t* samples = &self.samples[0]
    cdef intp_t p
    sums_clear(out)
    if self.weight == NULL:
        for p in range(start, end):
            sums_add(out, self.cost + 4 * samples[p], 1.0)
    else:
        for p in range(start, end):
            sums_add(out, self.cost + 4 * samples[p], self.weight[samples[p]])


cdef float64_t _sum_in_order(const float64_t[::1] values) noexcept:
    cdef float64_t total = 0.0
    cdef intp_t i
    for i in range(values.shape[0]):
        total += values[i]
    return total
