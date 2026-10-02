# distutils: language = c++
"""
Find the best split of a node of a cost-sensitive decision tree.

The split search is ported from scikit-learn's ``node_split_best`` and ``node_split_random``
(sklearn/tree/_splitter.pyx and sklearn/tree/_partitioner.pyx), Copyright (c) the scikit-learn
developers, BSD-3-Clause license. It keeps their feature sampling, constant-feature bookkeeping and
tie handling. Missing values, sparse input and monotonic constraints are not supported. The
criterion's running sums are inline (see ``_criterion.pxd``), and every sample's costs are read from
a single record.
"""

from libc.math cimport INFINITY, fmin
from libc.string cimport memcpy

import numpy as np

from ._criterion cimport (
    CriterionKind, ClassSums, impurity, impurity_improvement, proxy_improvement,
    sums_add, sums_clear, sums_remove, sums_subtract,
)
from ._utils cimport float_to_key, key_to_float, radix_sort, rand_int, rand_uniform, sort

# Two feature values closer than this count as equal: a split between them is never considered.
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Nodes at least this large are sorted by radix sort, smaller ones by introsort.
cdef intp_t RADIX_SORT_MIN_SAMPLES = 512

# The relative rounding error of a sum, per term, assumed by max_cost_decrease. Generous: the bound it
# guards may only ever overestimate.
cdef float64_t ROUNDING = 4.0 * np.finfo(np.float64).eps


cdef inline void _init_split(SplitRecord* split, intp_t start_pos) noexcept nogil:
    split.impurity_left = INFINITY
    split.impurity_right = INFINITY
    split.pos = start_pos
    split.feature = 0
    split.threshold = 0.0
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
    ):
        cdef const float32_t[::1, :] X_view = X
        cdef const float64_t[:, ::1] cost_view = cost
        cdef const float64_t[::1] weight_view
        if cost_view.shape[0] != X_view.shape[0] or cost_view.shape[1] != 4:
            raise ValueError('cost must have shape (n_rows, 4)')

        self._X_ref = X
        self._cost_ref = cost
        self._weight_ref = weight
        self.n_rows = X_view.shape[0]
        self.n_features = X_view.shape[1]
        self.X = &X_view[0, 0]
        self.cost = &cost_view[0, 0]

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
            self.sort_keys = np.empty(self.n_samples, dtype=np.uint32)
            self.sort_keys_buffer = np.empty(self.n_samples, dtype=np.uint32)
            self.sort_samples_buffer = np.empty(self.n_samples, dtype=np.intp)

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
                oracle += w * fmin(record[0], record[1])
        self.start = start
        self.end = end
        self.total = total
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
        cdef float64_t node_cost = fmin(pos_cost, neg_cost)
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

        cdef intp_t f_i = n_features
        cdef intp_t f_j, p, p_prev
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

            if not self.random:
                _sort_node(self, Xf)
                min_feature_value = feature_values[start]
                max_feature_value = feature_values[end - 1]
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

            if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]

            if not self.random:
                # Try every position between two distinct values.
                _reset(self)
                p = start
                while p < end:
                    p += 1
                    while p < end and feature_values[p] <= feature_values[p - 1] + FEATURE_THRESHOLD:
                        p += 1
                    p_prev = p - 1
                    if p == end:
                        continue
                    if p - start < min_samples_leaf or end - p < min_samples_leaf:
                        continue

                    current_split.pos = p
                    _update(self, p)
                    sums_subtract(&right, &self.total, &self.left)
                    if self.left.w < min_weight_leaf or right.w < min_weight_leaf:
                        continue

                    current_proxy_improvement = proxy_improvement(kind, &self.left, &right)
                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        # The sum of halves avoids overflowing to infinity.
                        current_split.threshold = feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                        best_split = current_split
            else:
                # Try one random threshold.
                current_split.threshold = rand_uniform(min_feature_value, max_feature_value, random_state)
                if current_split.threshold == max_feature_value:
                    current_split.threshold = min_feature_value
                current_split.pos = _partition_values(self, current_split.threshold)
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
                    best_split = current_split

        if best_split.pos < end:
            # The random search left the samples partitioned by the last feature it drew.
            if not self.random or current_split.feature != best_split.feature:
                _partition(self, best_split.feature, best_split.threshold)
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
    if (new_pos - self.pos) <= (self.end - new_pos):
        left = self.left
        if weight == NULL:
            for p in range(self.pos, new_pos):
                sums_add(&left, cost + 4 * samples[p], 1.0)
        else:
            for p in range(self.pos, new_pos):
                sums_add(&left, cost + 4 * samples[p], weight[samples[p]])
    else:
        left = self.total
        if weight == NULL:
            for p in range(self.end - 1, new_pos - 1, -1):
                sums_remove(&left, cost + 4 * samples[p], 1.0)
        else:
            for p in range(self.end - 1, new_pos - 1, -1):
                sums_remove(&left, cost + 4 * samples[p], weight[samples[p]])
    self.left = left
    self.pos = new_pos


cdef inline void _sort_node(Splitter self, const float32_t* Xf) noexcept nogil:
    """Sort samples[start:end] by their value of feature ``Xf``, into feature_values[start:end]."""
    cdef intp_t start = self.start
    cdef intp_t end = self.end
    cdef intp_t p
    cdef intp_t* samples = &self.samples[0]
    cdef float32_t* feature_values = &self.feature_values[0]
    cdef uint32_t* keys
    if end - start < RADIX_SORT_MIN_SAMPLES:
        for p in range(start, end):
            feature_values[p] = Xf[samples[p]]
        sort(feature_values + start, samples + start, end - start)
    else:
        keys = &self.sort_keys[0]
        for p in range(start, end):
            keys[p] = float_to_key(Xf[samples[p]])
        radix_sort(keys + start, samples + start, &self.sort_keys_buffer[0], &self.sort_samples_buffer[0], end - start)
        for p in range(start, end):
            feature_values[p] = key_to_float(keys[p])


cdef inline void _partition(Splitter self, intp_t feature, float64_t threshold) noexcept nogil:
    """Reorder samples[start:end] so those going left by the given rule come first."""
    cdef intp_t partition_start = self.start
    cdef intp_t partition_end = self.end
    cdef intp_t* samples = &self.samples[0]
    cdef const float32_t* Xf = self.X + feature * self.n_rows
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


cdef inline intp_t _partition_values(Splitter self, float64_t threshold) noexcept nogil:
    """Partition samples[start:end] and their feature_values by ``value <= threshold``."""
    cdef intp_t partition_start = self.start
    cdef intp_t partition_end = self.end
    cdef intp_t* samples = &self.samples[0]
    cdef float32_t* feature_values = &self.feature_values[0]
    while partition_start < partition_end:
        if feature_values[partition_start] <= threshold:
            partition_start += 1
        else:
            partition_end -= 1
            feature_values[partition_start], feature_values[partition_end] = (
                feature_values[partition_end], feature_values[partition_start])
            samples[partition_start], samples[partition_end] = samples[partition_end], samples[partition_start]
    return partition_end


cdef float64_t _sum_in_order(const float64_t[::1] values) noexcept:
    cdef float64_t total = 0.0
    cdef intp_t i
    for i in range(values.shape[0]):
        total += values[i]
    return total
