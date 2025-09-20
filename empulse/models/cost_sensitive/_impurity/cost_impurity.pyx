cimport numpy as cnp
import numpy as np
from sklearn.utils._typedefs cimport float64_t, intp_t
from sklearn.tree._criterion cimport ClassificationCriterion
from libc.math cimport fmin
from libc.math cimport log as ln
from libc.string cimport memcpy
from libc.string cimport memset


cdef inline float64_t log(float64_t x) noexcept nogil:
    return ln(x) / ln(2.0)


cdef class CostImpurity(ClassificationCriterion):
    cdef float64_t tp_cost, tn_cost, fp_cost, fn_cost
    cdef float64_t[:] tp_cost_array, tn_cost_array, fp_cost_array, fn_cost_array
    cdef bint all_class_dependent

    # New: cost accumulators
    cdef float64_t[:, ::1] pos_cost_sum_total
    cdef float64_t[:, ::1] neg_cost_sum_total
    cdef float64_t[:, ::1] pos_cost_sum_left
    cdef float64_t[:, ::1] neg_cost_sum_left
    cdef float64_t[:, ::1] pos_cost_sum_right
    cdef float64_t[:, ::1] neg_cost_sum_right

    def __deepcopy__(self, memo):
        # Convert memoryview to numpy array
        n_classes_array = np.asarray(self.n_classes)
        new_instance = CostImpurity(self.n_outputs, n_classes_array)

        # Copy all attributes (otherwise not included)
        new_instance.tp_cost = self.tp_cost
        new_instance.tn_cost = self.tn_cost
        new_instance.fp_cost = self.fp_cost
        new_instance.fn_cost = self.fn_cost
        new_instance.tp_cost_array = np.array(self.tp_cost_array)
        new_instance.tn_cost_array = np.array(self.tn_cost_array)
        new_instance.fp_cost_array = np.array(self.fp_cost_array)
        new_instance.fn_cost_array = np.array(self.fn_cost_array)
        new_instance.all_class_dependent = self.all_class_dependent
        new_instance.pos_cost_sum_total = np.array(self.pos_cost_sum_total)
        new_instance.neg_cost_sum_total = np.array(self.neg_cost_sum_total)
        new_instance.pos_cost_sum_left = np.array(self.pos_cost_sum_left)
        new_instance.neg_cost_sum_left = np.array(self.neg_cost_sum_left)
        new_instance.pos_cost_sum_right = np.array(self.pos_cost_sum_right)
        new_instance.neg_cost_sum_right = np.array(self.neg_cost_sum_right)

        # Copy other attributes if necessary
        memo[id(self)] = new_instance
        return new_instance

    def set_costs(self, float64_t tp_cost=0.0, float64_t tn_cost=0.0, float64_t fp_cost=0.0, float64_t fn_cost=0.0):
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def set_array_costs(self, float64_t[:] tp_cost, float64_t[:] tn_cost, float64_t[:] fp_cost, float64_t[:] fn_cost, intp_t n_samples):
        cdef bint is_tp_cost_array, is_tn_cost_array, is_fp_cost_array, is_fn_cost_array
        is_tp_cost_array = tp_cost.shape[0] > 0
        is_tn_cost_array = tn_cost.shape[0] > 0
        is_fp_cost_array = fp_cost.shape[0] > 0
        is_fn_cost_array = fn_cost.shape[0] > 0
        self.all_class_dependent = not (
            is_tp_cost_array or
            is_tn_cost_array or
            is_fp_cost_array or
            is_fn_cost_array
        )
        if is_tp_cost_array:
            self.tp_cost_array = tp_cost
        else:
            self.tp_cost_array = np.full(n_samples, self.tp_cost, dtype=np.float64)
        if is_tn_cost_array:
            self.tn_cost_array = tn_cost
        else:
            self.tn_cost_array = np.full(n_samples, self.tn_cost, dtype=np.float64)
        if is_fp_cost_array:
            self.fp_cost_array = fp_cost
        else:
            self.fp_cost_array = np.full(n_samples, self.fp_cost, dtype=np.float64)
        if is_fn_cost_array:
            self.fn_cost_array = fn_cost
        else:
            self.fn_cost_array = np.full(n_samples, self.fn_cost, dtype=np.float64)
        self.pos_cost_sum_total = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)
        self.neg_cost_sum_total = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)
        self.pos_cost_sum_left = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)
        self.neg_cost_sum_left = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)
        self.pos_cost_sum_right = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)
        self.neg_cost_sum_right = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        # Standard class counts
        ClassificationCriterion.init(self, y, sample_weight, weighted_n_samples, sample_indices, start, end)
        # Cost accumulators
        cdef intp_t k, c, i, p
        cdef float64_t w = 1.0
        memset(&self.pos_cost_sum_total[0, 0], 0, self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memset(&self.neg_cost_sum_total[0, 0], 0, self.n_outputs * self.max_n_classes * sizeof(float64_t))
        for p in range(start, end):
            i = sample_indices[p]

            for k in range(self.n_outputs):
                c = <intp_t> y[i, k]
                if self.all_class_dependent:
                    if c == 1:
                        self.pos_cost_sum_total[k, c] += self.tp_cost
                        self.neg_cost_sum_total[k, c] += self.fn_cost
                    else:
                        self.pos_cost_sum_total[k, c] += self.fp_cost
                        self.neg_cost_sum_total[k, c] += self.tn_cost
                else:
                    if c == 1:
                        self.pos_cost_sum_total[k, c] += self.tp_cost_array[i]
                        self.neg_cost_sum_total[k, c] += self.fn_cost_array[i]
                    else:
                        self.pos_cost_sum_total[k, c] += self.fp_cost_array[i]
                        self.neg_cost_sum_total[k, c] += self.tn_cost_array[i]
        # Reset left/right
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        self.pos = self.start
        memset(&self.pos_cost_sum_left[0, 0], 0, self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memset(&self.neg_cost_sum_left[0, 0], 0, self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memcpy(&self.pos_cost_sum_right[0, 0], &self.pos_cost_sum_total[0, 0], self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memcpy(&self.neg_cost_sum_right[0, 0], &self.neg_cost_sum_total[0, 0], self.n_outputs * self.max_n_classes * sizeof(float64_t))
        ClassificationCriterion.reset(self)
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        self.pos = self.end
        memset(&self.pos_cost_sum_right[0, 0], 0, self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memset(&self.neg_cost_sum_right[0, 0], 0, self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memcpy(&self.pos_cost_sum_left[0, 0], &self.pos_cost_sum_total[0, 0], self.n_outputs * self.max_n_classes * sizeof(float64_t))
        memcpy(&self.neg_cost_sum_left[0, 0], &self.neg_cost_sum_total[0, 0], self.n_outputs * self.max_n_classes * sizeof(float64_t))
        ClassificationCriterion.reverse_reset(self)
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        cdef intp_t pos = self.pos
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef intp_t i, p, k, c
        cdef float64_t w = 1.0
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]
                if sample_weight is not None:
                    w = sample_weight[i]
                for k in range(self.n_outputs):
                    c = <intp_t> self.y[i, k]
                    self.sum_left[k, c] += w
                    if self.all_class_dependent:
                        if c == 1:
                            self.pos_cost_sum_left[k, c] += w * self.tp_cost
                            self.neg_cost_sum_left[k, c] += w * self.fn_cost
                        else:
                            self.pos_cost_sum_left[k, c] += w * self.fp_cost
                            self.neg_cost_sum_left[k, c] += w * self.tn_cost
                    else:
                        if c == 1:
                            self.pos_cost_sum_left[k, c] += w * self.tp_cost_array[i]
                            self.neg_cost_sum_left[k, c] += w * self.fn_cost_array[i]
                        else:
                            self.pos_cost_sum_left[k, c] += w * self.fp_cost_array[i]
                            self.neg_cost_sum_left[k, c] += w * self.tn_cost_array[i]
                self.weighted_n_left += w
        else:
            self.reverse_reset()
            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]
                if sample_weight is not None:
                    w = sample_weight[i]
                for k in range(self.n_outputs):
                    c = <intp_t> self.y[i, k]
                    self.sum_left[k, c] -= w
                    if self.all_class_dependent:
                        if c == 1:
                            self.pos_cost_sum_left[k, c] -= w * self.tp_cost
                            self.neg_cost_sum_left[k, c] -= w * self.fn_cost
                        else:
                            self.pos_cost_sum_left[k, c] -= w * self.fp_cost
                            self.neg_cost_sum_left[k, c] -= w * self.tn_cost
                    else:
                        if c == 1:
                            self.pos_cost_sum_left[k, c] -= w * self.tp_cost_array[i]
                            self.neg_cost_sum_left[k, c] -= w * self.fn_cost_array[i]
                        else:
                            self.pos_cost_sum_left[k, c] -= w * self.fp_cost_array[i]
                            self.neg_cost_sum_left[k, c] -= w * self.tn_cost_array[i]
                self.weighted_n_left -= w
        # Update right
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]
                self.pos_cost_sum_right[k, c] = self.pos_cost_sum_total[k, c] - self.pos_cost_sum_left[k, c]
                self.neg_cost_sum_right[k, c] = self.neg_cost_sum_total[k, c] - self.neg_cost_sum_left[k, c]
        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        cdef float64_t pos_cost = 0.0
        cdef float64_t neg_cost = 0.0
        cdef intp_t k, c
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                pos_cost += self.pos_cost_sum_total[k, c]
                neg_cost += self.neg_cost_sum_total[k, c]
        return fmin(pos_cost, neg_cost) / self.weighted_n_node_samples

    cdef void children_impurity(self, float64_t* impurity_left, float64_t* impurity_right) noexcept nogil:
        cdef float64_t pos_cost_left = 0.0
        cdef float64_t neg_cost_left = 0.0
        cdef float64_t pos_cost_right = 0.0
        cdef float64_t neg_cost_right = 0.0
        cdef intp_t k, c
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                pos_cost_left += self.pos_cost_sum_left[k, c]
                neg_cost_left += self.neg_cost_sum_left[k, c]
                pos_cost_right += self.pos_cost_sum_right[k, c]
                neg_cost_right += self.neg_cost_sum_right[k, c]
        impurity_left[0] = fmin(pos_cost_left, neg_cost_left) / self.weighted_n_left
        impurity_right[0] = fmin(pos_cost_right, neg_cost_right) / self.weighted_n_right


cdef class GiniCostImpurity(CostImpurity):

    def __deepcopy__(self, memo):
        # Convert memoryview to numpy array
        n_classes_array = np.asarray(self.n_classes)
        new_instance = GiniCostImpurity(self.n_outputs, n_classes_array)

        # Copy all attributes (otherwise not included)
        new_instance.tp_cost = self.tp_cost
        new_instance.tn_cost = self.tn_cost
        new_instance.fp_cost = self.fp_cost
        new_instance.fn_cost = self.fn_cost
        new_instance.tp_cost_array = np.array(self.tp_cost_array)
        new_instance.tn_cost_array = np.array(self.tn_cost_array)
        new_instance.fp_cost_array = np.array(self.fp_cost_array)
        new_instance.fn_cost_array = np.array(self.fn_cost_array)
        new_instance.all_class_dependent = self.all_class_dependent
        new_instance.pos_cost_sum_total = np.array(self.pos_cost_sum_total)
        new_instance.neg_cost_sum_total = np.array(self.neg_cost_sum_total)
        new_instance.pos_cost_sum_left = np.array(self.pos_cost_sum_left)
        new_instance.neg_cost_sum_left = np.array(self.neg_cost_sum_left)
        new_instance.pos_cost_sum_right = np.array(self.pos_cost_sum_right)
        new_instance.neg_cost_sum_right = np.array(self.neg_cost_sum_right)

        # Copy other attributes if necessary
        memo[id(self)] = new_instance
        return new_instance

    cdef float64_t node_impurity(self) noexcept nogil:
        cdef float64_t pos_sq_prior, neg_sq_prior
        cdef float64_t pos_count, neg_count
        cdef float64_t pos_cost = 0.0
        cdef float64_t neg_cost = 0.0
        cdef intp_t k, c
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                pos_cost += self.pos_cost_sum_total[k, c]
                neg_cost += self.neg_cost_sum_total[k, c]
        pos_count = self.sum_total[0, 1]
        pos_sq_prior = pos_count * pos_count / self.weighted_n_node_samples
        neg_count = self.sum_total[0, 0]
        neg_sq_prior = neg_count * neg_count / self.weighted_n_node_samples
        return fmin(pos_cost * pos_sq_prior, neg_cost * neg_sq_prior) / self.weighted_n_node_samples

    cdef void children_impurity(self, float64_t* impurity_left, float64_t* impurity_right) noexcept nogil:
        cdef float64_t pos_sq_prior_left, neg_sq_prior_left, pos_sq_prior_right, neg_sq_prior_right
        cdef float64_t pos_count_left, neg_count_left, pos_count_right, neg_count_right
        cdef float64_t pos_cost_left = 0.0
        cdef float64_t neg_cost_left = 0.0
        cdef float64_t pos_cost_right = 0.0
        cdef float64_t neg_cost_right = 0.0
        cdef intp_t k, c
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                pos_cost_left += self.pos_cost_sum_left[k, c]
                neg_cost_left += self.neg_cost_sum_left[k, c]
                pos_cost_right += self.pos_cost_sum_right[k, c]
                neg_cost_right += self.neg_cost_sum_right[k, c]

        pos_count_left = self.sum_left[0, 1]
        pos_sq_prior_left = pos_count_left * pos_count_left / self.weighted_n_left
        neg_count_left = self.sum_left[0, 0]
        neg_sq_prior_left = neg_count_left * neg_count_left / self.weighted_n_left

        pos_count_right = self.sum_right[0, 1]
        pos_sq_prior_right = pos_count_right * pos_count_right / self.weighted_n_right
        neg_count_right = self.sum_right[0, 0]
        neg_sq_prior_right = neg_count_right * neg_count_right / self.weighted_n_right

        impurity_left[0] = fmin(pos_cost_left * pos_sq_prior_left, neg_cost_left * neg_sq_prior_left) / self.weighted_n_left
        impurity_right[0] = fmin(pos_cost_right * pos_sq_prior_right, neg_cost_right * neg_sq_prior_right) / self.weighted_n_right


cdef class EntropyCostImpurity(CostImpurity):

    def __deepcopy__(self, memo):
        # Convert memoryview to numpy array
        n_classes_array = np.asarray(self.n_classes)
        new_instance = EntropyCostImpurity(self.n_outputs, n_classes_array)

        # Copy all attributes (otherwise not included)
        new_instance.tp_cost = self.tp_cost
        new_instance.tn_cost = self.tn_cost
        new_instance.fp_cost = self.fp_cost
        new_instance.fn_cost = self.fn_cost
        new_instance.tp_cost_array = np.array(self.tp_cost_array)
        new_instance.tn_cost_array = np.array(self.tn_cost_array)
        new_instance.fp_cost_array = np.array(self.fp_cost_array)
        new_instance.fn_cost_array = np.array(self.fn_cost_array)
        new_instance.all_class_dependent = self.all_class_dependent
        new_instance.pos_cost_sum_total = np.array(self.pos_cost_sum_total)
        new_instance.neg_cost_sum_total = np.array(self.neg_cost_sum_total)
        new_instance.pos_cost_sum_left = np.array(self.pos_cost_sum_left)
        new_instance.neg_cost_sum_left = np.array(self.neg_cost_sum_left)
        new_instance.pos_cost_sum_right = np.array(self.pos_cost_sum_right)
        new_instance.neg_cost_sum_right = np.array(self.neg_cost_sum_right)

        # Copy other attributes if necessary
        memo[id(self)] = new_instance
        return new_instance

    cdef float64_t node_impurity(self) noexcept nogil:
        cdef float64_t pos_entropy = 0.0
        cdef float64_t neg_entropy = 0.0
        cdef float64_t pos_count, neg_count
        cdef float64_t pos_cost = 0.0
        cdef float64_t neg_cost = 0.0
        cdef intp_t k, c
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                pos_cost += self.pos_cost_sum_total[k, c]
                neg_cost += self.neg_cost_sum_total[k, c]
        pos_cost /= self.weighted_n_node_samples
        neg_cost /= self.weighted_n_node_samples

        pos_count = self.sum_total[0, 1]
        if pos_count > 0.0:
            pos_entropy = log(pos_count / self.weighted_n_node_samples)
        neg_count = self.sum_total[0, 0]
        if neg_count > 0.0:
            neg_entropy = log(neg_count / self.weighted_n_node_samples)
        return fmin(pos_cost * -pos_entropy, neg_cost * -neg_entropy)

    cdef void children_impurity(self, float64_t* impurity_left, float64_t* impurity_right) noexcept nogil:
        cdef float64_t pos_entropy_left = 0.0
        cdef float64_t neg_entropy_left = 0.0
        cdef float64_t pos_entropy_right = 0.0
        cdef float64_t neg_entropy_right = 0.0
        cdef float64_t pos_count_left, neg_count_left, pos_count_right, neg_count_right
        cdef float64_t pos_cost_left = 0.0
        cdef float64_t neg_cost_left = 0.0
        cdef float64_t pos_cost_right = 0.0
        cdef float64_t neg_cost_right = 0.0
        cdef intp_t k, c
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                pos_cost_left += self.pos_cost_sum_left[k, c]
                neg_cost_left += self.neg_cost_sum_left[k, c]
                pos_cost_right += self.pos_cost_sum_right[k, c]
                neg_cost_right += self.neg_cost_sum_right[k, c]
        pos_cost_left /= self.weighted_n_left
        pos_cost_right /= self.weighted_n_right
        neg_cost_left /= self.weighted_n_left
        neg_cost_right /= self.weighted_n_right

        pos_count_left = self.sum_left[0, 1]
        if pos_count_left > 0.0:
            pos_entropy_left = log(pos_count_left / self.weighted_n_left)
        neg_count_left = self.sum_left[0, 0]
        if neg_count_left > 0.0:
            neg_entropy_left = log(neg_count_left / self.weighted_n_left)
        pos_count_right = self.sum_right[0, 1]
        if pos_count_right > 0.0:
            pos_entropy_right = log(pos_count_right / self.weighted_n_right)
        neg_count_right = self.sum_right[0, 0]
        if neg_count_right > 0.0:
            neg_entropy_right = log(neg_count_right / self.weighted_n_right)

        impurity_left[0] = fmin(pos_cost_left * pos_entropy_left, neg_cost_left * neg_entropy_left)
        impurity_right[0] = fmin(pos_cost_right * pos_entropy_right, neg_cost_right * neg_entropy_right)