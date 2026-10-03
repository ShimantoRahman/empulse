import numpy as np
cimport numpy as cnp

from .tree cimport Tree, SplitValues
from .forest cimport Forest
from .random cimport RandState

# The compiled expected maximum profit of samples grouped by score, given the coefficient parts of
# the profit (see empulse.metrics._cy_max_profit.piecewise._expected_max_profit_of_groups).
ctypedef double (*ExpectedMaxProfitOfGroups)(
    const double* y_score,
    const long long* n_positive,
    const long long* n_negative,
    Py_ssize_t n_groups,
    const double* parts,
    Py_ssize_t n_powers,
    double lower_bound,
    double upper_bound,
    int distribution,
    const double* distribution_parameters,
) noexcept nogil

cdef struct NativeFitness:
    # The maximum profit for these class values, unless expected_max_profit is set.
    float tp_benefit
    float tn_benefit
    float fp_cost
    float fn_cost
    # The expected maximum profit, computed in closed form.
    ExpectedMaxProfitOfGroups expected_max_profit
    const double* coefficient_parts
    Py_ssize_t n_powers
    double lower_bound
    double upper_bound
    int distribution
    const double* distribution_parameters

cdef Tree* find_best_tree(Forest* population) noexcept

cdef Forest* random_population(
    RandState* rng,
    int pop_size,
    int n_features,
    SplitValues* split_values,
    int max_depth,
) noexcept nogil

cdef void fit_population(
    Forest* population,
    const float[:, ::1] X,
    const int[:] y,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
    int** scratch,
    int n_threads,
) noexcept nogil

cdef void fit_population_native(
    Forest* population,
    const float[:, ::1] X,
    const int[:] y,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
    const NativeFitness* fitness,
    float alpha,
    int** scratch,
    int n_threads,
) noexcept nogil

cdef void evaluate_population(
    Forest* population,
    const float[:, ::1] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int n_samples,
    object fitness_function,
    bint fitness_from_leaves,
    float alpha,
)

cdef Tree* evolve_tree(
    RandState* rng,
    Forest* population,
    SplitValues* split_values,
    int n_features,
    int max_depth,
    float crossover_rate,
    float grow_rate,
    float prune_rate,
    float mutate_split_rate,
    int index,
) noexcept nogil

cdef inline void insert_offspring(Forest* population, Forest* offspring, int i) noexcept nogil

cdef inline void evaluate(
    Tree* tree,
    object fitness_function,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    cnp.ndarray[cnp.float32_t, ndim=1] predictions,
    float alpha,
)

cdef inline bint stop_evolution(
    Tree* challenger,
    Tree** champion,
    int* stagnation_counter,
    float tolerance,
    int patience,
) noexcept

cdef struct EvolutionResult:
    Tree* tree
    int n_generations

cdef EvolutionResult evolve_forest_stochastic(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    object fitness_function,
    int pop_size = *,
    int max_depth = *,
    int max_generations = *,
    int min_samples_split = *,
    int min_samples_leaf = *,
    float crossover_rate = *,
    float grow_rate = *,
    float prune_rate = *,
    float mutate_split_rate = *,
    float mutate_value_rate = *,
    int patience = *,
    float tol = *,
    float alpha = *,
    int random_state = *,
    int n_threads = *,
    bint fitness_from_leaves = *,
    bint cache_samples = *,
)

cdef EvolutionResult evolve_forest_native(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    NativeFitness fitness,
    int pop_size = *,
    int max_depth = *,
    int max_generations = *,
    int min_samples_split = *,
    int min_samples_leaf  = *,
    float crossover_rate = *,
    float grow_rate = *,
    float prune_rate = *,
    float mutate_split_rate = *,
    float mutate_value_rate = *,
    int patience = *,
    float tol = *,
    float alpha = *,
    int random_state = *,
    int n_threads = *,
    bint cache_samples = *,
)
