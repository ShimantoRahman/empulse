import numpy as np
cimport numpy as cnp

from .tree cimport Tree, SplitValues
from .forest cimport Forest

cdef Tree* find_best_tree(Forest* population) noexcept

cdef Forest* initialize_population(
    int pop_size,
    int n_features,
    SplitValues* split_values,
    int max_depth,
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int min_samples_split,
    int min_samples_leaf,
    float alpha,
    object fitness_function,
) noexcept

cdef Tree* evolve_tree(
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

cdef inline void fit_predict(
    float[:, :] X,
    int[:] y,
    Tree* tree,
    float[:] predictions,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
) noexcept nogil

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
)

cdef EvolutionResult evolve_forest_deterministic(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
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
)
