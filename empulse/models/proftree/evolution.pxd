import numpy as np
cimport numpy as cnp

from .tree cimport Tree, SplitValues
from .forest cimport Forest

cdef Forest* initialize_population(
    int pop_size,
    int n_features,
    SplitValues* split_values,
    int max_depth,
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
) noexcept

cdef Tree* find_best_tree(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int pop_size = *,
    int tournament_size = *,
    int n_elites = *,
    float crossover_rate = *,
    float mutation_rate = *,
    int max_depth = *,
    int max_generations = *,
    int patience = *,
    float tol = *,
    int random_state = *,
)
