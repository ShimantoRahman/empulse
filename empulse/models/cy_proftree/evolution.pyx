import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc, srand
from libc.time cimport time
from libc.math cimport isnan

from .tree cimport (Tree, SplitValues, create_tree, copy_tree, free_tree,
                    compute_split_values, free_split_values, reset_tree,
                    fit_tree, predict_labels_tree)
from .forest cimport Forest, create_forest, free_forest, sort_population
from .operators cimport tournament_selection, crossover, evolve
from .random cimport rand_fraction


cdef Tree* find_best_tree(Forest* population) noexcept:
    cdef int i
    cdef Tree* best_tree = population.trees[0]
    for i in range(1, population.size):
        if population.trees[i].fitness > best_tree.fitness:
            best_tree = population.trees[i]
    return copy_tree(best_tree)

cdef Forest* initialize_population(
    int pop_size,
    int n_features,
    SplitValues* split_values,
    int max_depth,
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
) noexcept:
    cdef Forest* population = create_forest(pop_size)
    cdef Tree* tree
    cdef int i, j
    cdef int n_samples = X.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float accuracy
    for i in range(pop_size):
        tree = create_tree()
        grow_subtree(tree.root, n_features, split_values, depth=0, max_depth=max_depth)
        fit_tree(tree, X, y, n_samples=n_samples)
        predict_labels_tree(tree, X, predictions, n_samples)
        accuracy = 0.0
        for j in range(n_samples):
            if predictions[j] == y[j]:
                accuracy += 1
        accuracy /= n_samples
        tree.fitness = accuracy
        population.trees[i] = tree
    return population

cpdef Tree* evolve_forest(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int pop_size = 100,
    int tournament_size = 3,
    int max_depth = 9,
    int max_generations = 10_000,
    int min_samples_split = 20,
    int min_samples_leaf  = 7,
    float crossover_rate = 0.2,
    float grow_rate = 0.2,
    float prune_rate = 0.2,
    float mutate_split_rate = 0.2,
    float mutate_value_rate = 0.2,
    int patience = 100,
    float tol = 1e-3,
    float alpha = 0.0,
    int random_state = -1,
):
    if random_state != -1:
        srand(<unsigned int>random_state)
    else:
        srand(<unsigned int>time(NULL))

    cdef cnp.int32_t n_samples = <int>X.shape[0]
    cdef cnp.int32_t n_features = <int>X.shape[1]
    cdef int i, j
    cdef SplitValues* split_values = compute_split_values(X)
    tournament_size = min(tournament_size, pop_size)

    cdef Forest* population = initialize_population(
        pop_size=pop_size,
        n_features=n_features,
        split_values=split_values,
        max_depth=max_depth,
        X=X,
        y=y,
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fitness_scores = np.zeros(pop_size, dtype=np.float32)
    cdef float[:] fitness_scores_view = fitness_scores

    cdef float gen_best
    cdef int stagnation_counter = 0
    cdef Tree* parent
    cdef Tree* partner
    cdef Tree* child
    cdef Tree* gen_best_tree
    cdef Tree* best_tree
    cdef Forest* offspring = create_forest(pop_size)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
    cdef float accuracy

    # set up the rates for various genetic operations
    cdef float probability = 0.0
    grow_rate = crossover_rate + grow_rate
    prune_rate = grow_rate + prune_rate
    mutate_split_rate = prune_rate + mutate_split_rate
    mutate_value_rate = mutate_split_rate + mutate_value_rate

    for _ in range(max_generations):
        for i in range(pop_size):

            probability = rand_fraction()
            parent = copy_tree(population.trees[i])

            if probability < crossover_rate:
                partner = copy_tree(choose_tree(population))
                child = crossover(parent, partner, max_depth)  # TODO: check whether copy_tree inside crossover is necessary
            elif probability < grow_rate:
                child = grow(parent, max_depth)
            elif probability < prune_rate:
                child = prune(parent)
            elif probability < mutate_split_rate:
                child = mutate_split_feature(parent, split_values, n_features, max_depth)
            else:
                child = mutate_split_value(parent, split_values)

            offspring.trees[i] = child

        for i in range(pop_size):
            free_tree(population.trees[i])
            population.trees[i] = offspring.trees[i]
            offspring.trees[i] = NULL

            child = population.trees[i]
            reset_tree(child)
            fit_tree(child, X, y, n_samples, min_samples_split, min_samples_leaf)
            predict_labels_tree(child, X, predictions_view, n_samples)

            child.fitness = yield predictions_view
            child.fitness += alpha * child.size

            # if isnan(child.fitness):
            #     accuracy = 0.0
            #     for j in range(n_samples):
            #         if predictions_view[j] == y[j]:
            #             accuracy += 1
            #     accuracy /= n_samples
            #     child.fitness = accuracy


        gen_best_tree = find_best_tree(population)
        if gen_best_tree.fitness > best_tree.fitness * (1.0 + tol):
            free_tree(best_tree)
            best_tree = gen_best_tree
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= patience:
            break

    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    yield best_tree
