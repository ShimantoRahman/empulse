# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free

from .tree cimport (Tree, SplitValues, create_tree, copy_tree, free_tree,
                    compute_split_values, free_split_values, reset_tree,
                    fit_tree, predict_proba_tree, split, prune_illegal_nodes)
from .forest cimport Forest, create_forest, free_forest, choose_different_tree
from .operators cimport crossover, grow, prune_internal, mutate_split_feature, mutate_split_value
from .random cimport rand_fraction, set_seed
from .max_profit cimport max_profit_score


cdef Tree* find_best_tree(Forest* population) noexcept:
    if population is NULL or population.n_trees == 0:
        return NULL
    cdef int i
    cdef Tree* best_tree = population.trees[0]
    for i in range(1, population.n_trees):
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
    int min_samples_split,
    int min_samples_leaf,
    float alpha,
    object fitness_function,
) noexcept:
    cdef Forest* population = create_forest(pop_size)
    cdef Tree* tree
    cdef int i, j
    cdef int n_samples = X.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
    for i in range(pop_size):
        tree = create_tree()
        split(tree.root, n_features, split_values, depth=0, max_depth=max_depth)
        fit_tree(tree, X, y, n_samples)
        predict_proba_tree(tree, X, predictions_view, n_samples)
        evaluate(tree, fitness_function, y, predictions, alpha)
        population.trees[i] = tree
    return population

cdef Forest* initialize_population_max_profit(
    int pop_size,
    int n_features,
    SplitValues* split_values,
    int max_depth,
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
    int min_samples_split,
    int min_samples_leaf,
    float alpha,
) noexcept:
    cdef Forest* population = create_forest(pop_size)
    cdef Tree* tree
    cdef int i, j
    cdef int n_samples = X.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
    for i in range(pop_size):
        tree = create_tree()
        split(tree.root, n_features, split_values, depth=0, max_depth=max_depth)
        fit_tree(tree, X, y, n_samples)
        predict_proba_tree(tree, X, predictions_view, n_samples)
        evaluate_max_profit(tree, y, predictions, tp_benefit, tn_benefit, fp_cost, fn_cost, alpha)
        population.trees[i] = tree
    return population


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
) noexcept nogil:
    cdef float probability = rand_fraction()
    cdef Tree* tree = copy_tree(population.trees[index])
    cdef Tree* partner
    cdef Tree* child

    if probability < crossover_rate:
        partner = choose_different_tree(population, index)
        tree = crossover(tree, partner, max_depth=max_depth)
        free_tree(partner)
    elif probability < grow_rate:
        grow(tree, split_values=split_values, n_features=n_features, max_depth=max_depth)
    elif probability < prune_rate:
        prune_internal(tree)
    elif probability < mutate_split_rate:
        mutate_split_feature(tree, n_features=n_features, split_values=split_values)
    else:
        mutate_split_value(tree, split_values=split_values)

    return tree

cdef inline void insert_offspring(Forest* population, Forest* offspring, int i) noexcept nogil:
    """
    Compare offspring with parent at same index. 
    Keep the better one (or offspring if equal).
    """
    cdef Tree* parent = population.trees[i]
    cdef Tree* child = offspring.trees[i]

    # If child is better or equal, replace parent
    if child.fitness >= parent.fitness:
        free_tree(parent)
        population.trees[i] = child
        offspring.trees[i] = NULL  # Transfer ownership to population
    else:
        # Parent is better, keep parent and free child
        free_tree(child)
        offspring.trees[i] = NULL  # Already freed, set to NULL

cdef inline void fit_predict(
    float[:, :] X,
    int[:] y,
    Tree* tree,
    float[:] predictions,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
) noexcept nogil:
    reset_tree(tree)
    fit_tree(tree, X, y, n_samples)
    prune_illegal_nodes(tree, tree.root, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    predict_proba_tree(tree, X, predictions, n_samples)


cdef inline void evaluate(
    Tree* tree,
    object fitness_function,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    cnp.ndarray[cnp.float32_t, ndim=1] predictions,
    float alpha,
):
    cdef float fitness = fitness_function(y, predictions)
    tree.fitness = fitness
    tree.fitness -= alpha * tree.n_nodes

cdef inline void evaluate_max_profit(
    Tree* tree,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    cnp.ndarray[cnp.float32_t, ndim=1] predictions,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
    float alpha,
) noexcept:
    cdef float fitness = max_profit_score(
        y,
        predictions,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    tree.fitness = fitness
    tree.fitness -= alpha * tree.n_nodes

cdef inline bint stop_evolution(
    Tree* challenger, Tree** champion, int* stagnation_counter, float tolerance, int patience
) noexcept:
    if challenger.fitness > champion[0].fitness * (1.0 + tolerance):
        free_tree(champion[0])  # Always free the old champion
        champion[0] = challenger  # New champion takes ownership
        stagnation_counter[0] = 0
    else:
        free_tree(challenger)  # Free the losing challenger
        stagnation_counter[0] += 1

    return stagnation_counter[0] >= patience


cdef struct EvolutionResult:
    Tree* tree
    int n_generations


cdef EvolutionResult evolve_forest_stochastic(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    object fitness_function,
    int pop_size = 100,
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
    set_seed(random_state)

    cdef float[:, :] X_view = X
    cdef int[:] y_view = y
    cdef cnp.int32_t n_samples = <int>X.shape[0]
    cdef cnp.int32_t n_features = <int>X.shape[1]
    cdef int i, generation
    cdef SplitValues* split_values = compute_split_values(X)

    cdef Forest* population = initialize_population(
        pop_size=pop_size,
        n_features=n_features,
        split_values=split_values,
        max_depth=max_depth,
        X=X,
        y=y,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        alpha=alpha,
        fitness_function=fitness_function,
    )

    cdef int stagnation_counter = 0
    cdef Tree* parent
    cdef Tree* partner
    cdef Tree* child
    cdef Tree* gen_best_tree
    cdef Tree* best_tree = find_best_tree(population)
    cdef Forest* offspring = create_forest(pop_size)
    for i in range(pop_size):
        offspring.trees[i] = NULL
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions

    # set up the rates for various genetic operations
    cdef float probability = 0.0
    grow_rate = crossover_rate + grow_rate
    prune_rate = grow_rate + prune_rate
    mutate_split_rate = prune_rate + mutate_split_rate

    for generation in range(max_generations):
        for i in range(pop_size):
            offspring.trees[i] = evolve_tree(
                population=population,
                split_values=split_values,
                n_features=n_features,
                max_depth=max_depth,
                crossover_rate=crossover_rate,
                grow_rate=grow_rate,
                prune_rate=prune_rate,
                mutate_split_rate=mutate_split_rate,
                index=i,
            )

        for i in range(pop_size):
            insert_offspring(population, offspring, i)
            child = population.trees[i]
            fit_predict(X_view, y_view, child, predictions_view, n_samples, min_samples_split, min_samples_leaf)
            evaluate(child, fitness_function, y, predictions, alpha)

        gen_best_tree = find_best_tree(population)
        if stop_evolution(
            challenger=gen_best_tree,
            champion=&best_tree,
            stagnation_counter=&stagnation_counter,
            tolerance=tol,
            patience=patience,
        ):
            generation += 1
            break

    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    cdef EvolutionResult result
    result.tree = best_tree
    result.n_generations = generation + 1
    return result


cdef EvolutionResult evolve_forest_deterministic(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
    int pop_size = 100,
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
    set_seed(random_state)

    cdef float[:, :] X_view = X
    cdef int[:] y_view = y
    cdef cnp.int32_t n_samples = <int>X.shape[0]
    cdef cnp.int32_t n_features = <int>X.shape[1]
    cdef int i, generation
    cdef SplitValues* split_values = compute_split_values(X)

    cdef Forest* population = initialize_population_max_profit(
        pop_size=pop_size,
        n_features=n_features,
        split_values=split_values,
        max_depth=max_depth,
        X=X,
        y=y,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        alpha=alpha,
    )

    cdef int stagnation_counter = 0
    cdef Tree* parent
    cdef Tree* partner
    cdef Tree* child
    cdef Tree* gen_best_tree
    cdef Tree* best_tree = find_best_tree(population)
    cdef Forest* offspring = create_forest(pop_size)
    for i in range(pop_size):
        offspring.trees[i] = NULL
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions

    # set up the rates for various genetic operations
    cdef float probability = 0.0
    grow_rate = crossover_rate + grow_rate
    prune_rate = grow_rate + prune_rate
    mutate_split_rate = prune_rate + mutate_split_rate

    for generation in range(max_generations):
        for i in range(pop_size):
            offspring.trees[i] = evolve_tree(
                population=population,
                split_values=split_values,
                n_features=n_features,
                max_depth=max_depth,
                crossover_rate=crossover_rate,
                grow_rate=grow_rate,
                prune_rate=prune_rate,
                mutate_split_rate=mutate_split_rate,
                index=i,
            )

        for i in range(pop_size):
            insert_offspring(population, offspring, i)
            child = population.trees[i]
            fit_predict(X_view, y_view, child, predictions_view, n_samples, min_samples_split, min_samples_leaf)
            evaluate_max_profit(child, y, predictions, tp_benefit, tn_benefit, fp_cost, fn_cost, alpha)

        gen_best_tree = find_best_tree(population)
        if stop_evolution(
            challenger=gen_best_tree,
            champion=&best_tree,
            stagnation_counter=&stagnation_counter,
            tolerance=tol,
            patience=patience,
        ):
            generation += 1
            break

    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    cdef EvolutionResult result
    result.tree = best_tree
    result.n_generations = generation + 1
    return result
