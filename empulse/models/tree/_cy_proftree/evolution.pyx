# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs
from libc.stdlib cimport free
from cython.parallel cimport prange

from .tree cimport (Tree, SplitValues, create_tree, copy_tree, free_tree,
                    compute_split_values, free_split_values, reset_tree,
                    fit_tree, predict_proba_tree, split, prune_illegal_nodes)
from .forest cimport Forest, create_forest, free_forest, choose_different_tree
from .operators cimport count_nodes, crossover, grow, prune_internal, mutate_split_feature, mutate_split_value
from .random cimport RandState, rand_fraction, seed_rand
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

cdef Forest* random_population(
    RandState* rng,
    int pop_size,
    int n_features,
    SplitValues* split_values,
    int max_depth,
) noexcept nogil:
    """Draw the initial population: trees of a single random split, not yet fitted."""
    cdef Forest* population = create_forest(pop_size)
    cdef int i
    for i in range(pop_size):
        population.trees[i] = create_tree()
        split(rng, population.trees[i].root, n_features, split_values, depth=0, max_depth=max_depth)
    return population


# Fitting the trees is the bulk of every generation and each tree is fitted independently, so it runs
# in parallel over the population. Only the variation operators draw random numbers, and they run
# serially beforehand, so the fit does not depend on the number of threads.

cdef void fit_population(
    Forest* population,
    const float[:, ::1] X,
    const int[:] y,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
    int n_threads,
) noexcept nogil:
    cdef int i
    for i in prange(
        population.n_trees, schedule='dynamic', num_threads=n_threads, use_threads_if=n_threads > 1
    ):
        refit(X, y, population.trees[i], n_samples, min_samples_split, min_samples_leaf)

cdef void fit_population_max_profit(
    Forest* population,
    const float[:, ::1] X,
    const int[:] y,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
    float alpha,
    int n_threads,
) noexcept nogil:
    """Fit every tree and set its fitness, which needs nothing but the tree's own leaf counts."""
    cdef int i
    for i in prange(
        population.n_trees, schedule='dynamic', num_threads=n_threads, use_threads_if=n_threads > 1
    ):
        refit(X, y, population.trees[i], n_samples, min_samples_split, min_samples_leaf)
        evaluate_max_profit(population.trees[i], tp_benefit, tn_benefit, fp_cost, fn_cost, alpha)

cdef void evaluate_population(
    Forest* population,
    const float[:, ::1] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int n_samples,
    object fitness_function,
    float alpha,
):
    """Set the fitness of every fitted tree with a Python fitness function, one tree at a time."""
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
    cdef int i
    for i in range(population.n_trees):
        predict_proba_tree(population.trees[i], X, predictions_view, n_samples)
        evaluate(population.trees[i], fitness_function, y, predictions, alpha)


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
) noexcept nogil:
    cdef float probability = rand_fraction(rng)
    cdef Tree* tree = copy_tree(population.trees[index])
    cdef Tree* partner
    cdef Tree* child

    if probability < crossover_rate:
        partner = choose_different_tree(rng, population, index)
        tree = crossover(rng, tree, partner, max_depth=max_depth)
        free_tree(partner)
    elif probability < grow_rate:
        grow(rng, tree, split_values=split_values, n_features=n_features, max_depth=max_depth)
    elif probability < prune_rate:
        prune_internal(rng, tree)
    elif probability < mutate_split_rate:
        mutate_split_feature(rng, tree, n_features=n_features, split_values=split_values)
    else:
        mutate_split_value(rng, tree, split_values=split_values)

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

cdef inline void refit(
    const float[:, ::1] X,
    const int[:] y,
    Tree* tree,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
) noexcept nogil:
    reset_tree(tree)
    fit_tree(tree, X, y, n_samples)
    prune_illegal_nodes(tree, tree.root, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)


cdef inline void evaluate(
    Tree* tree,
    object fitness_function,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    cnp.ndarray[cnp.float32_t, ndim=1] predictions,
    float alpha,
):
    cdef float fitness = fitness_function(y, predictions)
    tree.fitness = fitness
    # Counted afresh: the variation operators and prune_illegal_nodes change the tree's shape.
    tree.n_nodes = count_nodes(tree.root)
    tree.fitness -= alpha * tree.n_nodes

cdef inline void evaluate_max_profit(
    Tree* tree,
    float tp_benefit,
    float tn_benefit,
    float fp_cost,
    float fn_cost,
    float alpha,
) noexcept nogil:
    # Computed from the leaf counts that fitting left in the tree, not from per-sample predictions.
    cdef float fitness = max_profit_score(
        tree.root,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    tree.fitness = fitness
    # Counted afresh: the variation operators and prune_illegal_nodes change the tree's shape.
    tree.n_nodes = count_nodes(tree.root)
    tree.fitness -= alpha * tree.n_nodes

cdef inline bint stop_evolution(
    Tree* challenger, Tree** champion, int* stagnation_counter, float tolerance, int patience
) noexcept:
    # The tolerance is relative to the champion's magnitude. Scaling the fitness itself by
    # (1 + tolerance) lowers the bar when the fitness is negative (e.g. costs only), so an equal
    # or slightly worse challenger counted as an improvement and patience never ran out.
    if challenger.fitness > champion[0].fitness + tolerance * fabs(champion[0].fitness):
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
    int n_threads = 1,
):
    # The RNG state lives on this stack frame: nothing outside this fit can reach it, so two
    # concurrent fits neither interleave draws nor reseed one another.
    cdef RandState rng
    seed_rand(&rng, <unsigned int>random_state)

    cdef const float[:, ::1] X_view = X
    cdef const int[:] y_view = y
    cdef cnp.int32_t n_samples = <int>X.shape[0]
    cdef cnp.int32_t n_features = <int>X.shape[1]
    cdef int i, generation
    cdef SplitValues* split_values = compute_split_values(X)

    cdef Forest* population = random_population(&rng, pop_size, n_features, split_values, max_depth)
    fit_population(population, X_view, y_view, n_samples, min_samples_split, min_samples_leaf, n_threads)
    evaluate_population(population, X_view, y, n_samples, fitness_function, alpha)

    cdef int stagnation_counter = 0
    cdef Tree* parent
    cdef Tree* partner
    cdef Tree* child
    cdef Tree* gen_best_tree
    cdef Tree* best_tree = find_best_tree(population)
    cdef Forest* offspring = create_forest(pop_size)
    for i in range(pop_size):
        offspring.trees[i] = NULL

    # set up the rates for various genetic operations
    cdef float probability = 0.0
    grow_rate = crossover_rate + grow_rate
    prune_rate = grow_rate + prune_rate
    mutate_split_rate = prune_rate + mutate_split_rate

    for generation in range(max_generations):
        for i in range(pop_size):
            offspring.trees[i] = evolve_tree(
                &rng,
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
        fit_population(population, X_view, y_view, n_samples, min_samples_split, min_samples_leaf, n_threads)
        evaluate_population(population, X_view, y, n_samples, fitness_function, alpha)

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
    int n_threads = 1,
):
    # The RNG state lives on this stack frame: nothing outside this fit can reach it, so two
    # concurrent fits neither interleave draws nor reseed one another.
    cdef RandState rng
    seed_rand(&rng, <unsigned int>random_state)

    cdef const float[:, ::1] X_view = X
    cdef const int[:] y_view = y
    cdef cnp.int32_t n_samples = <int>X.shape[0]
    cdef cnp.int32_t n_features = <int>X.shape[1]
    cdef int i, generation
    cdef SplitValues* split_values = compute_split_values(X)

    cdef Forest* population = random_population(&rng, pop_size, n_features, split_values, max_depth)
    fit_population_max_profit(
        population, X_view, y_view, n_samples, min_samples_split, min_samples_leaf,
        tp_benefit, tn_benefit, fp_cost, fn_cost, alpha, n_threads,
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

    # set up the rates for various genetic operations
    cdef float probability = 0.0
    grow_rate = crossover_rate + grow_rate
    prune_rate = grow_rate + prune_rate
    mutate_split_rate = prune_rate + mutate_split_rate

    for generation in range(max_generations):
        for i in range(pop_size):
            offspring.trees[i] = evolve_tree(
                &rng,
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
        fit_population_max_profit(
            population, X_view, y_view, n_samples, min_samples_split, min_samples_leaf,
            tp_benefit, tn_benefit, fp_cost, fn_cost, alpha, n_threads,
        )

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
