# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs
from libc.stdlib cimport free, malloc
from cython.parallel cimport prange, threadid

from .tree cimport (Tree, SplitValues, create_tree, copy_tree, free_tree,
                    compute_split_values, free_split_values, reset_tree,
                    refit_tree, predict_proba_tree, split, prune_illegal_nodes)
from .forest cimport Forest, create_forest, free_forest, choose_different_tree
from .operators cimport count_nodes, crossover, grow, prune_internal, mutate_split_feature, mutate_split_value
from .random cimport RandState, rand_fraction, seed_rand
from .max_profit cimport Leaf, collect_leaves, max_profit_score
from libcpp.vector cimport vector


cdef Tree* find_best_tree(Forest* population) noexcept:
    if population is NULL or population.n_trees == 0:
        return NULL
    cdef int i
    cdef Tree* best_tree = population.trees[0]
    for i in range(1, population.n_trees):
        if population.trees[i].fitness > best_tree.fitness:
            best_tree = population.trees[i]
    return copy_tree(best_tree, with_samples=False)

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
# serially beforehand, so the fit does not depend on the number of threads. With cached samples,
# each thread sorts them in a scratch buffer of its own, allocated once per fit (a buffer this large
# is mapped afresh by every allocation on Windows).

cdef int** create_scratch(int n_threads, int n_samples) noexcept nogil:
    cdef int** scratch = <int**>malloc(n_threads * sizeof(int*))
    cdef int i
    for i in range(n_threads):
        scratch[i] = <int*>malloc(2 * <size_t>n_samples * sizeof(int))
    return scratch

cdef void free_scratch(int** scratch, int n_threads) noexcept nogil:
    if scratch is NULL:
        return
    cdef int i
    for i in range(n_threads):
        free(scratch[i])
    free(scratch)

cdef void fit_population(
    Forest* population,
    const float[:, ::1] X,
    const int[:] y,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
    int** scratch,
    int n_threads,
) noexcept nogil:
    """Fit every tree; ``scratch`` holds a buffer per thread to cache samples with, or is NULL."""
    cdef int i
    for i in prange(
        population.n_trees, schedule='dynamic', num_threads=n_threads, use_threads_if=n_threads > 1
    ):
        refit_tree(
            population.trees[i], X, y, n_samples, min_samples_split, min_samples_leaf,
            scratch is not NULL, scratch[threadid()] if scratch is not NULL else NULL,
        )

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
) noexcept nogil:
    """
    Fit every tree and set its fitness, which needs nothing but the tree's own leaf counts.

    ``scratch`` holds a buffer per thread to cache samples with, or is NULL.
    """
    cdef int i
    for i in prange(
        population.n_trees, schedule='dynamic', num_threads=n_threads, use_threads_if=n_threads > 1
    ):
        refit_tree(
            population.trees[i], X, y, n_samples, min_samples_split, min_samples_leaf,
            scratch is not NULL, scratch[threadid()] if scratch is not NULL else NULL,
        )
        evaluate_native(population.trees[i], fitness, alpha)

cdef void evaluate_population(
    Forest* population,
    const float[:, ::1] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int n_samples,
    object fitness_function,
    bint fitness_from_leaves,
    float alpha,
):
    """
    Set the fitness of every fitted tree with a Python fitness function, one tree at a time.

    The fitness function takes either the labels and every sample's prediction, or, with
    ``fitness_from_leaves``, each leaf's prediction and its numbers of positive and negative samples.
    """
    cdef int i
    if fitness_from_leaves:
        for i in range(population.n_trees):
            evaluate_leaves(population.trees[i], fitness_function, alpha)
        return

    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
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
    # The samples are copied when the offspring is refit, in parallel, and only if it is not refit whole.
    cdef Tree* tree = copy_tree(population.trees[index], with_samples=False)
    tree.samples_source = population.trees[index].samples
    cdef Tree* partner
    cdef Tree* child

    if probability < crossover_rate:
        # The partner is only read, so it need not be copied.
        partner = choose_different_tree(rng, population, index)
        tree = crossover(rng, tree, partner, max_depth=max_depth)
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

    if child.fitness >= parent.fitness:
        free_tree(parent)
        population.trees[i] = child
        offspring.trees[i] = NULL  # Transfer ownership to population
    else:
        free_tree(child)
        offspring.trees[i] = NULL  # Already freed, set to NULL

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

cdef inline void evaluate_leaves(Tree* tree, object fitness_function, float alpha):
    cdef vector[Leaf] leaves
    collect_leaves(tree.root, leaves)
    cdef Py_ssize_t n_leaves = leaves.size()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scores = np.empty(n_leaves, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] n_positive = np.empty(n_leaves, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] n_negative = np.empty(n_leaves, dtype=np.int64)
    cdef Py_ssize_t i
    for i in range(n_leaves):
        scores[i] = leaves[i].score
        n_positive[i] = leaves[i].n_positive
        n_negative[i] = leaves[i].n_negative
    cdef float fitness = fitness_function(scores, n_positive, n_negative)
    tree.fitness = fitness
    # Counted afresh: the variation operators and prune_illegal_nodes change the tree's shape.
    tree.n_nodes = count_nodes(tree.root)
    tree.fitness -= alpha * tree.n_nodes

cdef inline void evaluate_native(Tree* tree, const NativeFitness* fitness, float alpha) noexcept nogil:
    # Computed from the leaf counts that fitting left in the tree, not from per-sample predictions.
    cdef vector[Leaf] leaves
    cdef vector[double] scores
    cdef vector[long long] n_positive, n_negative
    cdef size_t i
    if fitness.expected_max_profit is NULL:
        tree.fitness = max_profit_score(
            tree.root,
            tp_benefit=fitness.tp_benefit,
            tn_benefit=fitness.tn_benefit,
            fp_cost=fitness.fp_cost,
            fn_cost=fitness.fn_cost,
        )
    else:
        collect_leaves(tree.root, leaves)
        scores.resize(leaves.size())
        n_positive.resize(leaves.size())
        n_negative.resize(leaves.size())
        for i in range(leaves.size()):
            scores[i] = leaves[i].score
            n_positive[i] = leaves[i].n_positive
            n_negative[i] = leaves[i].n_negative
        tree.fitness = <float>fitness.expected_max_profit(
            scores.data(),
            n_positive.data(),
            n_negative.data(),
            <Py_ssize_t>leaves.size(),
            fitness.coefficient_parts,
            fitness.n_powers,
            fitness.lower_bound,
            fitness.upper_bound,
            fitness.distribution,
            fitness.distribution_parameters,
        )
    # Counted afresh: the variation operators and prune_illegal_nodes change the tree's shape.
    tree.n_nodes = count_nodes(tree.root)
    tree.fitness -= alpha * tree.n_nodes

cdef inline bint stop_evolution(
    Tree* challenger, Tree** champion, int* stagnation_counter, float tolerance, int patience
) noexcept:
    # The tolerance is relative to the champion's magnitude, so the bar also rises when the fitness
    # is negative (e.g. costs only).
    if challenger.fitness > champion[0].fitness + tolerance * fabs(champion[0].fitness):
        free_tree(champion[0])
        champion[0] = challenger  # the champion takes ownership
        stagnation_counter[0] = 0
    else:
        free_tree(challenger)
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
    bint fitness_from_leaves = False,
    bint cache_samples = True,
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
    cdef int** scratch = create_scratch(n_threads, n_samples) if cache_samples else NULL
    fit_population(population, X_view, y_view, n_samples, min_samples_split, min_samples_leaf, scratch, n_threads)
    evaluate_population(population, X_view, y, n_samples, fitness_function, fitness_from_leaves, alpha)

    cdef int stagnation_counter = 0
    cdef Tree* parent
    cdef Tree* partner
    cdef Tree* child
    cdef Tree* gen_best_tree
    cdef Tree* best_tree = find_best_tree(population)
    cdef Forest* offspring = create_forest(pop_size)
    for i in range(pop_size):
        offspring.trees[i] = NULL

    # Cumulative rates for choosing a genetic operation.
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

        # Each offspring is fitted and scored before it competes with its parent: until then it
        # carries its parent's fitness, or NaN after a crossover.
        fit_population(offspring, X_view, y_view, n_samples, min_samples_split, min_samples_leaf, scratch, n_threads)
        evaluate_population(offspring, X_view, y, n_samples, fitness_function, fitness_from_leaves, alpha)
        for i in range(pop_size):
            insert_offspring(population, offspring, i)

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

    free_scratch(scratch, n_threads)
    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    cdef EvolutionResult result
    result.tree = best_tree
    result.n_generations = generation + 1
    return result


cdef EvolutionResult evolve_forest_native(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    NativeFitness fitness,
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
    bint cache_samples = True,
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
    cdef int** scratch = create_scratch(n_threads, n_samples) if cache_samples else NULL
    fit_population_native(
        population, X_view, y_view, n_samples, min_samples_split, min_samples_leaf,
        &fitness, alpha, scratch, n_threads,
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

    # Cumulative rates for choosing a genetic operation.
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

        # Each offspring is fitted and scored before it competes with its parent: until then it
        # carries its parent's fitness, or NaN after a crossover.
        fit_population_native(
            offspring, X_view, y_view, n_samples, min_samples_split, min_samples_leaf,
            &fitness, alpha, scratch, n_threads,
        )
        for i in range(pop_size):
            insert_offspring(population, offspring, i)

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

    free_scratch(scratch, n_threads)
    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    cdef EvolutionResult result
    result.tree = best_tree
    result.n_generations = generation + 1
    return result
