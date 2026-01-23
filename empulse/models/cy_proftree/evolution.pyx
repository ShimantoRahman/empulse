import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc, srand
from libc.time cimport time
from libc.math cimport isnan

from .tree cimport (Tree, SplitValues, create_tree, copy_tree, free_tree,
                    compute_split_values, free_split_values, reset_tree,
                    fit_tree, predict_proba_tree, split, prune_illegal_nodes)
from .forest cimport Forest, create_forest, free_forest, sort_population
from .operators cimport tournament_selection, crossover, evolve
from .random cimport rand_fraction, set_seed


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
    cdef float accuracy
    for i in range(pop_size):
        tree = create_tree()
        split(tree.root, n_features, split_values, depth=0, max_depth=max_depth)
        fit_tree(tree, X, y, n_samples)
        predict_proba_tree(tree, X, predictions_view, n_samples)
        evaluate(tree, fitness_function, y, predictions, alpha)
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
    float mutate_split_rate
):

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
        node = random_leaf_node(tree.root, &depth)
        child = mutate_split_feature(parent, split_values, n_features, max_depth)
    else:
        child = mutate_split_value(parent, split_values)

    return child

cdef inline void insert_offspring(Forest* population, Forest* offspring, int i):
    free_tree(population.trees[i])
    population.trees[i] = offspring.trees[i]
    offspring.trees[i] = NULL

cdef inline void fit_predict(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    Tree* tree,
    float[:] predictions,
    int n_samples,
    int min_samples_split,
    int min_samples_leaf,
):
    reset_tree(tree)
    fit_tree(tree, X, y, n_samples)
    prune_illegal_nodes(tree.root, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    predict_proba_tree(tree, X, predictions, n_samples)


cdef inline void evaluate(
    Tree* tree,
    object fitness_function,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    cnp.ndarray[cnp.float32_t, ndim=1] predictions,
    float alpha,
):
    tree.fitness = fitness_function(y, predictions)
    tree.fitness += alpha * tree.size


cdef inline bool stop_evolution(
    Tree* challenger, Tree* champion, int* stagnation_counter, float tolerance, int patience
):
    if challenger.fitness > champion.fitness * (1.0 + tolerance):
        free_tree(champion)
        champion = challenger
        stagnation_counter[0] = 0
    else:
        stagnation_counter[0] += 1

    return stagnation_counter[0] >= patience

cpdef Tree* evolve_forest_stochastic(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    object fitness_function,
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
    set_seed(random_state)

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
    # mutate_value_rate = mutate_split_rate + mutate_value_rate

    for _ in range(max_generations):
        for i in range(pop_size):
            offspring.trees[i] = evolve_tree(
                population=population,
                split_values=split_values,
                n_features=n_features,
                max_depth=max_depth,
                crossover_rate=crossover_rate,
                prune_rate=prune_rate,
                mutate_split_rate=mutate_split_rate
            )

        for i in range(pop_size):
            insert_offspring(population, offspring, i)
            child = population.trees[i]
            fit_predict(X, y, child, predictions_view, n_samples, min_samples_split, min_samples_leaf)
            evaluate(child, fitness_function, y, predictions, alpha)

        gen_best_tree = find_best_tree(population)
        if stop_evolution(
            challenger=gen_best_tree,
            champion=best_tree,
            stagnation_counter=&stagnation_counter,
            tolerance=tol,
            patience=patience,
        ):
            break

    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    return best_tree
