import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc, srand
from libc.time cimport time
from libc.math cimport isnan

from .tree cimport (Tree, SplitValues, create_tree, copy_tree, free_tree,
                    compute_split_values, free_split_values, grow_subtree,
                    fit_tree, predict_labels_tree)
from .forest cimport Forest, create_forest, free_forest, tournament_selection, sort_population
from .operators cimport Children, crossover, mutate
from .random cimport rand_fraction

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

cdef Tree* find_best_tree(
    cnp.ndarray[cnp.float32_t, ndim=2] X,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    int pop_size = 100,
    int tournament_size = 3,
    int n_elites = 1,
    float crossover_rate = 0.9,
    float mutation_rate = 0.1,
    int max_depth = 10,
    int max_generations = 50,
    int patience = 5,
    float tol = 1e-3,
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
    n_elites = min(n_elites, pop_size)
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
    cdef Forest* sorted_population = create_forest(pop_size)
    sort_population(population, sorted_population, fitness_scores_view)

    cdef float gen_best
    cdef float best_fitness = population.trees[0].fitness
    cdef int stagnation_counter = 0
    cdef Tree* mother
    cdef Tree* father
    cdef Tree* tree
    cdef int offspring_needed = pop_size - n_elites
    cdef Forest* offspring = create_forest(offspring_needed)
    cdef Children* children
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
    cdef float accuracy

    for _ in range(max_generations):
        j = 0
        while j < offspring_needed:
            mother = tournament_selection(population, tournament_size)
            father = tournament_selection(population, tournament_size)

            if rand_fraction() < crossover_rate:
                children = crossover(mother, father, max_depth)
            else:
                children = <Children*>malloc(sizeof(Children))
                children.son = copy_tree(mother)
                children.daughter = copy_tree(father)

            if rand_fraction() < mutation_rate:
                mutate(
                    tree=children.son,
                    n_features=n_features,
                    split_values=split_values,
                    max_depth=max_depth,
                )
            if rand_fraction() < mutation_rate:
                mutate(
                    tree=children.daughter,
                    n_features=n_features,
                    split_values=split_values,
                    max_depth=max_depth,
                )

            offspring.trees[j] = children.son
            if j + 1 < offspring_needed:
                offspring.trees[j + 1] = children.daughter
            j += 2
            free(children)

        for i in range(offspring_needed):
            free_tree(population.trees[i + n_elites])
            population.trees[i + n_elites] = offspring.trees[i]
            offspring.trees[i] = NULL

            tree = population.trees[i + n_elites]
            fit_tree(tree, X, y, n_samples)
            predict_labels_tree(tree, X, predictions_view, n_samples)

            if isnan(tree.fitness):
                accuracy = 0.0
                for j in range(n_samples):
                    if predictions_view[j] == y[j]:
                        accuracy += 1
                accuracy /= n_samples
                tree.fitness = accuracy

        sort_population(population, sorted_population, fitness_scores_view)

        gen_best = population.trees[0].fitness
        if gen_best > best_fitness * (1.0 + tol):
            best_fitness = gen_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= patience:
            break

    cdef Tree* best_tree = copy_tree(population.trees[0])

    free_split_values(split_values)
    free_forest(population)
    free(offspring.trees)
    free(offspring)
    free(sorted_population.trees)
    free(sorted_population)
    return best_tree