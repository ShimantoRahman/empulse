import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, srand, rand, RAND_MAX
from libc.time cimport time
from libc.math cimport isnan, NAN

cdef int rand_int(int low, int high) noexcept nogil:
    """Return a random integer in [low, high)."""
    return low + rand() % (high - low)

cdef bint rand_bool() noexcept nogil:
    """Return a random boolean value."""
    return rand() % 2 == 0

cdef float rand_fraction() noexcept nogil:
    return rand() / (<float>RAND_MAX + 1.0)

cdef struct Node:
    Node* left
    Node* right
    Node* parent
    float split_value
    int feature_index
    int n_samples
    int n_positive_samples

cdef Node* create_node() noexcept nogil:
    cdef Node* node = <Node*>malloc(sizeof(Node))
    node.left = NULL
    node.right = NULL
    node.parent = NULL
    node.split_value = -1.0
    node.feature_index = -1
    node.n_samples = 0
    node.n_positive_samples = 0
    return node

cdef Node* copy_node(Node* node, Node* parent = NULL) noexcept nogil:
    if node is NULL:
        return NULL
    cdef Node* new_node = <Node*>malloc(sizeof(Node))
    new_node.split_value = node.split_value
    new_node.feature_index = node.feature_index
    new_node.n_samples = node.n_samples
    new_node.n_positive_samples = node.n_positive_samples
    new_node.parent = parent
    new_node.left = copy_node(node.left, new_node)
    new_node.right = copy_node(node.right, new_node)
    return new_node

cdef void free_node(Node* node) noexcept nogil:
    if node is NULL:
        return
    free_node(node.left)
    free_node(node.right)
    free(node)

cdef struct Tree:
    Node* root
    float fitness

cdef Tree* create_tree(bint with_root = True) noexcept nogil:
    cdef Tree* tree = <Tree*>malloc(sizeof(Tree))
    if with_root:
        tree.root = create_node()
    tree.fitness = -1.0
    return tree

cdef Tree* copy_tree(Tree* tree) noexcept nogil:
    cdef Tree* new_tree = <Tree*>malloc(sizeof(Tree))
    new_tree.root = copy_node(tree.root, NULL)
    new_tree.fitness = tree.fitness
    return new_tree

cdef void free_tree(Tree* tree) noexcept nogil:
    if tree is NULL:
        return
    free_node(tree.root)
    free(tree)

cdef bint is_leaf(Node* node) noexcept nogil:
    return node.left is NULL and node.right is NULL

cdef float node_probability(Node* node) noexcept nogil:
    if node.n_samples == 0:
        return 0.5  # fallback when unseen
    return <float>node.n_positive_samples / <float>node.n_samples

cdef void update_node_stats(Node* node, int y) noexcept nogil:
    node.n_samples += 1
    if y == 1:
        node.n_positive_samples += 1

cdef Node* get_leaf(Node* start_node, float[:] x) noexcept nogil:
    """
    Traverse the tree and return the leaf for a sample.
    """
    cdef Node* node = start_node
    while not is_leaf(node):
        if x[node.feature_index] <= node.split_value:
            node = node.left
        else:
            node = node.right
    return node


cdef void fit_tree(Tree* tree, float[:, :] X, int[:] y, int n_samples) noexcept nogil:
    """
    Push training data down the tree to update leaf probabilities.
    """
    cdef float[:] x_i
    cdef int y_i
    cdef Node* leaf
    for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]
        leaf = get_leaf(tree.root, x_i)
        update_node_stats(leaf, y_i)

cdef void predict_proba_tree(Tree* tree, float[:, :] X, float[:] probabilities, int n_samples) noexcept nogil:
    """
    Predict probabilities for all samples in X.
    """
    for i in range(n_samples):
        leaf = get_leaf(tree.root, X[i])
        probabilities[i] = node_probability(leaf)

cdef void predict_labels_tree(Tree* tree, float[:, :] X, float[:] probabilities, int n_samples):
    predict_proba_tree(tree, X, probabilities, n_samples)
    for i in range(n_samples):
        probabilities[i] = 1 if probabilities[i] >= 0.5 else 0

cdef struct SplitValues:
    float **values  # array of pointers to arrays
    int *lengths     # number of split values per feature
    int n_features

cdef SplitValues* compute_split_values(cnp.ndarray[cnp.float32_t, ndim=2] X) noexcept:
    cdef int n_features = <int>X.shape[1]
    cdef SplitValues* sv = <SplitValues*>malloc(sizeof(SplitValues))
    sv.n_features = n_features
    sv.values = <float **>malloc(n_features * sizeof(float *))
    sv.lengths = <int *>malloc(n_features * sizeof(int))

    cdef int j, n_unique, i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vals
    for j in range(n_features):
        vals = np.unique(X[:, j])
        n_unique = <int>vals.shape[0]
        sv.lengths[j] = n_unique
        sv.values[j] = <float *>malloc(n_unique * sizeof(float))
        for i in range(n_unique):
            sv.values[j][i] = <float>vals[i]
    return sv

cdef void free_split_values(SplitValues* sv) noexcept nogil:
    cdef int j
    for j in range(sv.n_features):
        free(sv.values[j])
    free(sv.values)
    free(sv.lengths)
    free(sv)

cdef void grow_subtree(
    Node* node,
    int n_features,
    SplitValues* split_values,
    int depth = 0,
    int max_depth = -1,
) noexcept nogil:
    """
    Randomly generates the node and its children.

    Args:
        depth (int, optional): Current depth of the tree.
        max_depth (int, optional): Maximum depth of the tree.
        random (Random, optional): Random number generator.
        att_indexes (numpy.ndarray, optional): Attribute indexes.
        att_values (dict, optional): Attribute values.
        class_count (int, optional): Number of classes.

    Returns
    -------
        Node: Randomly generated node with children.
    """
    if max_depth == -1:
        max_depth = depth

    # if it's the root, first level or 50/50 chance of building new children.
    # Must be below maximal depth.
    cdef int split_value_index
    if depth <= 1 or (rand_bool() and depth < max_depth):
        node.feature_index = rand_int(0, n_features)
        split_value_index = rand_int(0, split_values.lengths[node.feature_index] - 1)  # TODO: check -1
        node.split_value = split_values.values[node.feature_index][split_value_index]
        node.left = create_node()
        node.left.parent = node
        node.right = create_node()
        node.right.parent = node
        grow_subtree(
            node.left,
            depth=depth + 1,
            max_depth=max_depth,
            n_features=n_features,
            split_values=split_values,
        )
        grow_subtree(
            node.right,
            depth=depth + 1,
            max_depth=max_depth,
            n_features=n_features,
            split_values=split_values,
        )

cdef Node* random_subnode(Node* root) noexcept nogil:
    """
    Pick a random node from the tree using a random walk.
    Each node has a chance to be picked, but not uniform.
    """
    cdef Node* node = root
    while True:
        # With probability 1/3, stop at current node
        # Otherwise, randomly pick a child to continue
        if is_leaf(node) or rand_int(0, 3) == 0:
            return node
        # Randomly choose left or right child (if both exist)
        if node.left is not NULL and node.right is not NULL:
            if rand_bool():
                node = node.left
            else:
                node = node.right
        elif node.left is not NULL:
            node = node.left
        elif node.right is not NULL:
            node = node.right
        else:
            return node  # Should not happen, but safe fallback

cdef Node* random_subnode_with_depth(Node* root, int* out_depth) noexcept nogil:
    """
    Pick a random node from the tree using a random walk.
    Each node has a chance to be picked, but not uniform.
    Also returns the depth at which the node is found via out_depth.
    """
    cdef Node* node = root
    cdef int depth = 0
    while True:
        if is_leaf(node) or rand_int(0, 3) == 0:
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        if node.left is not NULL and node.right is not NULL:
            if rand_bool():
                node = node.left
            else:
                node = node.right
        elif node.left is not NULL:
            node = node.left
        elif node.right is not NULL:
            node = node.right
        else:
            if out_depth is not NULL:
                out_depth[0] = depth
            return node
        depth += 1

cdef struct Children:
    Tree* son
    Tree* daughter

cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil:
    """Recursively prune subtree to not exceed max_depth."""
    if node is NULL or max_depth < 0:
        return

    if max_depth == 0:
        # Convert to leaf - free children and set to NULL
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
    else:
        # Recursively prune children
        prune_subtree_at_depth(node.left, max_depth - 1)
        prune_subtree_at_depth(node.right, max_depth - 1)

cdef Children* crossover(Tree* tree1, Tree* tree2, int max_depth) noexcept nogil:
    """Perform subtree crossover between two trees with depth constraints."""
    cdef Tree* tree1_copy = copy_tree(tree1)
    cdef Tree* tree2_copy = copy_tree(tree2)

    cdef int depth1, depth2
    cdef Node* node1 = random_subnode_with_depth(tree1_copy.root, &depth1)
    cdef Node* node2 = random_subnode_with_depth(tree2_copy.root, &depth2)

    cdef Node* parent1 = node1.parent
    cdef Node* parent2 = node2.parent

    # Calculate remaining depth budget for each swap
    cdef int remaining_depth1 = max_depth - depth2  # depth budget at node1's position
    cdef int remaining_depth2 = max_depth - depth1  # depth budget at node2's position

    # Prune subtrees if they exceed depth limits
    if remaining_depth1 >= 0:
        prune_subtree_at_depth(node2, remaining_depth1)
    else:
        # Convert to leaf if no depth budget
        free_node(node2.left)
        free_node(node2.right)
        node2.left = NULL
        node2.right = NULL

    if remaining_depth2 >= 0:
        prune_subtree_at_depth(node1, remaining_depth2)
    else:
        # Convert to leaf if no depth budget
        free_node(node1.left)
        free_node(node1.right)
        node1.left = NULL
        node1.right = NULL

    # Swap the (possibly pruned) subtrees
    if parent1 is NULL:
        tree1_copy.root = node2
        node2.parent = NULL
    else:
        if parent1.left is node1:
            parent1.left = node2
        else:
            parent1.right = node2
        node2.parent = parent1

    if parent2 is NULL:
        tree2_copy.root = node1
        node1.parent = NULL
    else:
        if parent2.left is node2:
            parent2.left = node1
        else:
            parent2.right = node1
        node1.parent = parent2

    cdef Children* children = <Children*>malloc(sizeof(Children))
    children.son = tree1_copy
    children.daughter = tree2_copy
    children.son.fitness = NAN
    children.daughter.fitness = NAN
    return children

cdef void mutate(
        Tree* tree,
        int n_features,
        SplitValues* split_values,
        int max_depth,
) noexcept nogil:
    """
    Perform mutation on the tree:
    - If leaf: grow subtree.
    - If internal node: modify feature/split or replace subtree.
    """
    # set NULL fitness to force re-evaluation
    tree.fitness = NAN

    cdef int depth
    node = random_subnode_with_depth(tree.root, &depth)
    if is_leaf(node):
        grow_subtree(
            node=node,
            n_features=n_features,
            split_values=split_values,
            depth=depth,
            max_depth=max_depth,
        )
    else:
        choice = rand_int(0, 3)
        if choice == 0: # change both feature and split value
            node.feature_index = rand_int(0, n_features)
            idx = rand_int(0, split_values.lengths[node.feature_index])
            node.split_value = split_values.values[node.feature_index][idx]
        elif choice == 1: # only change split value
            idx = rand_int(0, split_values.lengths[node.feature_index])
            node.split_value = split_values.values[node.feature_index][idx]
        else:
            grow_subtree(
                node=node,
                n_features=n_features,
                split_values=split_values,
                depth=depth,
                max_depth=max_depth,
            )

cdef struct Forest:
    Tree** trees
    int n_trees

cdef Forest* create_forest(int n_trees) noexcept nogil:
    cdef Forest* forest = <Forest*>malloc(sizeof(Forest))
    forest.n_trees = n_trees
    forest.trees = <Tree**>malloc(n_trees * sizeof(Tree*))
    return forest

cdef void free_forest(Forest* forest) noexcept nogil:
    if forest is NULL:
        return
    cdef int i
    for i in range(forest.n_trees):
        free_tree(forest.trees[i])
    free(forest.trees)
    free(forest)

cdef Tree* tournament_selection(Forest* population, int tournament_size) noexcept:
    """
    Select one tree by tournament.
    Population is assumed to be sorted by fitness in descending order.
    """
    cdef int n = population.n_trees
    cdef int i, j, index
    cdef int selected
    cdef int[:] indices = np.empty(tournament_size, dtype=np.int32)  # TODO: make this an argument where to store indices, so we don't allocate every time
    cdef bint is_duplicate

    for i in range(tournament_size):
        while True:
            index = rand() % n
            is_duplicate = False
            for j in range(i):
                if indices[j] == index:
                    is_duplicate = True
                    break
            if not is_duplicate:
                indices[i] = index
                break

    cdef int best_index = indices[0]
    for i in range(1, tournament_size):
        if indices[i] < best_index:
            best_index = indices[i]
    return population.trees[best_index]

# cdef void quicksort_indices_by_fitness(int* indices, float* fitness, int low, int high) noexcept nogil:
#     """Quicksort indices array based on fitness values (descending order)."""
#     if low < high:
#         cdef int pi = partition_indices(indices, fitness, low, high)
#         quicksort_indices_by_fitness(indices, fitness, low, pi - 1)
#         quicksort_indices_by_fitness(indices, fitness, pi + 1, high)
#
# cdef int partition_indices(int* indices, float* fitness, int low, int high) noexcept nogil:
#     """Partition for quicksort - pivot is fitness[indices[high]]."""
#     cdef float pivot = fitness[indices[high]]
#     cdef int i = low - 1
#     cdef int j, temp
#
#     for j in range(low, high):
#         # For descending order: if fitness[indices[j]] >= pivot
#         if fitness[indices[j]] >= pivot:
#             i += 1
#             # Swap indices[i] and indices[j]
#             temp = indices[i]
#             indices[i] = indices[j]
#             indices[j] = temp
#
#     # Swap indices[i+1] and indices[high]
#     temp = indices[i + 1]
#     indices[i + 1] = indices[high]
#     indices[high] = temp
#
#     return i + 1
#
# cdef void sort_population_by_fitness(Forest* population, int* sorted_indices, float* fitness_buffer) noexcept nogil:
#     """Sort population indices by fitness in descending order."""
#     cdef int i
#     cdef int n = population.n_trees
#
#     # Initialize indices array and copy fitness values
#     for i in range(n):
#         sorted_indices[i] = i
#         fitness_buffer[i] = population.trees[i].fitness
#
#     # Sort indices based on fitness values
#     quicksort_indices_by_fitness(sorted_indices, fitness_buffer, 0, n - 1)

cdef Forest* sort_population(Forest* population, Forest* sorted_population, float[:] fitness_scores) noexcept:
    cdef int i
    for i in range(population.n_trees):
        fitness_scores[i] = population.trees[i].fitness
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sorted_indices = np.argsort(fitness_scores)[::-1]
    for i in range(population.n_trees):
        sorted_population.trees[i] = population.trees[sorted_indices[i]]
    free(population.trees)
    population.trees = sorted_population.trees
    sorted_population.trees = <Tree**>malloc(population.n_trees * sizeof(Tree*))

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
    cdef int i
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
    cdef int i = 0
    cdef int j = 0
    cdef SplitValues* split_values = compute_split_values(X)
    n_elites = min(n_elites, pop_size)  # make sure we don't have more elites than population size
    tournament_size = min(tournament_size, pop_size)  # same for tournament size

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

    # 3. evolutionary loop
    cdef float gen_best
    cdef float best_fitness = population.trees[0].fitness
    cdef int stagnation_counter = 0
    # cdef Forest* parents
    cdef Tree* mother
    cdef Tree* father
    cdef Tree* tree
    cdef int offspring_needed = pop_size - n_elites
    cdef Forest* offspring = create_forest(offspring_needed)
    cdef Children* children
    cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
    cdef float[:] predictions_view = predictions
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

        # evaluate fitness of new population (elites can be skipped)
        for i in range(offspring_needed):
            free_tree(population.trees[i + n_elites])
            population.trees[i + n_elites] = offspring.trees[i]
            offspring.trees[i] = NULL # transfer ownership to population

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

        # check improvement
        gen_best = population.trees[0].fitness
        if gen_best > best_fitness * (1.0 + tol):
            best_fitness = gen_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= patience:
            break


    # 4. keep best tree for inference
    best_tree = copy_tree(population.trees[0])  # copy to avoid freeing it

    free_split_values(split_values)
    free_forest(population)
    # offspring trees have been moved to population, so only pointers need to be freed
    free(offspring.trees)
    free(offspring)
    # same here
    free(sorted_population.trees)
    free(sorted_population)
    return best_tree

cdef class EvolutionaryTree:
    cdef Tree* tree

    def __cinit__(self):
        self.tree = NULL

    def fit(self, cnp.ndarray[cnp.float32_t, ndim=2] X, cnp.ndarray[cnp.int32_t, ndim=1] y,
            int pop_size=100,
            int tournament_size=3,
            int n_elites=1,
            float crossover_rate=0.9,
            float mutation_rate=0.1,
            int max_depth=10,
            int max_generations=50,
            int patience=5,
            float tol=1e-3,
            int random_state=-1):
        if self.tree is not NULL:
            free_tree(self.tree)
        self.tree = find_best_tree(
            X,
            y,
            pop_size=pop_size,
            tournament_size=tournament_size,
            n_elites=n_elites,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_depth=max_depth,
            max_generations=max_generations,
            patience=patience,
            tol=tol,
            random_state=random_state
        )

    def predict_proba(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        if self.tree is NULL:
            raise ValueError("The model has not been fitted yet.")
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] probabilities = np.empty(n_samples, dtype=np.float32)
        cdef float[:] probabilities_view = probabilities
        predict_proba_tree(self.tree, X, probabilities_view, n_samples)
        return probabilities

    def predict(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        if self.tree is NULL:
            raise ValueError("The model has not been fitted yet.")
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
        cdef float[:] predictions_view = predictions
        predict_labels_tree(self.tree, X, predictions_view, n_samples)
        return predictions

    def __dealloc__(self):
        if self.tree is not NULL:
            free_tree(self.tree)

cpdef float test(cnp.ndarray[cnp.float32_t, ndim=2] X, cnp.ndarray[cnp.int32_t, ndim=1] y, int pop_size, int max_generations):
    cdef Tree* tree = find_best_tree(
        X,
        y,
        pop_size=pop_size,
        tournament_size=3,
        n_elites=2,
        crossover_rate=0.9,
        mutation_rate=0.1,
        max_depth=5,
        patience=5,
        tol=1e-3,
        max_generations=max_generations
    )
    cdef float fitness = tree.fitness
    free_tree(tree)
    return fitness
