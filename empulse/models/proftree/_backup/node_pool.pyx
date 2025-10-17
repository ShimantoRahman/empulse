import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, srand, rand, RAND_MAX
from libc.time cimport time
from libc.math cimport isnan, NAN
from libc.string cimport memset

cdef int rand_int(int low, int high) noexcept nogil:
    """Return a random integer in [low, high)."""
    return low + rand() % (high - low)

cdef bint rand_bool() noexcept nogil:
    """Return a random boolean value."""
    return rand() % 2 == 0

cdef float rand_fraction() noexcept nogil:
    return rand() / (<float>RAND_MAX + 1.0)

cdef struct NodePool:
    Node* nodes          # Array of all nodes
    int* free_list       # Stack of free node indices
    int capacity         # Total number of nodes that can be stored
    int used_count       # Number of nodes currently in use
    int free_top         # Top of the free list stack (-1 if empty)

cdef NodePool* create_node_pool(int capacity) noexcept nogil:
    """Create a node pool with specified capacity."""
    cdef NodePool* pool = <NodePool*>malloc(sizeof(NodePool))
    if pool is NULL:
        return NULL

    pool.capacity = capacity
    pool.used_count = 0
    pool.free_top = capacity - 1

    # Allocate node array
    pool.nodes = <Node*>malloc(capacity * sizeof(Node))
    if pool.nodes is NULL:
        free(pool)
        return NULL

    # Allocate and initialize free list
    pool.free_list = <int*>malloc(capacity * sizeof(int))
    if pool.free_list is NULL:
        free(pool.nodes)
        free(pool)
        return NULL

    # Initialize free list with all indices
    cdef int i
    for i in range(capacity):
        pool.free_list[i] = i
        # Initialize nodes to safe defaults
        pool.nodes[i].split_value = -1.0
        pool.nodes[i].feature_index = -1

    return pool

cdef void free_node_pool(NodePool* pool) noexcept nogil:
    """Free the entire node pool."""
    if pool is NULL:
        return
    free(pool.nodes)
    free(pool.free_list)
    free(pool)

cdef int allocate_node(NodePool* pool) noexcept nogil:
    """Allocate a node from the pool. Returns node index or -1 if failed."""
    if pool is NULL or pool.free_top < 0:
        return -1  # Pool full or invalid

    # Pop from free list
    cdef int node_index = pool.free_list[pool.free_top]
    pool.free_top -= 1
    pool.used_count += 1

    # Reset node to default state
    pool.nodes[node_index].split_value = -1.0
    pool.nodes[node_index].feature_index = -1

    return node_index

cdef void deallocate_node(NodePool* pool, int node_index) noexcept nogil:
    """Return a node to the pool for reuse."""
    if pool is NULL or node_index < 0 or node_index >= pool.capacity:
        return
    if pool.free_top >= pool.capacity - 1:
        return  # Free list full (shouldn't happen)

    # Push to free list
    pool.free_top += 1
    pool.free_list[pool.free_top] = node_index
    pool.used_count -= 1

    # Clear node data for safety
    pool.nodes[node_index].split_value = -1.0
    pool.nodes[node_index].feature_index = -1

cdef Node* get_node(NodePool* pool, int node_index) noexcept nogil:
    """Get pointer to node at given index."""
    if pool is NULL or node_index < 0 or node_index >= pool.capacity:
        return NULL
    return &pool.nodes[node_index]

cdef bint is_pool_full(NodePool* pool) noexcept nogil:
    """Check if pool is full."""
    return pool is NULL or pool.free_top < 0

cdef int get_pool_usage(NodePool* pool) noexcept nogil:
    """Get number of nodes currently in use."""
    if pool is NULL:
        return 0
    return pool.used_count

cdef struct Node:
    float split_value
    int feature_index

cdef struct Tree:
    int* node_indices
    int* n_samples  # how many samples were seen in every nodes
    int* n_positive_samples  # how many positive samples were seen in every node
    int depth
    float fitness

cdef Tree* create_tree(int max_nodes) noexcept nogil:
    cdef Tree* tree = <Tree*>malloc(sizeof(Tree))
    if tree is NULL:
        return NULL

    tree.node_indices = <int*>malloc(max_nodes * sizeof(int))
    if tree.node_indices is NULL:
        free(tree)
        return NULL

    # Initialize all indices to -1 (no node)
    memset(tree.node_indices, -1, max_nodes * sizeof(int))

    tree.n_samples = <int*>malloc(max_nodes * sizeof(int))
    tree.n_positive_samples = <int*>malloc(max_nodes * sizeof(int))

    if tree.n_samples is NULL or tree.n_positive_samples is NULL:
        free(tree.node_indices)
        free(tree.n_samples)
        free(tree.n_positive_samples)
        free(tree)
        return NULL

    memset(tree.n_samples, 0, max_nodes * sizeof(int))
    memset(tree.n_positive_samples, 0, max_nodes * sizeof(int))

    tree.depth = 0
    tree.fitness = -1.0
    return tree

cdef void free_tree_nodes(NodePool* pool, Tree* tree, int max_nodes) noexcept nogil:
    """Free all nodes used by a tree back to the pool."""
    if tree is NULL or pool is NULL:
        return

    cdef int i
    for i in range(max_nodes):
        if tree.node_indices[i] != -1:
            deallocate_node(pool, tree.node_indices[i])
            tree.node_indices[i] = -1

cdef Tree* copy_tree_with_pool(NodePool* pool, Tree* source, int max_nodes) noexcept nogil:
    """Copy a tree using the node pool."""
    if source is NULL or pool is NULL:
        return NULL

    cdef Tree* new_tree = create_tree(max_nodes)
    if new_tree is NULL:
        return NULL

    cdef int i, new_node_index
    cdef Node* source_node
    cdef Node* new_node

    # Copy all used nodes
    for i in range(max_nodes):
        if source.node_indices[i] != -1:
            new_node_index = allocate_node(pool)
            if new_node_index == -1:
                # Pool full, cleanup and fail
                free_tree_nodes(pool, new_tree, max_nodes)
                free_tree(new_tree)
                return NULL

            new_tree.node_indices[i] = new_node_index
            source_node = get_node(pool, source.node_indices[i])
            new_node = get_node(pool, new_node_index)

            # Copy node data
            new_node.split_value = source_node.split_value
            new_node.feature_index = source_node.feature_index

        # Copy sample counts
        new_tree.n_samples[i] = source.n_samples[i]
        new_tree.n_positive_samples[i] = source.n_positive_samples[i]

    new_tree.depth = source.depth
    new_tree.fitness = source.fitness
    return new_tree

cdef void free_tree(Tree* tree) noexcept nogil:
    if tree is NULL:
        return
    free(tree.node_indices)
    free(tree.n_samples)
    free(tree.n_positive_samples)
    free(tree)

cdef inline int left_child_index(int node_index) noexcept nogil:
    return 2 * node_index

cdef inline int right_child_index(int node_index) noexcept nogil:
    return 2 * node_index + 1

cdef inline bint is_leaf(int* node_indices, int node_index) noexcept nogil:
    return node_indices[left_child_index(node_index)] == -1 and node_indices[right_child_index(node_index)] == -1

cdef inline float node_probability(int* n_samples_array, int* n_positive_samples_array, int node_index) noexcept nogil:
    cdef int n_samples = n_samples_array[node_index]
    if n_samples == 0:
        return 0.5  # fallback when unseen
    return <float>n_positive_samples_array[node_index] / <float>n_samples

cdef inline void visit_node(int* n_samples_array, int* n_positive_samples_array, int node_index, int y) noexcept nogil:
    n_samples_array[node_index] += 1
    if y == 1:
        n_positive_samples_array[node_index] += 1

cdef int get_leaf_index(NodePool* pool, int* node_indices, float[:] x) noexcept nogil:
    """Traverse the tree and return the leaf index for a sample."""
    cdef int current_index = 0  # Start at root
    cdef Node* node

    while not is_leaf(node_indices, current_index):
        if node_indices[current_index] == -1:
            break  # Safety check

        node = get_node(pool, node_indices[current_index])
        if node is NULL:
            break

        if x[node.feature_index] <= node.split_value:
            current_index = left_child_index(current_index)
        else:
            current_index = right_child_index(current_index)

    return current_index

cdef void fit_tree(NodePool* node_pool, Tree* tree, float[:, :] X, int[:] y, int n_samples) noexcept nogil:
    """
    Push training data down the tree to update leaf probabilities.
    """
    cdef float[:] x_i
    cdef int y_i
    cdef Node* leaf
    for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]
        leaf_index = get_leaf_index(node_pool, tree.node_indices, x_i)
        visit_node(tree.n_samples, tree.n_positive_samples, leaf_index, y_i)

cdef void predict_proba_tree(NodePool* node_pool, Tree* tree, float[:, :] X, float[:] probabilities, int n_samples) noexcept nogil:
    """
    Predict probabilities for all samples in X.
    """
    for i in range(n_samples):
        leaf_index = get_leaf_index(node_pool, tree.node_indices, X[i])
        probabilities[i] = node_probability(tree.n_samples, tree.n_positive_samples, leaf_index)

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
        NodePool* pool,
        Tree* tree,
        int tree_index,
        int n_features,
        SplitValues* split_values,
        int depth = 0,
        int max_depth = -1,
        int max_nodes = 1024,
) noexcept nogil:
    """Grow subtree using node pool."""
    if pool is NULL or tree is NULL:
        return

    if max_depth == -1:
        max_depth = depth

    if depth > tree.depth:
        tree.depth = depth

    # Allocate node if not already allocated
    if tree.node_indices[tree_index] == -1:
        tree.node_indices[tree_index] = allocate_node(pool)
        if tree.node_indices[tree_index] == -1:
            return  # Pool full

    cdef Node* node = get_node(pool, tree.node_indices[tree_index])
    if node is NULL:
        return

    # Decide whether to create children
    if depth <= 1 or (rand_bool() and depth < max_depth):
        cdef int left_idx = left_child_index(tree_index)
        cdef int right_idx = right_child_index(tree_index)

        # Check bounds
        if left_idx >= max_nodes or right_idx >= max_nodes:
            return

        # Set split parameters
        node.feature_index = rand_int(0, n_features)
        cdef int split_value_index = rand_int(0, split_values.lengths[node.feature_index])
        node.split_value = split_values.values[node.feature_index][split_value_index]

        # Recursively grow children
        grow_subtree(
            pool, tree, left_idx, n_features, split_values,
            depth + 1, max_depth, max_nodes
        )
        grow_subtree(
            pool, tree, right_idx, n_features, split_values,
            depth + 1, max_depth, max_nodes
        )

cdef int random_subnode_index(Tree* tree, int max_nodes) noexcept nogil:
    """
    Select a random active node index from the tree.
    Returns the tree index (not the pool index) of a random active node.
    """
    if tree is NULL:
        return -1

    # Count active nodes and build list of active indices
    cdef int active_count = 0
    cdef int i

    # First pass: count active nodes
    for i in range(max_nodes):
        if tree.node_indices[i] != -1:
            active_count += 1

    if active_count == 0:
        return -1

    # Second pass: select random active node
    cdef int target_index = rand_int(0, active_count)
    cdef int current_count = 0

    for i in range(max_nodes):
        if tree.node_indices[i] != -1:
            if current_count == target_index:
                return i  # Return tree index
            current_count += 1

    return -1  # Should never reach here

cdef int random_subnode_index_with_depth(Tree* tree, int max_nodes, int* out_depth) noexcept nogil:
    """
    Select a random active node index from the tree and return its depth.
    Returns the tree index and sets out_depth to the node's depth in the tree.
    """
    cdef int node_index = random_subnode_index(tree, max_nodes)
    if node_index == -1:
        if out_depth is not NULL:
            out_depth[0] = -1
        return -1

    # Calculate depth from tree index (assuming complete binary tree indexing)
    cdef int depth = 0
    cdef int temp_index = node_index

    if temp_index == 0:
        depth = 0  # Root node
    else:
        while temp_index > 0:
            temp_index = (temp_index - 1) // 2  # Parent index
            depth += 1

    if out_depth is not NULL:
        out_depth[0] = depth

    return node_index

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
