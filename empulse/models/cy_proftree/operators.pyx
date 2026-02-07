# distutils: language = c++

from libc.math cimport NAN

from .node cimport Node, free_node, copy_node
from .tree cimport (
    Tree,
    SplitValues,
    random_subnode_with_depth,
    random_subnode,
    prune,
    split,
    random_leaf_node,
    random_subnode_with_leaf_children,
)
from .random cimport rand_int


cdef void prune_subtree_at_depth(Node* node, int max_depth) noexcept nogil:
    if node is NULL or max_depth < 0:
        return

    if max_depth == 0:
        free_node(node.left)
        free_node(node.right)
        node.left = NULL
        node.right = NULL
    else:
        # Only recurse if both children exist (maintain binary tree property)
        if node.left is not NULL and node.right is not NULL:
            prune_subtree_at_depth(node.left, max_depth - 1)
            prune_subtree_at_depth(node.right, max_depth - 1)
        elif node.left is not NULL or node.right is not NULL:
            # If only one child exists, prune both to maintain binary property
            free_node(node.left)
            free_node(node.right)
            node.left = NULL
            node.right = NULL

cdef int count_nodes(Node* node) noexcept nogil:
    if node is NULL:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

cdef Tree* crossover(Tree* mother, Tree* father, int max_depth) noexcept nogil:
    cdef int mother_depth, father_depth
    cdef Node* mother_node = random_subnode_with_depth(mother.root, &mother_depth)
    cdef Node* father_node = random_subnode_with_depth(father.root, &father_depth)

    cdef Node* parent_node = mother_node.parent
    cdef bint is_left_child = False
    cdef Node* new_subtree

    new_subtree = copy_node(father_node, NULL)

    cdef int remaining_depth = max_depth - father_depth
    if remaining_depth >= 0:
        prune_subtree_at_depth(new_subtree, remaining_depth)
    else:
        free_node(new_subtree.left)
        free_node(new_subtree.right)
        new_subtree.left = NULL
        new_subtree.right = NULL

    if parent_node is NULL:
        free_node(mother.root)
        mother.root = new_subtree
        mother.root.parent = NULL
    else:
        if parent_node.left is mother_node:
            is_left_child = True
            parent_node.left = NULL   # detach before free
        else:
            parent_node.right = NULL  # detach before free

        free_node(mother_node)

        new_subtree.parent = parent_node
        if is_left_child:
            parent_node.left = new_subtree
        else:
            parent_node.right = new_subtree

    mother.fitness = NAN
    mother.n_nodes = count_nodes(mother.root)
    return mother

cdef inline void grow(Tree* tree, SplitValues* split_values, int n_features, int max_depth) noexcept nogil:
    """Add a random split rule to a random leaf node."""
    cdef int depth = 0
    cdef Node* leaf = random_leaf_node(tree.root, &depth)
    split(leaf, n_features, split_values, depth, max_depth)
    tree.n_nodes += 2

cdef inline void prune_internal(Tree* tree) noexcept nogil:
    """Prune a random internal node which has two leaf nodes as successors."""
    cdef Node* node = random_subnode_with_leaf_children(tree.root)
    if node is not NULL and node is not tree.root:
        prune(node)
        tree.n_nodes -= 2


cdef inline void mutate_split_feature(Tree* tree, int n_features, SplitValues* split_values) noexcept nogil:
    cdef Node* node = random_subnode(tree.root)
    if node is NULL:
        return
    node.feature_index = rand_int(0, n_features)
    cdef int idx = rand_int(0, split_values.lengths[node.feature_index])
    node.split_value = split_values.values[node.feature_index][idx]

cdef inline void mutate_split_value(Tree* tree, SplitValues* split_values) noexcept nogil:
    cdef Node* node = random_subnode(tree.root)
    if node is NULL:
        return
    cdef int idx = rand_int(0, split_values.lengths[node.feature_index])
    node.split_value = split_values.values[node.feature_index][idx]
