from libc.stdlib cimport malloc, free

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

cdef bint is_leaf(Node* node) noexcept nogil:
    return node.left is NULL and node.right is NULL

cdef float node_probability(Node* node) noexcept nogil:
    if node.n_samples == 0:
        return 0.5
    return <float>node.n_positive_samples / <float>node.n_samples

cdef void update_node_stats(Node* node, int y) noexcept nogil:
    node.n_samples += 1
    node.n_positive_samples += y