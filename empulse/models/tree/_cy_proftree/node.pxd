cdef struct Node:
    Node* left
    Node* right
    Node* parent
    float split_value
    int feature_index
    int n_samples
    int n_positive_samples

cdef Node* create_node() noexcept nogil
cdef Node* copy_node(Node* node, Node* parent = *) noexcept nogil
cdef void free_node(Node* node) noexcept nogil
cdef void reset_node(Node* node) noexcept nogil
cdef inline bint is_leaf(Node* node) noexcept nogil:
    # Defined in the header: a cdef function cimported from another module is called through a
    # pointer and never inlined, and tree traversal calls this once per node per sample.
    return node.left is NULL and node.right is NULL
cdef float node_probability(Node* node) noexcept nogil
