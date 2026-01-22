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
cdef bint is_leaf(Node* node) noexcept nogil
cdef float node_probability(Node* node) noexcept nogil
cdef void update_node_stats(Node* node, int y) noexcept nogil
