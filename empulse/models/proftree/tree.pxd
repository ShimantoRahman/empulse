cimport numpy as cnp

from .node cimport Node

cdef struct Tree:
    Node* root
    float fitness

cdef struct SplitValues:
    float **values
    int *lengths
    int n_features

cdef Tree* create_tree(bint with_root = *) noexcept nogil
cdef Tree* copy_tree(Tree* tree) noexcept nogil
cdef void free_tree(Tree* tree) noexcept nogil

cdef Node* get_leaf(Node* start_node, float[:] x) noexcept nogil

cdef void fit_tree(Tree* tree, float[:, :] X, int[:] y, int n_samples) noexcept nogil

cdef void predict_proba_tree(Tree* tree, float[:, :] X, float[:] probabilities, int n_samples) noexcept nogil
cdef void predict_labels_tree(Tree* tree, float[:, :] X, float[:] probabilities, int n_samples)

cdef SplitValues* compute_split_values(cnp.ndarray[cnp.float32_t, ndim=2] X) noexcept
cdef void free_split_values(SplitValues* sv) noexcept nogil

cdef void grow_subtree(
    Node* node,
    int n_features,
    SplitValues* split_values,
    int depth = *,
    int max_depth = *,
) noexcept nogil

cdef Node* random_subnode(Node* root) noexcept nogil
cdef Node* random_subnode_with_depth(Node* root, int* out_depth) noexcept nogil
cdef Node* random_leaf_node(Node* root, int* out_depth) noexcept nogil

cdef void prune(Node* node) noexcept nogil
cdef void split(
    Node* node,
    int n_features,
    SplitValues* split_values,
    int depth,
    int max_depth,
) noexcept nogil
