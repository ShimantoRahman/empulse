"""
The decision each leaf of a cost-sensitive tree makes: the class that costs least on its training samples.

A leaf's class fractions say which class is most common in it, not which prediction is cheapest. With
imbalanced classes and a false negative that costs more than a false positive, a leaf where a third of
the samples are positive is usually cheaper to predict positive, while its majority is negative. The
trees are grown to lower exactly this cost, so each leaf predicts the class that does.
"""

import numpy as np
from sklearn.tree._tree import Tree

from ..._types import FloatNDArray, IntArrayLike, IntNDArray


def positive_leaves(
    tree: Tree,
    leaves: IntNDArray,
    y: IntArrayLike,
    tp_cost: FloatNDArray | float,
    tn_cost: FloatNDArray | float,
    fp_cost: FloatNDArray | float,
    fn_cost: FloatNDArray | float,
    counts: IntNDArray | None = None,
) -> np.ndarray:
    """
    Return, for every node of a fitted tree, whether predicting positive costs less than negative.

    The cost of a decision in a node is the sum, over the training samples in it, of what that decision
    costs for each sample, weighted as the tree weighted the samples when it was grown. The weight each
    class carries in a node is read from the tree itself, so it includes whatever the tree was fitted
    with (bootstrap counts, class weights). Within a class, a sample counts as often as the tree drew
    it, which is all that varies between samples of the same class. Where both decisions cost the
    same, as for a cost matrix whose outcomes cost the same whatever is predicted, the costs have no
    preference, and the node predicts its more frequent class. A tie in that too goes to negative.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The fitted tree.
    leaves : ndarray of shape (n_samples,)
        The leaf each training sample falls in (``apply`` of the training data).
    y : ndarray of shape (n_samples,)
        The training labels, encoded as 0 and 1.
    tp_cost, tn_cost, fp_cost, fn_cost : float or ndarray of shape (n_samples,)
        The cost of each outcome for each training sample.
    counts : ndarray of shape (n_samples,), optional
        How many times the tree drew each sample (its bootstrap counts). ``None`` means once each.

    Returns
    -------
    positive : ndarray of shape (n_nodes,), dtype bool
        Whether each node predicts the positive class. Only the leaves' entries are meaningful.
    """
    n_nodes = tree.node_count
    is_positive = np.asarray(y).reshape(-1) == 1
    weights = np.ones(is_positive.shape, dtype=np.float64) if counts is None else np.asarray(counts, np.float64)
    # The cost of predicting each sample positive, and negative.
    cost_if_positive = np.broadcast_to(np.where(is_positive, tp_cost, fp_cost), is_positive.shape)
    cost_if_negative = np.broadcast_to(np.where(is_positive, fn_cost, tn_cost), is_positive.shape)

    def mean_per_node(cost: FloatNDArray, members: np.ndarray) -> FloatNDArray:
        """Average *cost* over the members of each node, counting each as often as the tree drew it."""
        total = np.bincount(leaves[members], weights=(weights * cost)[members], minlength=n_nodes)
        drawn = np.bincount(leaves[members], weights=weights[members], minlength=n_nodes)
        return np.divide(total, drawn, out=np.zeros(n_nodes), where=drawn > 0)  # type: ignore[no-any-return]

    # The total weight of each class in each node, as the tree weighted it.
    class_weight = tree.value[:, 0, :] * tree.weighted_n_node_samples[:, None]
    negative, positive = class_weight[:, 0], class_weight[:, 1]
    is_negative = ~is_positive
    cost_positive = negative * mean_per_node(cost_if_positive, is_negative) + positive * mean_per_node(
        cost_if_positive, is_positive
    )
    cost_negative = negative * mean_per_node(cost_if_negative, is_negative) + positive * mean_per_node(
        cost_if_negative, is_positive
    )
    tie = cost_positive == cost_negative
    return (cost_positive < cost_negative) | (tie & (positive > negative))  # type: ignore[no-any-return]
