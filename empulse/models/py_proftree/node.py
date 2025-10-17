import sys
import warnings
from typing import Any, Union

import numpy as np

from ..._types import FloatNDArray, IntNDArray

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class Node:
    """
    Node class for decision trees.

    Each node represents a decision in the tree.
    The node can be a mid-tree node or a leaf node.
    Mid-tree nodes have an attribute index and an attribute value, while leaf nodes have a class index.
    The node can have a parent, left child, and right child.
    The node also has a fitness value, true class values, and predicted class values for evaluation.

    Args:
        att_index (int, optional): Attribute index or -1 if this is a leaf.
        att_value (any, optional): Attribute value or class index if this is a leaf.

    Attributes
    ----------
        parent (Node): Parent node.
        left (Node): Left child node.
        right (Node): Right child node.
        att_index (int, optional): Attribute index or -1 if this is a leaf.
        att_value (any, optional): Attribute value or class index if this is a leaf.
        fitness (None): Placeholder for fitness value.
        y_true (list): List of true class values for evaluation.
        y_pred (list): List of predicted class values for evaluation.
    """

    def __init__(self, att_index: int | None = None, att_value: Any = None):
        if att_index is None and att_value is not None:
            self.__init__(-1, att_value)  # leaf node
        else:
            self.parent = None
            self.left = None
            self.right = None

            self.att_index = att_index if att_index is not None else None  # if -1 then this is leaf
            self.att_value = att_value if att_value is not None else None

            self.fitness = None
            self.y_true = []
            self.y_pred = []

    @staticmethod
    def copy(node: 'Node', parent: Union['Node', None] = None) -> 'Node':
        """
        Return a deep copy of the given node. The parent node can be specified for the copied node.

        Args:
            node (Node): Node to be copied.
            parent (Node, optional): Parent node of the copied node.

        Returns
        -------
            Node: Deep copy of the given node.
        """
        copy = Node(node.att_index, node.att_value)
        copy.parent = parent
        copy.left = Node.copy(node.left, parent=copy) if node.left is not None else None
        copy.right = Node.copy(node.right, parent=copy) if node.right is not None else None
        copy.fitness = node.fitness
        copy.y_true = node.y_true
        copy.y_pred = node.y_pred
        return copy

    def set_left(self, n: Self) -> None:
        """
        Set the left child of this node to the given node. Also sets the parent of the given node to this node.

        Args:
            n (Node): Node to be set as the left child.
        """
        self.left = n
        n.parent = self

    def set_right(self, n: Self) -> None:
        """
        Set the right child of this node to the given node. Also sets the parent of the given node to this node.

        Args:
            n (Node): Node to be set as the right child.
        """
        self.right = n
        n.parent = self

    def get_root(self) -> Self:
        """
        Return the root node of the tree.

        Returns
        -------
            Node: Root node of the tree.
        """
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_children(self) -> tuple[Self, Self]:
        """
        Return the children of this node.

        Returns
        -------
            tuple: Tuple containing the left and right children nodes.
        """
        return self.left, self.right

    def get_leaves(self) -> list[Self]:
        """
        Return the leaves of this node.

        Returns
        -------
            list: List of leaves of this node.
        """
        if self.left is None:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    def depth(self) -> int:
        """
        Return the depth of this node.

        Returns
        -------
            int: Depth of this node.
        """
        if self.parent is None:
            return 1
        return self.parent.depth() + 1

    def max_depth(self) -> int:
        """
        Return the maximum depth of the tree.

        The depth of the tree is the number of edges on the longest path from the root to a leaf.

        Returns
        -------
            int: Maximum depth of the tree.
        """
        return self.max_depth_helper(self.get_root())

    @staticmethod
    def max_depth_helper(n: 'Node') -> int:
        """
        Help to calculate the max_depth.

        Args:
            n (Node): Node to be used as the root.

        Returns
        -------
            int: Maximum depth of the tree.
        """
        if n is None:
            return 0
        l_depth = Node.max_depth_helper(n.left)
        r_depth = Node.max_depth_helper(n.right)
        return max(l_depth, r_depth) + 1

    def size(self) -> int:
        """
        Size is the number of all nodes (mid-tree nodes + leaves) in the trees.

        Returns
        -------
            int: Number of all nodes in the trees.
        """
        return self.size_helper(self.get_root())

    @staticmethod
    def size_helper(n: 'Node') -> int:
        """
        Help to calculate the size.

        Args:
            n (Node): Node to be used as the root.

        Returns
        -------
            int: Number of all nodes in the trees.
        """
        if n is None:
            return 0
        return Node.size_helper(n.left) + Node.size_helper(n.right) + 1

    def make_node(
        self,
        depth: int = 0,
        max_depth: int | None = None,
        random: np.random.RandomState | None = None,
        att_indexes: IntNDArray | None = None,
        att_values: dict[int, list[float]] | None = None,
        class_count: int | None = None,
    ) -> Self:
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
        att_index = None
        value_index = None
        att_value = None
        if max_depth is None:
            max_depth = depth

        try:
            # if it's the root, first level or 50/50 chance of building new children.
            # Must be below maximal depth.
            if depth <= 1 or ((random.choice([True, False])) and depth < max_depth):
                subset_index = random.randint(0, len(att_indexes))
                att_index = att_indexes[subset_index]
                value_index = random.randint(0, len(att_values[att_index]))
                att_value = att_values[att_index][value_index]
                node = Node(att_index=att_index, att_value=att_value)
                node.left = self.make_node(
                    depth=depth + 1,
                    max_depth=max_depth,
                    random=random,
                    att_indexes=att_indexes,
                    att_values=att_values,
                    class_count=class_count,
                )
                node.left.parent = node
                node.right = self.make_node(
                    depth=depth + 1,
                    max_depth=max_depth,
                    random=random,
                    att_indexes=att_indexes,
                    att_values=att_values,
                    class_count=class_count,
                )
                node.right.parent = node
            else:  # result (leaf)
                r = random.randint(0, class_count)
                node = Node(att_index=-1, att_value=r)
        except (IndexError, KeyError, TypeError) as e:
            warnings.warn(f'Error when building node:{e}', stacklevel=2)
            node = None

        return node

    def clear_evaluation(self) -> None:
        """
        Clear the evaluation of this node and all its children.

        The true class values and predicted class values are reset to empty lists.
        """
        self.y_true = []
        self.y_pred = []
        if self.left:
            self.left.clear_evaluation()
        if self.right:
            self.right.clear_evaluation()

    def is_evaluated(self) -> bool:
        """
        Return true if this node and all its children are evaluated.

        A node is evaluated if it has true class values and predicted class values.

        Returns
        -------
            bool: True if this node and all its children are evaluated.
        """
        if len(self.y_pred) == 0 and len(self.y_true) == 0:
            return False

        left_evaluated = self.left.is_evaluated() if self.left else True
        right_evaluated = self.right.is_evaluated() if self.right else True

        return left_evaluated or right_evaluated

    def predict_one(self, X: FloatNDArray, y: int | None = None, train: bool = False) -> int:
        """
        Predicts the class of the given instance.

        If the actual class is provided, the true class value and predicted class value are stored for evaluation.

        Args:
            X (list): Instance to be predicted.
            y (int): Actual class of the given instance.
            train (bool): If it's used for training or only predicting

        Returns
        -------
            int: Predicted class.
        """
        try:
            if self.att_index != -1:
                if X[self.att_index] > self.att_value:
                    if self.left is not None:
                        predicted = self.left.predict_one(X, y)
                elif self.right is not None:
                    predicted = self.right.predict_one(X, y)
            else:
                predicted = int(self.att_value)

            if train is True:
                if y is not None:
                    self.y_true.append(int(y))
                self.y_pred.append(predicted)

            return predicted
        except (IndexError, KeyError, TypeError) as e:
            warnings.warn(f'Error when predicting: {e}', stacklevel=2)
            return -1

    def __str__(self) -> str:
        if self.att_index == -1:
            # Assuming att_value is a numeric index
            return str(int(self.att_value))
            # For nominal values, you might need a mapping from index to value
        else:
            return f'Attribute {self.att_index} > {self.att_value}'

    def __repr__(self):
        return self.__str__()
