import numpy as np

from ...._types import IntNDArray
from ..node import Node


def mutation(
    root: Node,
    att_indexes: IntNDArray,
    att_values: dict[int, list[float]],
    class_count: int,
    random: np.random.RandomState,
) -> Node:
    """
    Mutation introduces random changes to a tree to maintain genetic diversity and explore new solutions.

    This helps in avoiding local optima by introducing new genetic structures.

    Pseudocode of the implementation:

    1. Randomly select a node within the tree.
    2. Depending on whether the selected node is a leaf or an internal node, apply different mutation strategies:

        - **Leaf Mutation:** Change the class label or replace the leaf with a new subtree.
        - **Internal Node Mutation:** Change the attribute, change the attribute value,
          replace the node with a class label, or replace the node with a new subtree.

    3. Ensure the mutated node maintains the structural integrity of the tree.
    4. Return the root of the mutated tree.

    Args:
        root (Node): The root node of the tree.
        att_indexes (list): List of attribute indexes.
        att_values (list): List of attribute values.
        class_count (int): Number of classes.
        random (Random): Random number generator.

    Returns
    -------
        Node: The mutated node.
    """
    node = Node.copy(root)
    depth = node.max_depth()

    while True:
        if node.att_index == -1:  # for leaves
            mutate_leaf(node, att_indexes, att_values, class_count, random)
            break
        elif random.randint(0, depth) == 0:  # for mid-tree nodes
            mutate_operator(node, att_indexes, att_values, class_count, random)
            break

        # go to next random child
        if random.choice([True, False]):
            if node.left is not None:
                node = node.left
            else:
                break
        elif node.right is not None:
            node = node.right
        else:
            break

    return node.get_root()


def mutate_leaf(
    node: Node,
    att_indexes: IntNDArray,
    att_values: dict[int, list[float]],
    class_count: int,
    random: np.random.RandomState,
) -> None:
    """
    Mutate a leaf node.

    The mutation operation can be one of the following:

    - Change the class of a leaf node.
    - Exchange a leaf node for a new subtree.

    Args:
        node (Node): The leaf node.
        att_indexes (list): List of attribute indexes.
        att_values (list): List of attribute values.
        class_count (int): Number of classes.
        random (Random): Random number generator.
    """
    if random.choice([True, False]):  # change class
        change_class(node, class_count, random)
    else:  # exchange for new subtree
        exchange_class_for_tree(node, att_indexes, att_values, class_count, random)


def change_class(node: Node, class_count: int, random: np.random.RandomState) -> None:
    """
    Change class of a leaf node. The new class must be different from the old class.

    Args:
        node (Node): The leaf node.
        class_count (int): Number of classes.
        random (Random): Random number generator.
    """
    result_old = node.att_value
    result_new = result_old

    while result_old == result_new:  # classes must be different
        result_new = random.randint(0, class_count)

    node.att_value = result_new


def exchange_class_for_tree(
    node: Node,
    att_indexes: IntNDArray,
    att_values: dict[int, list[float]],
    class_count: int,
    random: np.random.RandomState,
) -> None:
    """
    Exchange a leaf node for a new subtree. The new subtree is created with a random depth.

    Args:
        node (Node): The leaf node.
        att_indexes (list): List of attribute indexes.
        att_values (list): List of attribute values.
        class_count (int): Number of classes.
        random (Random): Random number generator.
    """
    parent = node.parent
    left = False
    if parent.left == node:
        left = True

    n = Node()
    subtree = n.make_node(
        max_depth=node.depth(), random=random, att_indexes=att_indexes, att_values=att_values, class_count=class_count
    )
    subtree.parent = parent

    if left:
        parent.set_left(subtree)
    else:
        parent.set_right(subtree)


def mutate_operator(
    node: Node,
    att_indexes: IntNDArray,
    att_values: dict[int, list[float]],
    class_count: int,
    random: np.random.RandomState,
) -> None:
    """
    Mutate a mid-tree node.

    The mutation operation can be one of the following:

    - Change the attribute of an operator node.
    - Change the attribute value of an operator node.
    - Exchange an operator node for a class.
    - Exchange an operator node for a new subtree.

    Args:
        node (Node): The mid-tree node.
        att_indexes (list): List of attribute indexes.
        att_values (list): List of attribute values.
        class_count (int): Number of classes.
        random (Random): Random number generator.
    """
    rand = random.random()
    if rand < 0.25 or node.parent is None:  # exchange attribute for other attribute
        change_attribute(node, att_indexes, att_values, random)
    elif rand < 0.5:  # exchange attribute value for other value
        change_attribute_value(node, att_values, random)
    elif rand < 0.75:  # exchange attribute for class
        exchange_tree_for_class(node, class_count, random)
    else:  # exchange for new subtree
        exchange_tree_for_tree(node, att_indexes, att_values, class_count, random)


def change_attribute(
    node: Node, att_indexes: IntNDArray, att_values: dict[int, list[float]], random: np.random.RandomState
) -> None:
    """
    Change attribute of a mid-tree node. The new attribute must be different from the old attribute.

    Args:
        node (Node): The mid-tree node.
        att_indexes (list): List of attribute indexes.
        att_values (list): List of attribute values.
        random (Random): Random number generator.
    """
    att_index_old = node.att_index
    att_index_new = att_index_old

    att_value_old = node.att_value
    att_value_new = att_value_old

    while att_index_old == att_index_new:
        att_index_new = random.randint(0, len(att_indexes))
        att_value_new = random.randint(0, len(att_values[att_index_new]))

    node.att_index = att_index_new
    node.att_value = att_values[att_index_new][att_value_new]


def change_attribute_value(node: Node, att_values: dict[int, list[float]], random: np.random.RandomState) -> None:
    """
    Change attribute value of a mid-tree node. The new attribute value must be different from the old attribute value.

    Args:
        node (Node): The mid-tree node.
        att_values (list): List of attribute values.
        random (Random): Random number generator.
    """
    att_index = node.att_index

    att_value_old = node.att_value
    att_value_new = att_value_old

    while att_value_old == att_value_new:
        att_value_new = random.randint(0, len(att_values[att_index]))

    node.att_value = att_values[att_index][att_value_new]


def exchange_tree_for_class(node: Node, class_count: int, random: np.random.RandomState) -> None:
    """
    Exchange a mid-tree node for a class. The new class is selected randomly.

    Args:
        node (Node): The mid-tree node.
        class_count (int): Number of classes.
        random (Random): Random number generator.
    """
    parent = node.parent
    left = False
    if parent.left == node:
        left = True

    leaf = Node(att_index=-1, att_value=random.randint(0, class_count))
    leaf.parent = parent

    if left:
        parent.set_left(leaf)
    else:
        parent.set_right(leaf)


def exchange_tree_for_tree(
    node: Node,
    att_indexes: IntNDArray,
    att_values: dict[int, list[float]],
    class_count: int,
    random: np.random.RandomState,
) -> None:
    """
    Exchange a mid-tree node for a new subtree. The new subtree is created with a random depth.

    Args:
        node (Node): The mid-tree node.
        att_indexes (list): List of attribute indexes.
        att_values (list): List of attribute values.
        class_count (int): Number of classes.
        random (Random): Random number generator.
    """
    parent = node.parent
    left = False
    if parent.left == node:
        left = True

    n = Node()
    subtree = n.make_node(
        max_depth=node.depth(), random=random, att_indexes=att_indexes, att_values=att_values, class_count=class_count
    )
    subtree.parent = parent

    if left:
        parent.set_left(subtree)
    else:
        parent.set_right(subtree)
