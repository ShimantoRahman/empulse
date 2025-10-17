import numpy as np

from ..node import Node


def crossover(tree1: Node, tree2: Node, random: np.random.RandomState) -> Node:
    """
    Combine the genetic information of two parent trees to generate new offspring.

    This enables exploration,
    which helps in creating diversity in the population and combining good traits from both parents.

    Pseudocode of the implementation:

    1. Copy the parent trees to avoid altering the originals.
    2. Randomly select a crossover point in each tree.
    3. Swap the subtrees at the selected points between the two trees.
    4. Return the new tree created from the crossover.

    Args:
        tree1 (Node): The first tree for crossover.
        tree2 (Node): The second tree for crossover.
        random (Random): Random number generator.

    Returns
    -------
        Node: The new tree resulting from crossover.
    """
    n1 = Node.copy(tree1)
    n2 = Node.copy(tree2)
    size1 = n1.max_depth()
    size2 = n2.max_depth()

    while True:
        if (n1.left is None or random.randint(0, size1) == 0) and n1.parent is not None:
            break

        n1 = n1.left if random.choice([True, False]) else n1.right

    while True:
        if (n2.left is None or random.randint(0, size2) == 0) and n2.parent is not None:
            break

        n2 = n2.left if random.choice([True, False]) else n2.right

    p = n1.parent
    if p.left == n1:
        p.set_left(n2)
    else:
        p.set_right(n2)

    return p.get_root()
