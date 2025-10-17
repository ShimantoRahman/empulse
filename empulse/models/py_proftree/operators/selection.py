import numpy as np

from ..node import Node


def selection(
    population: list[Node], selection_tournament_size: int, random: np.random.RandomState
) -> tuple[Node, Node]:
    """
    Chooses parent trees from the current population to produce offspring for the next generation.

    By default, `GATree` class uses tournament selection,
    a method where a subset of the population is randomly chosen,
    and the best individual from this subset is selected.

    Pseudocode of the implementation:

    1. Randomly select `selection_tournament_size` trees from the population.
    2. Evaluate the fitness of the selected trees.
    3. Choose the tree with the best fitness from the selected subset.
    4. Repeat the process to select another parent.
    5. Ensure the selected parents are different to maintain genetic diversity.

    Args:
        population (list): List of trees.
        selection_tournament_size (int): Number of trees to select.
        random (Random): Random number generator.

    Returns
    -------
        tuple: The two selected trees.
    """
    valid = False
    while not valid:
        selection = []

        # Select two trees
        for _ in range(2):
            indices = random.choice(len(population), selection_tournament_size, replace=False)
            candidates = [population[i] for i in indices]
            candidates.sort(key=lambda x: x.fitness, reverse=False)
            selection.append(candidates[0])

        # Check if trees are different
        if selection[0] != selection[1]:
            valid = True

    return selection[0], selection[1]
