from itertools import islice

import pytest
import numpy as np
from empulse.models.optimizers import RGA
from unittest.mock import Mock, MagicMock


@pytest.fixture()
def rga():
    return RGA(
        population_size=10,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism=0.05,
        verbose=False,
        random_state=101,
        n_jobs=1,
    )


def test_rga_init(rga):
    assert rga.population_size == 10
    assert rga.crossover_rate == 0.8
    assert rga.mutation_rate == 0.1
    assert rga.elitism == 0.05
    assert rga.verbose is False
    assert rga.rng is not None
    assert rga.n_jobs == 1


def test_rga_optimize(rga):
    def objective(x):
        return np.sum(x ** 2)

    bounds = [(-10, 10)]
    for _ in islice(rga.optimize(objective, bounds), 100):
        pass
    assert rga.lower_bounds is not None
    assert rga.upper_bounds is not None
    assert rga.delta_bounds is not None
    assert rga.n_dim == 1
    assert rga.result.x == pytest.approx(10, 1e-2)


def test_rga_generate_population(rga):
    rga.lower_bounds = np.array([-10, -10])
    rga.upper_bounds = np.array([10, 10])
    rga.delta_bounds = np.array([20, 20])
    rga.n_dim = 2
    rga.population = rga._generate_population()
    assert rga.population.shape == (10, 2)
    assert np.all(rga.population >= -10) and np.all(rga.population <= 10)


def test_rga_fitness_calculation(rga):
    def objective(x):
        return np.sum(x ** 2)

    bounds = [(-10, 10)]
    for _ in islice(rga.optimize(objective, bounds), 10):
        pass
    assert isinstance(rga.population, np.ndarray)
    assert isinstance(rga.fitness, np.ndarray)
    assert rga.population.shape == (10, 1)
    assert rga.fitness == (10,)
    # Check if fitness is calculated correctly
    for fitness_val, x in zip(rga.fitness, rga.population):
        assert fitness_val == pytest.approx(objective(x))


def test_rga_mutate():
    rga = RGA(
        population_size=10,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism=0.05,
        verbose=False,
        random_state=101,
        n_jobs=1,
    )
    rga.population = np.array([[1]] * 10)
    rga.fitness = np.array([0.0] * 10)
    rga.lower_bounds = np.array([0])
    rga.upper_bounds = np.array([1])
    rga.n_dim = 2

    rga.rng = Mock()
    rga.rng.uniform = MagicMock()
    rga.rng.uniform.side_effect = [1] * 8 + [0] * 4
    rga.rng.choice = MagicMock(return_value=0)

    rga._mutate()
    assert np.array_equal(rga.population, np.array([[1]] * 8 + [[0]] * 2))


def test_rga_crossover(rga):
    rga.lower_bounds = np.array([-10, -10])
    rga.upper_bounds = np.array([10, 10])
    rga.delta_bounds = np.array([20, 20])
    rga.n_dim = 2
    rga.population = rga._generate_population()
    original_population = rga.population.copy()
    rga._crossover()
    assert not np.array_equal(rga.population, original_population)
