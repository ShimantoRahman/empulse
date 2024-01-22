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

    bounds = [(-10, 10)] * 2
    rga.optimize(objective, bounds)
    assert rga.lower_bounds is not None
    assert rga.upper_bounds is not None
    assert rga.delta_bounds is not None
    assert rga.n_dim == 2


def test_rga_generate_population(rga):
    rga.lower_bounds = np.array([-10, -10])
    rga.upper_bounds = np.array([10, 10])
    rga.delta_bounds = np.array([20, 20])
    rga.n_dim = 2
    rga.population = rga._generate_population()
    assert rga.population.shape == (10, 2)
    assert np.all(rga.population >= -10) and np.all(rga.population <= 10)


@pytest.fixture
def rga_fixed():
    rga = RGA(
        population_size=10,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism=0.05,
        verbose=False,
        random_state=101,
        n_jobs=1,
    )
    # Replace self.rng with a mock object
    rga.rng = Mock()
    rga.rng.uniform = MagicMock(return_value=0.5)
    rga.rng.rand = MagicMock(return_value=np.ones((10, 2)))
    rga.rng.choice = MagicMock(return_value=np.arange(10))
    return rga


def test_rga_mutate(rga_fixed):
    rga = rga_fixed
    rga.lower_bounds = np.array([-10, -10])
    rga.upper_bounds = np.array([10, 10])
    rga.delta_bounds = np.array([20, 20])
    rga.n_dim = 2
    rga.population = rga._generate_population()
    original_population = rga.population.copy()
    rga._mutate()
    assert not np.array_equal(rga.population, original_population), "Population should change after mutation"


def test_rga_crossover(rga_fixed):
    rga = rga_fixed
    rga.lower_bounds = np.array([-10, -10])
    rga.upper_bounds = np.array([10, 10])
    rga.delta_bounds = np.array([20, 20])
    rga.n_dim = 2
    rga.population = rga._generate_population()
    original_population = rga.population.copy()
    rga._crossover()
    assert not np.array_equal(rga.population, original_population), "Population should change after crossover"
