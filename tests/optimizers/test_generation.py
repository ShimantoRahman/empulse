from itertools import islice

import pytest
import numpy as np
from empulse.models.optimizers import Generation


@pytest.fixture()
def generation():
    return Generation(
        population_size=10,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism=0.05,
        verbose=False,
        random_state=101,
        n_jobs=1,
    )


def test_rga_init(generation):
    assert generation.population_size == 10
    assert generation.crossover_rate == 0.8
    assert generation.mutation_rate == 0.1
    assert generation.elitism == 0.05
    assert generation.verbose is False
    assert generation.rng is not None
    assert generation.n_jobs == 1


def test_rga_optimize_parabola(generation):
    def objective(x):
        return np.sum(x ** 2)

    bounds = [(-10, 10)]
    for _ in islice(generation.optimize(objective, bounds), 100):
        pass
    assert generation.lower_bounds is not None
    assert generation.upper_bounds is not None
    assert generation.delta_bounds is not None
    assert generation.n_dim == 1
    assert generation.result.x == pytest.approx(10, 1e-2) or generation.result.x == pytest.approx(-10, 1e-2)


def test_rga_optimize_trigonometric(generation):
    def objective(x):
        return float(((x ** 2 + x) * np.cos(x))[0])

    bounds = [(-10, 10)]
    for _ in islice(generation.optimize(objective, bounds), 100):
        pass

    assert generation.result.x == pytest.approx(6.5606, abs=1e-2)
    assert generation.result.fun == pytest.approx(47.7056, abs=1e-2)


def test_rga_optimize_differentiable_unimodal_func(generation):

    def objective(x, a=10, b=3):
        x1, x2 = x[0], x[1]
        z = x1 ** 2 + b * x2 ** 2
        return -(a - np.exp(-z))

    bounds = [(-1, 1)] * 2
    for _ in islice(generation.optimize(objective, bounds), 100):
        pass

    assert generation.result.x[0] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.x[1] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.fun == pytest.approx(-9.0, abs=1e-2)


def test_rga_optimize_non_differentiable_unimodal_func(generation):

    def objective(x, a=10, b=3):
        x1, x2 = x[0], x[1]
        z = x1 ** 2 + b * x2 ** 2
        return -(a - np.exp(-z))

    bounds = [(-1, 1)] * 2
    for _ in islice(generation.optimize(objective, bounds), 100):
        pass

    assert generation.result.x[0] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.x[1] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.fun == pytest.approx(-9.0, abs=1e-2)


def test_rga_optimize_rotated_ellipse(generation):

    def objective(x):
        x1, x2 = x[0], x[1]
        return -(2 * (x1 ** 2 - x1 * x2 + x2 ** 2))

    bounds = [(-1, 1)] * 2
    for _ in islice(generation.optimize(objective, bounds), 100):
        pass

    assert generation.result.x[0] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.x[1] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.fun == pytest.approx(0.0, abs=1e-2)


def test_rga_optimize_rastrigin(generation):

    def objective(x):
        x1, x2 = x[0], x[1]
        return -(20 + x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + x2 ** 2 - 10 * np.cos(2 * np.pi * x2))

    bounds = [(-5.12, 5.12)] * 2
    for _ in islice(generation.optimize(objective, bounds), 500):
        pass

    assert generation.result.x[0] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.x[1] == pytest.approx(0.0, abs=1e-2)
    assert generation.result.fun == pytest.approx(0.0, abs=1e-2)


def test_rga_generate_population(generation):
    generation.lower_bounds = np.array([-10, -10])
    generation.upper_bounds = np.array([10, 10])
    generation.delta_bounds = np.array([20, 20])
    generation.n_dim = 2
    generation.population = generation._generate_population()
    assert generation.population.shape == (10, 2)
    assert np.all(generation.population >= -10) and np.all(generation.population <= 10)


def test_rga_fitness_calculation(generation):
    def objective(x):
        return np.sum(x ** 2)

    bounds = [(-10, 10)]
    for _ in islice(generation.optimize(objective, bounds), 10):
        pass
    assert isinstance(generation.population, np.ndarray)
    assert isinstance(generation.fitness, np.ndarray)
    assert generation.population.shape == (10, 1)
    assert generation.fitness.shape == (10,)
    # Check if fitness is calculated correctly
    for fitness_val, x in zip(generation.fitness, generation.population):
        assert fitness_val == pytest.approx(objective(x))


