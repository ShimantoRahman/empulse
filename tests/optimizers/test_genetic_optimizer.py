import warnings

import numpy as np
import pytest

from empulse.optimizers import GeneticAlgorithmOptimizer, MemeticOptimizer


class _ConstantLossObjective:
    """A degenerate objective whose loss is always exactly 0.0.

    The stagnation check's relative improvement divides by ``abs(previous_score)``, which is exactly 0
    here. That must not silently produce ``nan`` (comparing ``False`` against tolerance and disabling
    early stopping): the run must converge via `patience` instead of always exhausting `max_iter`.
    """

    def logit_loss(self, weights: np.ndarray) -> float:
        return 0.0


class TestGeneticAlgorithmOptimizerStagnation:
    def test_constant_zero_loss_stops_early_via_patience(self):
        X = np.zeros((5, 3))
        optimizer = GeneticAlgorithmOptimizer(max_iter=50, patience=5, population_size=10, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = optimizer(_ConstantLossObjective(), X)
        assert result.success
        assert result.nit < 50

    def test_constant_zero_loss_no_invalid_value_warning(self):
        """A previous_score of 0.0 must not divide-by-zero into nan/inf (numpy RuntimeWarning)."""
        X = np.zeros((5, 3))
        optimizer = GeneticAlgorithmOptimizer(max_iter=10, patience=3, population_size=10, random_state=1)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            optimizer(_ConstantLossObjective(), X)


class _QuadraticGradObjective:
    """loss = sum(weights**2), gradient = 2 * weights.  Minimum at weights = 0.

    `logit_gradient_steps` mirrors the real `LogitObjective` contract: the returned generator is
    already primed (its first `next()` already consumed) so the caller can `send(theta)` directly.
    """

    def logit_loss(self, weights: np.ndarray) -> float:
        return float(np.sum(weights**2))

    def _steps(self):
        theta = yield
        while True:
            theta = yield 2.0 * theta

    def logit_gradient_steps(self):
        gen = self._steps()
        next(gen)
        return gen


class TestMemeticOptimizer:
    """`bounds` is a (min, max) tuple, matching GeneticAlgorithmOptimizer, and an invalid `optimizer`
    string fails fast at construction.
    """

    def test_bounds_is_tuple_not_float(self):
        X = np.zeros((5, 2))
        optimizer = MemeticOptimizer(bounds=(-3.0, 3.0), max_iter=3, population_size=10, random_state=0)
        result = optimizer(_QuadraticGradObjective(), X)
        assert result.x.shape == (2,)

    def test_success_is_false_when_max_iter_is_reached(self):
        X = np.zeros((5, 2))
        optimizer = MemeticOptimizer(max_iter=2, patience=10, population_size=10, random_state=0)
        result = optimizer(_QuadraticGradObjective(), X)
        assert result.success is False
        assert result.message == 'Maximum number of iterations reached.'

    def test_success_is_true_when_the_loss_stagnates(self):
        X = np.zeros((5, 2))
        optimizer = MemeticOptimizer(max_iter=1000, patience=3, tol=1e3, population_size=10, random_state=0)
        result = optimizer(_QuadraticGradObjective(), X)
        assert result.success is True
        assert result.nit < 1000

    def test_invalid_local_search_optimizer_raises_at_construction(self):
        with pytest.raises(ValueError, match="'adam' or 'sgd'"):
            MemeticOptimizer(optimizer='invalid')
