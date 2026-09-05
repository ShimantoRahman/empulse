import warnings

import numpy as np
import pytest

from empulse.optimizers import GeneticAlgorithmOptimizer, MemeticOptimizer


class _ConstantLossObjective:
    """A degenerate objective whose loss is always exactly 0.0.

    Used to reproduce MODELS_OPTIMIZERS_REVIEW.md item 18: the stagnation check's relative
    improvement used to divide by ``abs(previous_score)`` directly, which is exactly 0 here and
    silently produces ``nan`` (comparing ``False`` against tolerance and disabling early stopping)
    rather than raising, so the regression is that the run must actually converge via `patience`
    instead of always exhausting `max_iter`.
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
    """MODELS_OPTIMIZERS_REVIEW.md item 40: bounds should be a (min, max) tuple, matching
    GeneticAlgorithmOptimizer, and an invalid `optimizer` string should fail fast at construction.
    """

    def test_bounds_is_tuple_not_float(self):
        X = np.zeros((5, 2))
        optimizer = MemeticOptimizer(bounds=(-3.0, 3.0), max_iter=3, population_size=10, random_state=0)
        result = optimizer(_QuadraticGradObjective(), X)
        assert result.x.shape == (2,)

    def test_invalid_local_search_optimizer_raises_at_construction(self):
        with pytest.raises(ValueError, match="'adam' or 'sgd'"):
            MemeticOptimizer(optimizer='invalid')
