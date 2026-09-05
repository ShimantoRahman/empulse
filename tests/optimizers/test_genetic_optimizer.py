import warnings

import numpy as np

from empulse.optimizers import GeneticAlgorithmOptimizer


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
