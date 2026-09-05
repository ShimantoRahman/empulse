import threading

import numpy as np
import pytest

from empulse.metrics.metric.common import Direction
from empulse.models.cost_sensitive._ensemble_weighting import (
    accumulate_weighted_prediction,
    goodness_weights,
    subset_loss_params,
)


class TestGoodnessWeights:
    """Regression tests for OOB weighting giving more weight to worse estimators.

    `goodness_weights` must honour `direction` (a loss should be inverted before "more weight =
    better") and must not blow up when normalizing signed values.
    """

    def test_loss_lower_is_better(self):
        """For a MINIMIZE metric (a loss), the lowest value should get the highest weight."""
        losses = np.array([1.0, 5.0, 0.5, 10.0])
        weights = goodness_weights(losses, Direction.MINIMIZE)
        assert weights.argmax() == 2  # lowest loss
        assert weights.argmin() == 3  # highest loss

    def test_score_higher_is_better(self):
        """For a MAXIMIZE metric (a score), the highest value should get the highest weight."""
        scores = np.array([1.0, 5.0, 0.5, 10.0])
        weights = goodness_weights(scores, Direction.MAXIMIZE)
        assert weights.argmax() == 3  # highest score
        assert weights.argmin() == 2  # lowest score

    @pytest.mark.parametrize('direction', [Direction.MINIMIZE, Direction.MAXIMIZE])
    def test_weights_are_non_negative_and_sum_to_one(self, direction):
        values = np.array([-3.0, 0.5, 2.0, -1.0])
        weights = goodness_weights(values, direction)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize('direction', [Direction.MINIMIZE, Direction.MAXIMIZE])
    def test_signed_values_do_not_produce_negative_weights(self, direction):
        """Normalizing signed values by their raw sum can go negative; goodness-shifting must not."""
        values = np.array([-10.0, -5.0, 3.0, 8.0])
        weights = goodness_weights(values, direction)
        assert (weights >= 0).all()

    @pytest.mark.parametrize('direction', [Direction.MINIMIZE, Direction.MAXIMIZE])
    def test_identical_values_fall_back_to_uniform(self, direction):
        values = np.array([2.0, 2.0, 2.0])
        weights = goodness_weights(values, direction)
        np.testing.assert_allclose(weights, np.full(3, 1 / 3))

    @pytest.mark.parametrize('direction', [Direction.MINIMIZE, Direction.MAXIMIZE])
    def test_non_finite_values_fall_back_to_uniform(self, direction):
        values = np.array([1.0, np.nan, 3.0])
        weights = goodness_weights(values, direction)
        np.testing.assert_allclose(weights, np.full(3, 1 / 3))


class TestSubsetLossParams:
    """Regression tests for instance-dependent loss params not being subset to OOB rows."""

    def test_array_matching_n_samples_is_subset_by_index(self):
        params = {'fn_cost': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        index = np.array([0, 2, 4])
        result = subset_loss_params(params, index, n_samples=5)
        np.testing.assert_array_equal(result['fn_cost'], [1.0, 3.0, 5.0])

    def test_array_matching_n_samples_is_subset_by_boolean_mask(self):
        params = {'fn_cost': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        mask = np.array([True, False, True, False, True])
        result = subset_loss_params(params, mask, n_samples=5)
        np.testing.assert_array_equal(result['fn_cost'], [1.0, 3.0, 5.0])

    def test_scalar_passes_through_unchanged(self):
        params = {'fn_cost': 5.0}
        result = subset_loss_params(params, np.array([0, 2]), n_samples=5)
        assert result['fn_cost'] == 5.0

    def test_array_not_matching_n_samples_passes_through_unchanged(self):
        """An array parameter unrelated to the per-sample axis (e.g. a fixed-length constant) is untouched."""
        params = {'class_priors': np.array([0.3, 0.7])}
        result = subset_loss_params(params, np.array([0, 2]), n_samples=5)
        np.testing.assert_array_equal(result['class_priors'], [0.3, 0.7])

    def test_mixed_params(self):
        params = {'fn_cost': np.arange(5, dtype=np.float64), 'fp_cost': 1.0}
        index = np.array([1, 3])
        result = subset_loss_params(params, index, n_samples=5)
        np.testing.assert_array_equal(result['fn_cost'], [1.0, 3.0])
        assert result['fp_cost'] == 1.0


def test_accumulate_weighted_prediction():
    out = np.zeros((3, 2))
    lock = threading.Lock()

    def predict(X):
        return np.full((3, 2), 2.0)

    accumulate_weighted_prediction(predict, np.zeros((3, 1)), out, weight=0.5, lock=lock)
    np.testing.assert_allclose(out, np.full((3, 2), 1.0))
