import numpy as np
import pytest
from sklearn.datasets import make_classification

from empulse.models import CSLogitClassifier
from empulse.optimizers import LBFGSBOptimizer


@pytest.fixture
def data():
    return make_classification(n_samples=20, n_features=4, random_state=0)


class TestNormalizeCostShapes:
    """
    `Metric._prepare_parameters` accepts a length-1 array as shorthand for "one value applied to
    every sample", but `_normalize_cost_shapes` used to reject anything whose size wasn't exactly
    `n_samples`, so a plain length-1 cost array was rejected by the model even though the metric
    itself would have accepted it.
    """

    def test_length_one_array_cost_is_accepted(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.array([2.0]), fn_cost=5.0)

    def test_full_length_array_cost_is_accepted(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        model.fit(X, y, fp_cost=np.full(y.shape[0], 2.0), fn_cost=5.0)

    def test_wrong_length_array_cost_raises(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        with pytest.raises(ValueError, match='fp_cost'):
            model.fit(X, y, fp_cost=np.array([1.0, 2.0, 3.0]), fn_cost=5.0)

    def test_error_message_names_expected_lengths(self, data):
        X, y = data
        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=5))
        with pytest.raises(ValueError, match=f'expected length {y.shape[0]}'):
            model.fit(X, y, fp_cost=np.array([1.0, 2.0]), fn_cost=5.0)
