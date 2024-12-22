import numpy as np
import pytest

from empulse.samplers.bias_relabler import (
    _get_demotion_candidates,
    _get_promotion_candidates,
    _independent_pairs,
    BiasRelabler
)


def test_n_pairs_uneven():
    y_true = [1, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert _independent_pairs(y_true, protected_attr) == 1


def test_n_pairs_even():
    y_true = [1, 1, 1, 1, 0,
              1, 1, 1, 1, 0]
    protected_attr = np.array([1, 1, 1, 1, 1,
                               0, 0, 0, 0, 0])
    assert _independent_pairs(y_true, protected_attr) == 0


def test_no_protected_attr():
    y_true = [1, 1, 1, 1, 0,
              1, 1, 1, 1, 0]
    protected_attr = np.array([0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0])
    with pytest.warns(UserWarning, match="sensitive_feature only contains one class, no relabeling is performed."):
        assert _independent_pairs(y_true, protected_attr) == 0


def test_all_protected_attr():
    y_true = [1, 1, 1, 1, 0,
              1, 1, 1, 1, 0]
    protected_attr = np.array([1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1])
    with pytest.warns(UserWarning, match="sensitive_feature only contains one class, no relabeling is performed."):
        assert _independent_pairs(y_true, protected_attr) == 0


def test_demotion_pairs():
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_true = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    n_pairs = 2
    assert np.all(_get_demotion_candidates(y_pred, y_true, n_pairs) == np.array([1, 5]))


def test_promotion_pairs():
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_true = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    n_pairs = 2
    assert np.all(_get_promotion_candidates(y_pred, y_true, n_pairs) == np.array([3, 4]))


def test_bias_relabler():
    class MockEstimator:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            return np.column_stack((1 - proba, proba))

        def get_params(self, deep=False):
            return {}

    X = np.array([[0, 0] * 10]).reshape(10, 2)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    relabler = BiasRelabler(MockEstimator())
    X, y = relabler.fit_resample(X, y, sensitive_feature=protected_attr)
    assert np.all(y == np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1]))


def test_bias_relabler_two():
    class MockEstimator:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[::-1]
            return np.column_stack((1 - proba, proba))

        def get_params(self, deep=False):
            return {}

    X = np.array([[0, 0] * 10]).reshape(10, 2)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    relabler = BiasRelabler(MockEstimator())
    X, y = relabler.fit_resample(X, y, sensitive_feature=protected_attr)
    assert np.all(y == np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1]))
