import numpy as np
import pytest
from sklearn.linear_model import HuberRegressor
from empulse.models import RobustCSClassifier, CSLogitClassifier

@pytest.fixture
def data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.mark.parametrize("tp_cost, tn_cost, fn_cost, fp_cost", [
    (0.0, 0.0, np.random.rand(100), 0.0),
    (np.random.rand(100), 0.0, 0.0, np.random.rand(100)),
    (0.0, np.random.rand(100), np.random.rand(100), 0.0),
    (np.random.rand(100), np.random.rand(100), np.random.rand(100), np.random.rand(100))
])
def test_fit(data, tp_cost, tn_cost, fn_cost, fp_cost):
    X, y = data
    clf = RobustCSClassifier(CSLogitClassifier(), HuberRegressor(), detect_outliers_for='all')
    clf.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert hasattr(clf, "estimator_")
    assert hasattr(clf, "outlier_estimators_")
    for cost_name, original_cost in zip(['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost]):
        if isinstance(original_cost, np.ndarray):
            assert not np.array_equal(clf.costs_[cost_name], original_cost)
        else:
            assert clf.costs_[cost_name] == original_cost

@pytest.mark.parametrize("detect_outliers_for", [
    'tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', ['tp_cost', 'fn_cost']
])
def test_detect_outliers_for(data, detect_outliers_for):
    X, y = data
    tp_cost = np.random.rand(100)
    tn_cost = np.random.rand(100)
    fn_cost = np.random.rand(100)
    fp_cost = np.random.rand(100)
    clf = RobustCSClassifier(CSLogitClassifier(), HuberRegressor(), detect_outliers_for=detect_outliers_for)
    clf.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert hasattr(clf, "estimator_")
    assert hasattr(clf, "outlier_estimators_")
    for cost_name, original_cost in zip(['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost]):
        if isinstance(original_cost, np.ndarray) and cost_name in detect_outliers_for:
            assert not np.array_equal(clf.costs_[cost_name], original_cost)
        elif isinstance(original_cost, np.ndarray) and cost_name not in detect_outliers_for:
            assert np.array_equal(clf.costs_[cost_name], original_cost)
