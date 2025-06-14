import numpy as np
import pytest
import sympy
from sklearn.linear_model import HuberRegressor

from empulse.metrics import Cost, Metric
from empulse.models import CSLogitClassifier, RobustCSClassifier


@pytest.fixture
def data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y


@pytest.mark.parametrize(
    'tp_cost, tn_cost, fn_cost, fp_cost',
    [
        (0.0, 0.0, np.random.rand(100), 0.0),
        (np.random.rand(100), 0.0, 0.0, np.random.rand(100)),
        (0.0, np.random.rand(100), np.random.rand(100), 0.0),
        (np.random.rand(100), np.random.rand(100), np.random.rand(100), np.random.rand(100)),
    ],
)
def test_fit(data, tp_cost, tn_cost, fn_cost, fp_cost):
    X, y = data
    clf = RobustCSClassifier(CSLogitClassifier(), HuberRegressor(), detect_outliers_for='all')
    clf.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert hasattr(clf, 'estimator_')
    assert hasattr(clf, 'outlier_estimators_')
    for cost_name, original_cost in zip(
        ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=False
    ):
        if isinstance(original_cost, np.ndarray):
            assert not np.array_equal(clf.costs_[cost_name], original_cost)
        else:
            assert clf.costs_[cost_name] == original_cost


@pytest.mark.parametrize('detect_outliers_for', ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost', ['tp_cost', 'fn_cost']])
def test_detect_outliers_for(data, detect_outliers_for):
    X, y = data
    tp_cost = np.random.rand(100)
    tn_cost = np.random.rand(100)
    fn_cost = np.random.rand(100)
    fp_cost = np.random.rand(100)
    clf = RobustCSClassifier(CSLogitClassifier(), HuberRegressor(), detect_outliers_for=detect_outliers_for)
    clf.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert hasattr(clf, 'estimator_')
    assert hasattr(clf, 'outlier_estimators_')
    for cost_name, original_cost in zip(
        ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=False
    ):
        if isinstance(original_cost, np.ndarray) and cost_name in detect_outliers_for:
            assert not np.array_equal(clf.costs_[cost_name], original_cost)
        elif isinstance(original_cost, np.ndarray) and cost_name not in detect_outliers_for:
            assert np.array_equal(clf.costs_[cost_name], original_cost)


def test_robustcs_metric_loss(data):
    X, y = data

    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    cost_loss = (
        Metric(Cost())
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .alias('accept_rate', gamma)
        .alias('incentive_cost', d)
        .alias('contact_cost', f)
        .mark_outlier_sensitive(clv)
        .mark_outlier_sensitive(d)
        .build()
    )
    rng = np.random.default_rng(42)
    clv_val = rng.uniform(100, 200, size=X.shape[0])
    d_val = rng.uniform(0, 100, size=X.shape[0])
    model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
    model.fit(X, y, clv=clv_val, accept_rate=0.3, incentive_cost=d_val, contact_cost=1)
    assert hasattr(model, 'estimator_')
    assert hasattr(model, 'outlier_estimators_')
    assert not np.array_equal(model.costs_['clv'], clv_val)
    assert not np.array_equal(model.costs_['d'], d_val)
