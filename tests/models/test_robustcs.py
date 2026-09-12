import numpy as np
import pytest
import sympy
from sklearn.linear_model import HuberRegressor

from empulse.metrics import Cost, CostMatrix, Metric
from empulse.models import CSLogitClassifier, RobustCSClassifier

ARRAY_COST_CASES = [
    ('fn_cost',),
    ('tp_cost', 'fp_cost'),
    ('tn_cost', 'fn_cost'),
    ('tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'),
]
COST_NAMES = ('tp_cost', 'tn_cost', 'fn_cost', 'fp_cost')


@pytest.mark.parametrize('array_costs', ARRAY_COST_CASES, ids='+'.join)
def test_fit(classification_data, seeded_rng, array_costs):
    X, y = classification_data
    costs = {name: (seeded_rng.random(100) if name in array_costs else 0.0) for name in COST_NAMES}
    tp_cost, tn_cost, fn_cost, fp_cost = (costs[name] for name in COST_NAMES)
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
def test_detect_outliers_for(classification_data, seeded_rng, detect_outliers_for):
    X, y = classification_data
    tp_cost, tn_cost, fn_cost, fp_cost = (seeded_rng.random(100) for _ in range(4))
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


def test_robustcs_metric_loss(classification_data):
    X, y = classification_data

    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .alias('accept_rate', gamma)
        .alias('incentive_cost', d)
        .alias('contact_cost', f)
        .mark_outlier_sensitive(clv)
        .mark_outlier_sensitive(d)
    )
    cost_loss = Metric(cost_matrix, Cost())
    rng = np.random.default_rng(42)
    clv_val = rng.uniform(100, 200, size=X.shape[0])
    d_val = rng.uniform(0, 100, size=X.shape[0])
    model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
    model.fit(X, y, clv=clv_val, accept_rate=0.3, incentive_cost=d_val, contact_cost=1)
    assert hasattr(model, 'estimator_')
    assert hasattr(model, 'outlier_estimators_')
    assert not np.array_equal(model.costs_['clv'], clv_val)
    assert not np.array_equal(model.costs_['incentive_cost'], d_val)
