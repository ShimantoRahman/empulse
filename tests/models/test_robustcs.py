import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import HuberRegressor

from empulse.metrics import Cost, CostMatrix, Metric, MixtureComponent, MixtureMetric
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


def test_robustcs_mixture_metric_loss(classification_data):
    """
    A ``MixtureMetric`` loss is imputed, not rejected.

    Previously ``RobustCSClassifier.fit`` raised ``NotImplementedError`` for any loss that
    wasn't a plain ``Metric``, since it could only walk a single ``CostMatrix``'s sympy
    expressions directly. ``BaseMetric._outlier_sensitive_parameters()`` now answers the same
    question uniformly, so a ``MixtureMetric`` (here, trivially, one component) works too.
    """
    X, y = classification_data

    clv = sympy.symbols('clv')
    cost_matrix = CostMatrix().add_tp_benefit(clv).add_fp_cost('b').mark_outlier_sensitive(clv)
    mixture = MixtureMetric([MixtureComponent(1.0, Metric(cost_matrix, Cost()), {})])

    rng = np.random.default_rng(42)
    clv_val = rng.uniform(100, 200, size=X.shape[0])
    model = RobustCSClassifier(CSLogitClassifier(loss=mixture))
    model.fit(X, y, clv=clv_val, b=1.0)
    assert hasattr(model, 'estimator_')
    assert hasattr(model, 'outlier_estimators_')
    assert not np.array_equal(model.costs_['clv'], clv_val)


@pytest.mark.parametrize(
    'labels',
    [np.array([-1, 1]), np.array(['no', 'yes']), np.array([2, 5])],
    ids=['minus_one_one', 'strings', 'two_five'],
)
def test_outlier_detection_does_not_depend_on_label_encoding(labels):
    """Regression test: the classes were selected with ``y > 0`` and ``y == 0``.

    With labels other than 0/1, no sample matched ``y == 0``, so the negative-class costs
    (``fp_cost``, ``tn_cost``) were silently never cleaned; with labels such as 2/5 every sample
    counted as positive.
    """
    X, y = make_classification(n_samples=300, random_state=0)
    fp_cost = np.random.default_rng(0).random(300)
    fp_cost[:5] = 1000.0  # outliers
    fn_cost = np.random.default_rng(1).random(300)
    fn_cost[5:10] = 1000.0

    reference = RobustCSClassifier(CSLogitClassifier()).fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    model = RobustCSClassifier(CSLogitClassifier()).fit(X, labels[y], fn_cost=fn_cost, fp_cost=fp_cost)

    for cost_name in ('fp_cost', 'fn_cost'):
        assert model.outlier_estimators_[cost_name] is not None
        np.testing.assert_allclose(model.costs_[cost_name], reference.costs_[cost_name])
    assert model.costs_['fp_cost'].max() < 1000.0


@pytest.mark.parametrize(
    'labels',
    [np.array([-1, 1]), np.array(['no', 'yes']), np.array([2, 5])],
    ids=['minus_one_one', 'strings', 'two_five'],
)
def test_metric_outlier_detection_does_not_depend_on_label_encoding(labels):
    """The metric-loss path selected each parameter's class the same way, with ``y > 0``/``y == 0``."""
    X, y = make_classification(n_samples=300, random_state=0)
    fn, fp = sympy.symbols('fn fp')
    # `fp` only appears in the false-positive cost, so its outliers are detected on the negatives.
    loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp).mark_outlier_sensitive(fp), Cost())
    fp_cost = np.random.default_rng(0).random(300)
    fp_cost[:5] = 1000.0

    reference = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, y, fn=1.0, fp=fp_cost)
    model = RobustCSClassifier(CSLogitClassifier(loss=loss)).fit(X, labels[y], fn=1.0, fp=fp_cost)

    assert 'fp' in model.outlier_estimators_
    np.testing.assert_allclose(model.costs_['fp'], reference.costs_['fp'])
    assert model.costs_['fp'].max() < 1000.0
