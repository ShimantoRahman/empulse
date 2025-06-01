import numpy as np
import pytest
import sympy
from sklearn.linear_model import LogisticRegression

from empulse.metrics import Metric
from empulse.models import CSThresholdClassifier


@pytest.fixture
def data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    return X, y


@pytest.mark.parametrize('calibrator', ['sigmoid', 'isotonic', None])
def test_fit(data, calibrator):
    X, y = data
    clf = CSThresholdClassifier(LogisticRegression(max_iter=2), calibrator=calibrator)
    clf.fit(X, y)
    assert hasattr(clf, 'estimator_')


@pytest.mark.parametrize(
    'tp_cost, tn_cost, fn_cost, fp_cost',
    [(1.0, 0.0, 5.0, 1.0), (0.0, 1.0, 1.0, 5.0), (2.0, 2.0, 2.0, 2.0), (1.0, 1.0, 1.0, 1.0)],
)
def test_predict(data, tp_cost, tn_cost, fn_cost, fp_cost):
    X, y = data
    clf = CSThresholdClassifier(LogisticRegression(max_iter=2), calibrator='sigmoid')
    clf.fit(X, y)
    y_pred = clf.predict(X, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert y_pred.shape == y.shape


def test_csthreshold_cost_loss(data):
    X, y = data
    rng = np.random.default_rng(42)
    clvs = rng.uniform(100, 200, size=y.shape[0])

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    cost_loss = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias('accept_rate', gamma)
        .alias('incentive_cost', d)
        .alias('contact_cost', f)
        .build()
    )

    tp_benefit = 0.3 * (clvs - 10 - 1) + (1 - 0.3) * -1
    tp_cost = -tp_benefit
    fp_cost = 10 + 1

    model_custom_loss = CSThresholdClassifier(LogisticRegression(), calibrator=None, loss=cost_loss)
    model_custom_loss.fit(X, y)
    preds1 = model_custom_loss.predict(X, accept_rate=0.3, incentive_cost=10, clv=clvs, contact_cost=1)
    model = CSThresholdClassifier(LogisticRegression(), calibrator=None)
    model.fit(X, y)
    preds2 = model.predict(X, tp_cost=tp_cost, fp_cost=fp_cost)
    assert np.allclose(preds1, preds2), 'Predictions from custom loss and default loss do not match.'
