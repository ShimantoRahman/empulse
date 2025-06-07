import pytest
from sklearn.linear_model import LogisticRegression

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
