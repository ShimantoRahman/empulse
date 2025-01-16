import pytest
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from empulse.datasets import load_give_me_some_credit
from empulse.metrics import cost_loss
from empulse.models.cost_sensitive.cost_threshold import CSThresholdClassifier


@pytest.fixture
def data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    return X, y


@pytest.mark.parametrize('calibrator', ['sigmoid', 'isotonic', None])
def test_fit(data, calibrator):
    X, y = data
    clf = CSThresholdClassifier(LogisticRegression(), calibrator=calibrator)
    clf.fit(X, y)
    assert hasattr(clf, 'estimator_')


@pytest.mark.parametrize(
    'tp_cost, tn_cost, fn_cost, fp_cost',
    [(1.0, 0.0, 5.0, 1.0), (0.0, 1.0, 1.0, 5.0), (2.0, 2.0, 2.0, 2.0), (1.0, 1.0, 1.0, 1.0)],
)
def test_predict(data, tp_cost, tn_cost, fn_cost, fp_cost):
    X, y = data
    clf = CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid')
    clf.fit(X, y)
    y_pred = clf.predict(X, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    assert y_pred.shape == y.shape


def test_expected_cost_loss(data):
    X, y, tp_cost, fp_cost, tn_cost, fn_cost = load_give_me_some_credit(return_X_y_costs=True)

    # Regular Logistic Regression
    lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    cost_loss_lr = cost_loss(
        y, y_pred_lr, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, normalize=True
    )

    # Cost-Sensitive Threshold Classifier
    cs_clf = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('clf', CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid', random_state=42)),
        ]
    )
    cs_clf.fit(X, y)
    y_pred_cs = cs_clf.predict(X, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    cost_loss_cs = cost_loss(
        y, y_pred_cs, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, normalize=True
    )

    assert cost_loss_cs < cost_loss_lr, 'CS classifier should perform better than regular logistic regression'
