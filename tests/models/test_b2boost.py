import pytest
import numpy as np
from empulse.models import B2BoostClassifier
from xgboost import XGBClassifier
from sklearn.utils.validation import check_is_fitted, NotFittedError


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


@pytest.fixture(scope='module')
def clf(X, y):
    clf = B2BoostClassifier(n_estimators=2)
    clf.fit(X, y)
    return clf


def test_b2boost_init():
    clf = B2BoostClassifier()
    assert clf.clv == 200
    assert clf.incentive_cost == 10
    assert clf.contact_cost == 1
    assert clf.accept_rate == 0.3


def test_b2boost_with_different_parameters():
    clf = B2BoostClassifier(
        clv=100,
        incentive_cost=5,
        contact_cost=0.5,
        accept_rate=0.1,
        n_jobs=1,
        random_state=42
    )
    assert clf.clv == 100
    assert clf.incentive_cost == 5
    assert clf.contact_cost == 0.5
    assert clf.accept_rate == 0.1
    assert clf.model.n_jobs == 1
    assert clf.model.random_state == 42


def test_b2boost_fit(X, y):
    clf = B2BoostClassifier()
    clf.fit(X, y)
    assert isinstance(clf.model, XGBClassifier)
    assert clf.classes_ is not None
    try:
        check_is_fitted(clf.model)
    except NotFittedError:
        pytest.fail("XGBClassifier is not fitted")


def test_b2boost_predict_proba(clf, X):
    y_pred = clf.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_b2boost_predict(clf, X):
    y_pred = clf.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_b2boost_score(clf, X, y):
    score = clf.score(X, y)
    assert isinstance(score, float)


def test_cloneable_by_sklearn():
    from sklearn.base import clone
    clf = B2BoostClassifier(n_estimators=2)
    clf_clone = clone(clf)
    assert isinstance(clf_clone, B2BoostClassifier)
    assert clf.get_params() == clf_clone.get_params()


def test_works_in_cross_validation(X, y):
    from sklearn.model_selection import cross_val_score
    clf = B2BoostClassifier(n_estimators=2)
    scores = cross_val_score(clf, X, y, cv=2)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(scores.astype(np.float64) == scores)


def test_works_in_pipeline(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    clf = B2BoostClassifier(n_estimators=2)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X, y)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['clf'], B2BoostClassifier)
    assert isinstance(pipe.score(X, y), float)
    assert isinstance((pred := pipe.predict(X)), np.ndarray)
    assert np.all(~np.isnan(pred))


def test_works_in_ensemble(X, y):
    from sklearn.ensemble import BaggingClassifier
    clf = B2BoostClassifier(n_estimators=2)
    bagging = BaggingClassifier(clf, n_estimators=2)
    bagging.fit(X, y)
    assert isinstance(bagging.estimators_[0], B2BoostClassifier)
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)
