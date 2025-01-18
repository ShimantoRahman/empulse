import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.validation import NotFittedError, check_is_fitted
from xgboost import XGBClassifier

from empulse.models import B2BoostClassifier


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


@pytest.fixture(scope='module')
def clf(X, y):
    clf = B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1))
    clf.fit(X, y)
    return clf


def test_b2boost_init():
    clf = B2BoostClassifier()
    assert clf.clv == 200
    assert clf.incentive_fraction == 0.05
    assert clf.contact_cost == 15
    assert clf.accept_rate == 0.3


def test_b2boost_with_different_parameters():
    clf = B2BoostClassifier(
        XGBClassifier(n_jobs=1, random_state=42),
        clv=100,
        incentive_fraction=0.1,
        contact_cost=0.5,
        accept_rate=0.1,
    )
    assert clf.clv == 100
    assert clf.incentive_fraction == 0.1
    assert clf.contact_cost == 0.5
    assert clf.accept_rate == 0.1
    assert clf.estimator.n_jobs == 1
    assert clf.estimator.random_state == 42


def test_b2boost_fit(X, y):
    clf = B2BoostClassifier()
    clf.fit(X, y)
    assert isinstance(clf.estimator_, XGBClassifier)
    assert clf.classes_ is not None
    try:
        check_is_fitted(clf.estimator_)
    except AttributeError:  # TODO: remove when XGBClassifier is fixed
        pass
    except NotFittedError:
        pytest.fail('XGBClassifier is not fitted')


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

    clf = B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1))
    clf_clone = clone(clf)
    assert isinstance(clf_clone, B2BoostClassifier)


def test_works_in_cross_validation(X, y):
    from sklearn.model_selection import cross_val_score

    clf = B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1))
    scores = cross_val_score(clf, X, y, cv=2)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(scores.astype(np.float64) == scores)


def test_works_in_pipeline(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1))
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X, y)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['clf'], B2BoostClassifier)
    assert isinstance(pipe.score(X, y), float)
    assert isinstance((pred := pipe.predict(X)), np.ndarray)
    assert np.all(~np.isnan(pred))


def test_works_in_ensemble(X, y):
    from sklearn.ensemble import BaggingClassifier

    clf = B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1))
    bagging = BaggingClassifier(clf, n_estimators=2, random_state=42)
    bagging.fit(X, y)
    assert isinstance(bagging.estimators_[0], B2BoostClassifier)
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)


# Define the classifiers to test
CLASSIFIERS = [('xgboost', 'XGBClassifier'), ('lightgbm', 'LGBMClassifier'), ('catboost', 'CatBoostClassifier')]


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    return X, y


@pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
def test_b2boost_different_classifiers(library, classifier_name, dataset):
    # Import the classifier dynamically
    classifier_module = __import__(library, fromlist=[classifier_name])
    classifier_class = getattr(classifier_module, classifier_name)

    X, y = dataset
    model = B2BoostClassifier(estimator=classifier_class(n_estimators=2, verbose=0))
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == y.shape
    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


INVALID_PARAMS = [
    {'clv': '5'},
    {'accept_rate': '5'},
    {'incentive_fraction': '5'},
    {'contact_cost': '5'},
]


@pytest.mark.parametrize('invalid_params', INVALID_PARAMS)
def test_b2boost_invalid_params(invalid_params, dataset):
    X, y = dataset
    model = B2BoostClassifier(**invalid_params)
    with pytest.raises(InvalidParameterError):
        model.fit(X, y)
