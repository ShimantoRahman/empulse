import pytest

import numpy as np
from sklearn.utils.validation import check_is_fitted, NotFittedError
from sklearn.linear_model import LogisticRegression

from empulse.models import BiasResamplingClassifier


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


@pytest.fixture(scope='module')
def clf(X, y):
    clf = BiasResamplingClassifier(estimator=LogisticRegression(), strategy='statistical parity')
    clf.fit(X, y)
    return clf


def test_resampling_init():
    clf = BiasResamplingClassifier(estimator=LogisticRegression(), strategy='statistical parity')
    assert isinstance(clf.estimator, LogisticRegression)
    assert clf.transform_attr is None
    assert clf.strategy == 'statistical parity'


def test_resampling_with_different_parameters():
    clf = BiasResamplingClassifier(
        estimator=LogisticRegression(),
        strategy='demographic parity',
        transform_attr=lambda x: x
    )
    assert isinstance(clf.estimator, LogisticRegression)
    assert clf.transform_attr is not None
    assert isinstance(clf.transform_attr, type(lambda x: x))
    assert clf.strategy == 'demographic parity'


def test_resampling_fit(X, y):
    clf = BiasResamplingClassifier(estimator=LogisticRegression())
    clf.fit(X, y)
    assert clf.classes_ is not None
    try:
        check_is_fitted(clf.estimator)
    except NotFittedError:
        pytest.fail("BiasResamplingClassifier is not fitted")


def test_resampling_predict_proba(clf, X):
    y_pred = clf.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_resampling_predict(clf, X):
    y_pred = clf.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_resampling_score(clf, X, y):
    score = clf.score(X, y)
    assert isinstance(score, float)


def test_cloneable_by_sklearn():
    from sklearn.base import clone
    clf = BiasResamplingClassifier(estimator=LogisticRegression())
    clf_clone = clone(clf)
    assert isinstance(clf_clone, BiasResamplingClassifier)
    cloned_params = clf_clone.get_params()
    for key, value in clf.get_params().items():
        if key == 'estimator':
            assert value.__class__ == cloned_params[key].__class__
        else:
            assert value == cloned_params[key]


def test_works_in_cross_validation(X, y):
    from sklearn.model_selection import cross_val_score
    clf = BiasResamplingClassifier(estimator=LogisticRegression())
    scores = cross_val_score(clf, X, y, cv=2)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(scores.astype(np.float64) == scores)


def test_works_in_pipeline(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    clf = BiasResamplingClassifier(estimator=LogisticRegression())
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X, y)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['clf'], BiasResamplingClassifier)
    assert isinstance(pipe.score(X, y), float)
    assert isinstance((pred := pipe.predict(X)), np.ndarray)
    assert np.all(~np.isnan(pred))


def test_works_in_ensemble(X, y):
    from sklearn.ensemble import BaggingClassifier
    clf = BiasResamplingClassifier(estimator=LogisticRegression())
    bagging = BaggingClassifier(clf, n_estimators=2)
    bagging.fit(X, y)
    assert isinstance(bagging.estimators_[0], BiasResamplingClassifier)
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)
