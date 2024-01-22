import pytest
import numpy as np
from scipy.optimize import OptimizeResult

from empulse.models import ProfLogitClassifier


@pytest.fixture(scope='module')
def clf():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(max_iter=10)
    clf.fit(X, y)
    return clf


def test_proflogit_init():
    clf = ProfLogitClassifier()
    assert clf.C == 1.0
    assert clf.fit_intercept is True
    assert clf.soft_threshold is True
    assert clf.l1_ratio == 1.0
    assert clf.n_jobs is None
    assert clf.default_bounds == (-3, 3)


def test_proflogit_fit(clf):
    assert clf.n_dim == 3
    assert isinstance(clf.result, OptimizeResult)


def test_proflogit_fit_no_intercept():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(fit_intercept=False, max_iter=10)
    clf.fit(X, y)
    assert clf.n_dim == 2
    assert isinstance(clf.result, OptimizeResult)


def test_proflogit_predict_proba(clf):
    X = np.random.rand(10, 2)
    y_pred = clf.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_proflogit_predict(clf):
    X = np.random.rand(10, 2)
    y_pred = clf.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_proflogit_score(clf):
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    score = clf.score(X, y)
    assert isinstance(score, float)


def test_proflogit_with_missing_values():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    # Introduce missing values
    X[0, 0] = np.nan
    clf = ProfLogitClassifier(max_iter=10)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_proflogit_with_different_C():
    clf = ProfLogitClassifier(C=0.5)
    assert clf.C == 0.5
