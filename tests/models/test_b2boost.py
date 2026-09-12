from typing import TypeVar
from unittest import mock

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils.validation import NotFittedError, check_is_fitted
from xgboost import XGBClassifier

import empulse.models
from empulse.models import B2BoostClassifier


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


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    return X, y


def test_b2boost_when_xgboost_is_missing(dataset):
    X, y = dataset
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'XGBClassifier', TypeVar('XGBClassifier')):
        model = B2BoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use B2BoostClassifier.'):
            model.fit(X, y)
