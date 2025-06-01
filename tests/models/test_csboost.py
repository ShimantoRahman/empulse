from unittest import mock

import numpy as np
import pytest
import sympy
from scipy.special import expit as sigmoid
from sklearn.datasets import make_classification

import empulse.models
from empulse.metrics import Metric
from empulse.models import CSBoostClassifier

# Define the classifiers to test
CLASSIFIERS = [('xgboost', 'XGBClassifier'), ('lightgbm', 'LGBMClassifier'), ('catboost', 'CatBoostClassifier')]


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    fn_cost = np.random.rand(y.size)
    fp_cost = 5
    return X, y, fn_cost, fp_cost


@pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
def test_csboost_different_classifiers(library, classifier_name, dataset):
    # Import the classifier dynamically
    classifier_module = pytest.importorskip(library)
    classifier_class = getattr(classifier_module, classifier_name)

    X, y, fn_cost, fp_cost = dataset
    model = CSBoostClassifier(estimator=classifier_class(n_estimators=2, verbose=0))
    model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == y.shape
    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


def test_csboost_when_xgboost_is_missing(dataset):
    X, y, fn_cost, fp_cost = dataset
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'XGBClassifier', None):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.'):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_metric_loss(dataset):
    X, y, accept_rate, customer_lifetime_value = dataset
    customer_lifetime_value *= 10

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    cost_loss = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .alias('accept_rate', gamma)
        .alias('incentive_cost', d)
        .alias('contact_cost', f)
        .build()
    )

    model = CSBoostClassifier(loss=cost_loss)
    model.fit(X, y, accept_rate=accept_rate, incentive_cost=10, clv=customer_lifetime_value, contact_cost=1)
    y_proba = model.predict_proba(X)[:, 1]
    # calibrate probabilities
    y_proba = sigmoid((y_proba - np.mean(y_proba)) / np.std(y_proba))
    score = cost_loss(
        y, y_proba, accept_rate=accept_rate, incentive_cost=10, clv=customer_lifetime_value, contact_cost=1
    )
    inverse_score = cost_loss(
        y, 1 - y_proba, accept_rate=accept_rate, incentive_cost=10, clv=customer_lifetime_value, contact_cost=1
    )
    assert score < inverse_score
