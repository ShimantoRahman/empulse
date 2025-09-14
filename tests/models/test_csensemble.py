import numpy as np
import pytest
from sklearn.datasets import make_classification

from empulse.models import CSBaggingClassifier, CSForestClassifier


@pytest.fixture
def data():
    return make_classification(n_samples=100, random_state=42)


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
def test_csforest_criteria(data, criterion):
    X, y = data
    model = CSForestClassifier(criterion=criterion)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csforest_combination(data, combination):
    X, y = data
    model = CSForestClassifier(combination=combination)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csbagging_combination(data, combination):
    X, y = data
    model = CSBaggingClassifier(combination=combination)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)
