import numpy as np
import pytest
from sklearn.datasets import make_classification

from empulse.models import CSTreeClassifier


@pytest.fixture
def data():
    return make_classification(n_samples=100, random_state=42)


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
def test_cstree_criteria(data, criterion):
    X, y = data
    model = CSTreeClassifier(criterion=criterion)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)
