import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._param_validation import InvalidParameterError

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
    classifier_module = __import__(library, fromlist=[classifier_name])
    classifier_class = getattr(classifier_module, classifier_name)

    X, y, fn_cost, fp_cost = dataset
    model = CSBoostClassifier(estimator=classifier_class(n_estimators=2, verbose=0))
    model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == y.shape
    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


INVALID_PARAMS = [
    {'tp_cost': '5'},
    {'tn_cost': '5'},
    {'fp_cost': '5'},
    {'fn_cost': '5'},
]


@pytest.mark.parametrize('invalid_params', INVALID_PARAMS)
def test_csboost_invalid_params(invalid_params, dataset):
    X, y, _, _ = dataset
    model = CSBoostClassifier(**invalid_params)
    with pytest.raises(InvalidParameterError):
        model.fit(X, y)
