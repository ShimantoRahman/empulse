from typing import TypeVar
from unittest import mock

import numpy as np
import pytest
from sklearn.datasets import make_classification

import empulse.models
from empulse.models import CSBoostClassifier

# Define the classifiers to test
CLASSIFIERS = [('xgboost', 'XGBClassifier'), ('lightgbm', 'LGBMClassifier'), ('catboost', 'CatBoostClassifier')]


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    fn_cost = np.random.rand(y.size)
    fp_cost = 5
    return X, y, fn_cost, fp_cost


@pytest.mark.filterwarnings('ignore::UserWarning')
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
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'XGBClassifier', TypeVar('XGBClassifier')):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.'):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_when_lightgbm_is_missing_with_lgbm_estimator(dataset):
    """Test that CSBoostClassifier raises ValueError when LightGBM is missing but LGBMClassifier is passed."""
    X, y, fn_cost, fp_cost = dataset

    # Mock LGBMClassifier to be TypeVar (simulating it's not installed)
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'LGBMClassifier', TypeVar('LGBMClassifier')):
        # Create a mock estimator that would fail the isinstance check
        mock_estimator = mock.Mock()
        model = CSBoostClassifier(estimator=mock_estimator)

        with pytest.raises(
            ValueError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
        ):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_when_catboost_is_missing_with_catboost_estimator(dataset):
    """Test that CSBoostClassifier raises ValueError when CatBoost is missing but CatBoostClassifier is passed."""
    X, y, fn_cost, fp_cost = dataset

    # Mock CatBoostClassifier to be TypeVar (simulating it's not installed)
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'CatBoostClassifier', TypeVar('CatBoostClassifier')):
        # Create a mock estimator that would fail the isinstance check
        mock_estimator = mock.Mock()
        model = CSBoostClassifier(estimator=mock_estimator)

        with pytest.raises(
            ValueError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
        ):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_with_invalid_estimator_type(dataset):
    """Test that CSBoostClassifier raises ValueError when an unsupported estimator type is provided."""
    X, y, fn_cost, fp_cost = dataset

    from sklearn.ensemble import RandomForestClassifier

    model = CSBoostClassifier(estimator=RandomForestClassifier())

    with pytest.raises(
        ValueError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
    ):
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_when_all_libraries_missing(dataset):
    """Test that CSBoostClassifier fails gracefully when all boosting libraries are missing."""
    X, y, fn_cost, fp_cost = dataset

    with (
        mock.patch.object(empulse.models.cost_sensitive.csboost, 'XGBClassifier', TypeVar('XGBClassifier')),
        mock.patch.object(empulse.models.cost_sensitive.csboost, 'LGBMClassifier', TypeVar('LGBMClassifier')),
        mock.patch.object(empulse.models.cost_sensitive.csboost, 'CatBoostClassifier', TypeVar('CatBoostClassifier')),
    ):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.'):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


@pytest.mark.parametrize(
    'missing_library,library_name',
    [
        ('XGBClassifier', 'XGBoost'),
        ('LGBMClassifier', 'LightGBM'),
        ('CatBoostClassifier', 'CatBoost'),
    ],
)
def test_csboost_import_error_message_quality(dataset, missing_library, library_name):
    """Test that import error messages are informative and include installation instructions."""
    X, y, fn_cost, fp_cost = dataset

    with mock.patch.object(empulse.models.cost_sensitive.csboost, missing_library, TypeVar(missing_library)):
        if missing_library == 'XGBClassifier':
            model = CSBoostClassifier()
            with pytest.raises(ImportError) as exc_info:
                model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

            # Check that the error message contains helpful information
            error_message = str(exc_info.value)
            assert 'required' in error_message.lower()
            assert 'install' in error_message.lower() or 'pip install' in error_message.lower()
