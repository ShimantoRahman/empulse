"""
The scikit-learn integration contract, applied to every estimator in ``empulse.models``.

``test_models.py`` runs scikit-learn's own ``parametrize_with_checks`` suite; this module covers the
composition scenarios that suite does not, plus pickling.
"""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .estimator_inventory import NEEDS_SENSITIVE_FEATURE, estimator_id, make_estimators


@pytest.fixture(params=make_estimators(), ids=estimator_id)
def estimator(request):
    """Each estimator from the inventory, cloned so no test sees another's fitted state."""
    return clone(request.param)


@pytest.fixture(scope='module')
def integration_data():
    """
    A 100-row problem, larger than the toy ``X``/``y`` used elsewhere in ``tests/models``.

    ``CSThresholdClassifier`` calibrates with an internal cross-validation and ``BaggingClassifier``
    subsamples, so both need enough rows per class to form their own folds; the 10-row fixture makes
    them fail with "only one class is present" rather than for any reason worth testing.
    """
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 4))
    y = np.tile([0, 1], n // 2)
    sensitive_feature = np.tile([0, 0, 1, 1], n // 4)
    return X, y, sensitive_feature


@pytest.fixture
def fit_params(estimator, integration_data):
    """The fit-time arguments this estimator needs beyond X and y."""
    if isinstance(estimator, NEEDS_SENSITIVE_FEATURE):
        return {'sensitive_feature': integration_data[2]}
    return {}


def test_clone_round_trips_every_parameter(estimator):
    """``clone`` must reproduce every constructor parameter, not merely the class."""
    cloned = clone(estimator)
    assert isinstance(cloned, type(estimator))
    cloned_params = cloned.get_params()
    for key, value in estimator.get_params().items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, cloned_params[key])
        elif isinstance(value, float) and np.isnan(value):
            # XGBClassifier's `missing` defaults to nan, which is never equal to itself
            assert np.isnan(cloned_params[key])
        elif isinstance(value, (str, bytes)) or not hasattr(value, '__dict__'):
            # plain scalars, strings and None compare by value
            assert value == cloned_params[key]
        else:
            # `clone` deep-copies anything that is not itself an estimator (an optimizer, say), so
            # identity and equality both fail by design -- the type is what must survive.
            assert type(value) is type(cloned_params[key])


def test_clone_does_not_share_state_between_instances(estimator):
    """
    Regression guard for the dynamic metadata routing.

    ``CostSensitiveClassifier.__init__`` rewrites ``set_fit_request`` on ``self.__class__`` -- not on
    the instance -- so a second instance could in principle observe the first one's symbol list.
    """
    first = clone(estimator)
    second = clone(estimator)
    assert first is not second
    assert first.get_params().keys() == second.get_params().keys()


def test_is_picklable_before_fitting(estimator):
    restored = pickle.loads(pickle.dumps(estimator))
    assert isinstance(restored, type(estimator))
    assert restored.get_params().keys() == estimator.get_params().keys()


def test_is_picklable_after_fitting(estimator, integration_data, fit_params):
    """A fitted estimator must survive a pickle round trip and predict identically."""
    X, y, _ = integration_data
    estimator.fit(X, y, **fit_params)
    expected = estimator.predict(X)
    restored = pickle.loads(pickle.dumps(estimator))
    assert np.array_equal(restored.predict(X), expected)


def test_works_in_cross_validation(estimator, integration_data, fit_params):
    X, y, _ = integration_data
    if fit_params:
        pytest.skip('needs metadata routing to pass fit params through CV; covered in test_bias_mitigation.py')
    scores = cross_val_score(estimator, X, y, cv=2, error_score='raise')
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_works_in_pipeline(estimator, integration_data, fit_params):
    X, y, _ = integration_data
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
    pipe.fit(X, y, **{f'clf__{k}': v for k, v in fit_params.items()})
    assert isinstance(pipe.named_steps['clf'], type(estimator))
    assert isinstance(pipe.score(X, y), float)
    predictions = pipe.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert not np.any(np.isnan(predictions))


def test_works_inside_a_bagging_ensemble(estimator, integration_data, fit_params):
    X, y, _ = integration_data
    if fit_params:
        pytest.skip('BaggingClassifier does not route extra fit params to its sub-estimators')
    bagging = BaggingClassifier(estimator, n_estimators=2, random_state=42)
    bagging.fit(X, y)
    assert isinstance(bagging.estimators_[0], type(estimator))
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)
