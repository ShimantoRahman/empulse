"""The profit-driven models predict the positive class above a threshold learned from their training scores."""

import numpy as np
import pytest
from sklearn.base import clone

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.metrics.metric.prebuilt_metrics import make_generic_metric
from empulse.models import (
    CSBoostClassifier,
    CSLogitClassifier,
    CSTreeClassifier,
    ProfLogitClassifier,
    ProfMEMPMClassifier,
    ProfMPMClassifier,
    ProfSRClassifier,
    ProfTreeClassifier,
)
from empulse.optimizers import GeneticAlgorithmOptimizer

COSTS = {'tp_cost': -20.0, 'fp_cost': 4.0}

MODELS = [
    pytest.param(
        lambda **kw: ProfLogitClassifier(
            optimizer=GeneticAlgorithmOptimizer(max_iter=5, population_size=20, random_state=0), **kw
        ),
        id='ProfLogit',
    ),
    pytest.param(lambda **kw: ProfTreeClassifier(max_iter=5, population_size=10, random_state=0, **kw), id='ProfTree'),
    pytest.param(lambda **kw: ProfSRClassifier(max_iter=3, population_size=30, random_state=0, **kw), id='ProfSR'),
    pytest.param(ProfMPMClassifier, id='ProfMPM'),
    pytest.param(ProfMEMPMClassifier, id='ProfMEMPM'),
]


@pytest.fixture(scope='module')
def data(make_data):
    return make_data(n_samples=300, n_features=5, weights=[0.8], random_state=3)


def _positive_proba(model, X):
    return model.predict_proba(X)[:, 1]


@pytest.mark.parametrize('make', MODELS)
def test_auto_threshold_costs_the_training_samples_least(make, data):
    X, y = data
    model = make().fit(X, y, **COSTS)

    expected = make_generic_metric(MaxProfit()).model.decision_threshold(y, _positive_proba(model, X), **COSTS)
    assert model.threshold_ == expected
    np.testing.assert_array_equal(model.predict(X), (_positive_proba(model, X) > model.threshold_).astype(int))


@pytest.mark.parametrize('make', MODELS)
def test_one_half_predicts_the_most_probable_class(make, data):
    X, y = data
    model = make(threshold=0.5).fit(X, y, **COSTS)

    assert model.threshold_ == 0.5
    np.testing.assert_array_equal(model.predict(X), np.argmax(model.predict_proba(X), axis=1))


@pytest.mark.parametrize('make', MODELS)
def test_a_given_threshold_is_applied_as_is(make, data):
    X, y = data
    model = make(threshold=0.3).fit(X, y, **COSTS)

    assert model.threshold_ == 0.3
    np.testing.assert_array_equal(model.predict(X), (_positive_proba(model, X) > 0.3).astype(int))


@pytest.mark.parametrize('make', MODELS)
def test_threshold_follows_the_costs_of_each_fit(make, data):
    X, y = data
    model = make()
    # The scores themselves change with the costs, so the share predicted positive is what compares.
    cheap_contact = model.fit(X, y, tp_cost=-20.0, fp_cost=1.0).predict(X).mean()
    dear_contact = model.fit(X, y, tp_cost=-20.0, fp_cost=19.0).predict(X).mean()
    assert dear_contact < cheap_contact


@pytest.mark.parametrize('make', MODELS)
def test_threshold_uses_each_samples_own_costs(make, data, seeded_rng):
    X, y = data
    clv = seeded_rng.uniform(5, 50, y.size)
    loss = Metric(CostMatrix().add_tp_benefit('clv').add_fp_cost('contact'), MaxProfit())
    model = make(loss=loss).fit(X, y, clv=clv, contact=4.0)

    expected = loss.model.decision_threshold(y, _positive_proba(model, X), clv=clv, contact=4.0)
    assert model.threshold_ == expected


@pytest.mark.parametrize('make', MODELS)
def test_threshold_is_a_searchable_parameter(make, data):
    X, y = data
    model = clone(make()).set_params(threshold=0.7).fit(X, y, **COSTS)
    assert model.threshold_ == 0.7


@pytest.mark.parametrize('threshold', [-0.1, 1.5, 'learned'])
def test_rejects_a_threshold_outside_the_probabilities(threshold, data):
    X, y = data
    with pytest.raises(ValueError, match='threshold'):
        ProfMPMClassifier(threshold=threshold).fit(X, y, **COSTS)


@pytest.mark.parametrize('model', [CSLogitClassifier(), CSBoostClassifier(), CSTreeClassifier()], ids=type)
def test_cost_trained_models_have_no_threshold(model):
    assert 'threshold' not in model.get_params()
