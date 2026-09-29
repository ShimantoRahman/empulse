import numpy as np
import pytest

from empulse.models import ProfSRClassifier

pytest.importorskip('gplearn')


def test_evolves_every_generation_when_the_model_makes_a_profit(make_data):
    X, y = make_data(n_samples=300, n_features=4)
    model = ProfSRClassifier(generations=4, population_size=30, random_state=0)
    # A large benefit of a true positive makes every model profitable, so the loss gplearn
    # minimizes is negative from the first generation on.
    model.fit(X, y, tp_cost=-200, fp_cost=10)

    assert model.model_.run_details_['best_fitness'][0] < 0
    assert len(model.model_.run_details_['generation']) == 4
    assert model.n_iter_ == 4


def test_fitted_model_does_not_depend_on_n_jobs(make_data):
    X, y = make_data(n_samples=300, n_features=4)
    kwargs = {'generations': 3, 'population_size': 30, 'random_state': 0}
    sequential = ProfSRClassifier(**kwargs, n_jobs=1).fit(X, y, tp_cost=-200, fp_cost=10)
    parallel = ProfSRClassifier(**kwargs, n_jobs=2).fit(X, y, tp_cost=-200, fp_cost=10)

    np.testing.assert_array_equal(parallel.predict_proba(X), sequential.predict_proba(X))
