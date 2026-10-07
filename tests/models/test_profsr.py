import itertools
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from empulse.metrics import Cost, CostMatrix, Metric
from empulse.models import ProfSRClassifier

FIT_COSTS = {'tp_cost': -200, 'fp_cost': 10}


def fast_model(**kwargs):
    kwargs.setdefault('max_iter', 4)
    kwargs.setdefault('population_size', 30)
    kwargs.setdefault('random_state', 0)
    return ProfSRClassifier(**kwargs)


@pytest.fixture(scope='module')
def data(make_data):
    return make_data(n_samples=300, n_features=4)


def test_evolves_every_generation_when_the_model_makes_a_profit(data):
    X, y = data
    # A large benefit of a true positive makes every model profitable, so the loss is negative
    # from the first generation on.
    model = fast_model().fit(X, y, **FIT_COSTS)

    assert model.run_details_['best_fitness'][0] < 0
    assert len(model.run_details_['generation']) == 4
    assert model.n_iter_ == 4


def test_fitted_model_does_not_depend_on_n_jobs(data):
    X, y = data
    sequential = fast_model(max_iter=3, n_jobs=1).fit(X, y, **FIT_COSTS)
    parallel = fast_model(max_iter=3, n_jobs=2).fit(X, y, **FIT_COSTS)

    assert str(parallel.program_) == str(sequential.program_)
    np.testing.assert_array_equal(parallel.predict_proba(X), sequential.predict_proba(X))


def test_fit_is_reproducible_and_pickles(data):
    X, y = data
    first = fast_model().fit(X, y, **FIT_COSTS)
    second = fast_model().fit(X, y, **FIT_COSTS)
    restored = pickle.loads(pickle.dumps(first))

    assert str(first.program_) == str(second.program_)
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
    np.testing.assert_array_equal(restored.predict_proba(X), first.predict_proba(X))


def test_predict_proba_is_a_probability_per_class(data):
    X, y = data
    proba = fast_model().fit(X, y, **FIT_COSTS).predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


class TestScores:
    """
    A loss that only ranks the samples leaves the scale of the expression free.

    The expression ``X0`` ranks this data perfectly, but its outputs are in the hundreds, where the
    logistic function rounds every one of them to 1.
    """

    @pytest.fixture(scope='class')
    def large_scores(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(100, 1000, (500, 1))
        y = (X[:, 0] > 550).astype(int)
        return X[:300], X[300:], y[:300], y[300:]

    def test_probabilities_keep_the_ranking_of_a_ranking_loss(self, large_scores):
        X_train, X_test, y_train, y_test = large_scores
        model = ProfSRClassifier(max_iter=10, population_size=100, random_state=0).fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        assert roc_auc_score(y_test, model.program_.execute(X_test)) == 1.0
        assert roc_auc_score(y_test, model.decision_function(X_test)) == 1.0
        assert roc_auc_score(y_test, y_proba) == 1.0
        assert np.unique(y_proba).size == np.unique(model.program_.execute(X_test)).size

    def test_probabilities_of_a_probability_loss_are_the_squashed_scores(self, data):
        X, y = data
        loss = Metric(CostMatrix().add_fp_cost('fp').add_fn_cost('fn'), Cost())
        model = fast_model(loss=loss).fit(X, y, fp=1.0, fn=4.0)

        np.testing.assert_array_equal(model.decision_function(X), model.program_.execute(X))
        np.testing.assert_array_equal(model.predict_proba(X)[:, 1], expit(model.decision_function(X)))
        assert model.formula_ == f'sig({model.program_})'

    def test_formula_states_the_whole_rule(self, large_scores):
        X_train, _, y_train, _ = large_scores
        model = ProfSRClassifier(max_iter=10, population_size=100, random_state=0).fit(X_train, y_train)

        assert model.formula_ == (f'sig(({model.program_} - {model.score_center_:.6g}) / {model.score_scale_:.6g})')


class TestSelectProgram:
    """Any expression of the Pareto front can become the fitted one, with the scaling of its own outputs."""

    @pytest.fixture
    def model(self, data):
        X, y = data
        return fast_model(max_iter=6, population_size=60).fit(X, y, **FIT_COSTS)

    def test_every_point_of_the_front_can_be_selected(self, model, data):
        X, _ = data
        for point in model.pareto_front_:
            model.select_program(point.length)
            expected = (point.program.execute(X) - np.median(point.program.execute(X))) / model.score_scale_

            assert model.program_ is point.program
            np.testing.assert_allclose(model.decision_function(X), expected)
            np.testing.assert_array_equal(
                model.predict(X), model.classes_[(model.decision_function(X) > 0).astype(int)]
            )

    def test_the_parsimony_choice_can_be_restored(self, model):
        chosen, scaling = model.program_, (model.score_center_, model.score_scale_)
        model.select_program(model.pareto_front_[0].length)
        model.select_program(chosen.length_)

        assert model.program_ is chosen
        assert (model.score_center_, model.score_scale_) == scaling

    def test_rejects_a_length_not_on_the_front(self, model):
        with pytest.raises(ValueError, match='no expression of length 999'):
            model.select_program(999)


class TestStopping:
    def test_patience_stops_a_search_that_stops_improving(self, data):
        X, y = data
        # Any generation counts as stagnant, so the search stops one generation after its first.
        model = fast_model(max_iter=10, patience=1, tolerance=1e9).fit(X, y, **FIT_COSTS)

        assert model.n_iter_ == 2
        assert len(model.run_details_['generation']) == 2

    def test_time_limit_stops_after_the_current_generation(self, data):
        X, y = data
        model = fast_model(max_iter=10, max_time=1e-9).fit(X, y, **FIT_COSTS)

        assert model.n_iter_ == 1

    def test_runs_all_generations_by_default(self, data):
        X, y = data
        assert fast_model(max_iter=6).fit(X, y, **FIT_COSTS).n_iter_ == 6


class TestExpressionSize:
    def test_no_expression_exceeds_max_length(self, data):
        X, y = data
        model = fast_model(max_iter=6, population_size=60, max_length=7, init_depth=(2, 6)).fit(X, y, **FIT_COSTS)

        # The average over a population is bounded by the maximum, so it catches every generation.
        assert max(model.run_details_['average_length']) <= 7
        assert max(model.run_details_['best_length']) <= 7
        assert model.program_.length_ <= 7
        assert all(point.length <= 7 for point in model.pareto_front_)

    def test_max_length_none_leaves_expressions_unbounded(self, data):
        X, y = data
        model = fast_model(max_length=None, init_depth=(5, 6), init_method='full').fit(X, y, **FIT_COSTS)

        assert model.run_details_['average_length'][0] > 20

    def test_parsimony_coefficient_shrinks_expressions(self, data):
        X, y = data
        costs = {'max_iter': 8, 'population_size': 100, 'max_length': None, 'n_tuned_programs': 0}
        free = fast_model(**costs, parsimony_coefficient=0.0).fit(X, y, **FIT_COSTS)
        penalised = fast_model(**costs, parsimony_coefficient=5.0).fit(X, y, **FIT_COSTS)

        assert penalised.run_details_['average_length'][-1] < free.run_details_['average_length'][-1]


class TestParetoFront:
    def test_front_trades_length_for_loss(self, data):
        X, y = data
        # Long expressions are cheap to find here and have to beat every shorter one to be listed.
        model = fast_model(max_iter=8, population_size=100, parsimony_coefficient=0.0).fit(X, y, **FIT_COSTS)
        front = model.pareto_front_

        assert front
        lengths = [point.length for point in front]
        losses = [point.loss for point in front]
        assert lengths == sorted(set(lengths))
        assert all(later < earlier for earlier, later in itertools.pairwise(losses))
        assert all(point.length == point.program.length_ for point in front)

    def test_front_holds_the_best_loss_of_the_run(self, data):
        X, y = data
        model = fast_model(n_tuned_programs=0).fit(X, y, **FIT_COSTS)

        assert model.pareto_front_[-1].loss == min(model.run_details_['best_fitness'])

    def test_fitted_program_is_on_the_front(self, data):
        X, y = data
        model = fast_model().fit(X, y, **FIT_COSTS)

        assert any(point.program is model.program_ for point in model.pareto_front_)

    def test_parsimony_picks_the_fitted_program_from_the_front(self, data):
        X, y = data
        costs = {'max_iter': 8, 'population_size': 100, 'random_state': 1}
        greedy = fast_model(**costs, parsimony_coefficient=0.0).fit(X, y, **FIT_COSTS)
        frugal = fast_model(**costs, parsimony_coefficient=1e6).fit(X, y, **FIT_COSTS)

        # With no penalty the lowest loss wins; with an overwhelming one the shortest does.
        assert greedy.program_ is greedy.pareto_front_[-1].program
        assert frugal.program_.length_ == min(point.length for point in frugal.pareto_front_)


class TestBatches:
    def test_every_program_is_scored_each_generation(self, data):
        X, y = data
        model = fast_model(max_samples=0.5).fit(X, y, **FIT_COSTS)

        # Programs are scored on different rows, so none can reuse another's score.
        assert model.run_details_['n_evaluated'] == [30] * 4

    def test_out_of_batch_loss_of_the_best_expression_is_logged(self, data):
        X, y = data
        model = fast_model(max_samples=0.5).fit(X, y, **FIT_COSTS)

        assert np.isfinite(model.run_details_['best_oob_fitness']).all()
        assert np.isnan(fast_model().fit(X, y, **FIT_COSTS).run_details_['best_oob_fitness']).all()

    def test_instance_dependent_costs_follow_the_batch(self, data):
        X, y = data
        clv = np.random.default_rng(0).uniform(100, 500, y.size)
        loss = Metric(CostMatrix().add_tp_benefit('c').add_fp_cost('d'), Cost())
        model = fast_model(loss=loss, max_samples=0.5).fit(X, y, c=clv, d=10.0)

        assert np.isfinite(model.run_details_['best_fitness']).all()


class TestSearchSpace:
    def test_function_set_limits_the_operators(self, data):
        X, y = data
        model = fast_model(function_set=('add',), const_range=None).fit(X, y, **FIT_COSTS)
        text = str(model.program_)

        assert 'add(' in text or text.startswith('X')
        assert not any(name + '(' in text for name in ('sub', 'mul', 'div', 'exp', 'log', 'sig'))

    def test_without_constants_expressions_only_use_features(self, data):
        X, y = data
        model = fast_model(const_range=None).fit(X, y, **FIT_COSTS)

        assert model.program_.constants() == []
        assert all(not point.program.constants() for point in model.pareto_front_)

    def test_feature_names_appear_in_the_expression(self, data):
        X, y = data
        frame = pd.DataFrame(X, columns=['recency', 'tenure', 'spend', 'calls'])
        model = fast_model(const_range=None, function_set=('add', 'mul')).fit(frame, y, **FIT_COSTS)
        text = str(model.program_)

        assert any(name in text for name in frame.columns)
        assert 'X0' not in text and 'X1' not in text

    def test_constant_rate_zero_keeps_constants_out_of_the_search(self, data):
        X, y = data
        model = fast_model(constant_rate=0.0).fit(X, y, **FIT_COSTS)

        assert all(not point.program.constants() for point in model.pareto_front_)

    def test_features_are_numbered_without_names(self, data):
        X, y = data
        model = fast_model(const_range=None).fit(X, y, **FIT_COSTS)

        assert 'X' in str(model.program_)


class TestInvalidParameters:
    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'crossover_rate': 0.9, 'mutate_point_rate': 0.2}, 'sum of'),
            ({'function_set': ('add', 'cube')}, 'Unknown function'),
            ({'function_set': ()}, 'at least one function'),
            ({'init_depth': (4, 2)}, 'init_depth'),
            ({'init_depth': (0, 3)}, 'init_depth'),
            ({'const_range': (1.0, -1.0)}, 'const_range'),
            ({'max_length': 2, 'function_set': ('add',)}, 'max_length'),
        ],
    )
    def test_rejects(self, data, kwargs, message):
        X, y = data
        with pytest.raises(ValueError, match=message):
            fast_model(**kwargs).fit(X, y, **FIT_COSTS)
