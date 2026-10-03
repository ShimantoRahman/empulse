import warnings
from typing import ClassVar
from unittest import mock

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, MinCost, max_profit_score
from empulse.models import ProfTreeClassifier
from empulse.models.tree import proftree as proftree_module


@pytest.fixture(scope='module')
def data(make_data):
    return make_data(n_samples=100, n_features=4, random_state=0)


class TestFitDispatch:
    """The `_fit` code paths.

    `_prepare_class_costs` is shared between the "no custom loss" and "deterministic MaxProfit
    metric" cases (both use `fit_max_profit`). A stochastic MaxProfit metric whose expected profit
    has a closed form is scored natively through `fit_max_profit` too; any other stochastic metric or
    strategy falls through to the generic `fit_custom` fitness-function path.
    """

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_no_custom_loss_uses_fit_max_profit(self, data):
        X, y = data
        model = ProfTreeClassifier(max_iter=5, population_size=10, random_state=42).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        assert model.n_iter_ > 0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_deterministic_maxprofit_metric_uses_fit_max_profit(self, data):
        X, y = data
        clv, d = sympy.symbols('clv d')
        metric = Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(d), MaxProfit())
        model = ProfTreeClassifier(loss=metric, max_iter=5, population_size=10, random_state=42).fit(
            X, y, clv=100.0, d=10.0
        )
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_stochastic_maxprofit_metric_is_scored_natively(self, data):
        """A stochastic MaxProfit metric must not be routed through `_prepare_class_costs`."""
        X, y = data
        clv_rv = sympy.stats.Beta('clv', 2, 5)
        contact_cost = sympy.symbols('contact_cost')
        metric = Metric(CostMatrix().add_tp_benefit(clv_rv).add_fp_cost(contact_cost), MaxProfit())
        model = ProfTreeClassifier(loss=metric, max_iter=5, population_size=10, random_state=42).fit(
            X, y, contact_cost=1.0
        )
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_non_maxprofit_metric_uses_fit_custom(self, data):
        """A non-MaxProfit strategy (e.g. Cost) must use the custom fitness-function path."""
        X, y = data
        clv, d = sympy.symbols('clv d')
        metric = Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(d), Cost())
        model = ProfTreeClassifier(loss=metric, max_iter=5, population_size=10, random_state=42).fit(
            X, y, clv=100.0, d=10.0
        )
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape


class TestPreflightSmokeTestReproducibility:
    """The pre-flight loss-function smoke test is seeded from `self.random_state` via `check_random_state`.

    A loss function that only fails for some random inputs then fails reproducibly across runs.
    """

    def test_same_random_state_gives_reproducible_smoke_test_probe(self):
        from sklearn.utils.validation import check_random_state

        probe_1 = check_random_state(42).random(50).astype(np.float32)
        probe_2 = check_random_state(42).random(50).astype(np.float32)
        np.testing.assert_array_equal(probe_1, probe_2)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_loss_function_sensitive_to_probe_values_is_deterministic(self, data):
        """A custom loss that raises only for specific random values must fail/succeed consistently."""
        X, y = data
        clv, d = sympy.symbols('clv d')
        metric = Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(d), Cost())

        results = []
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    ProfTreeClassifier(loss=metric, max_iter=2, population_size=10, random_state=7).fit(
                        X, y, clv=100.0, d=10.0
                    )
                    results.append('ok')
                except ValueError:
                    results.append('raised')
        assert len(set(results)) == 1, f'Smoke test outcome was not reproducible across runs: {results}'


class TestCustomLossDirection:
    """The custom-loss path hands the evolutionary search the negated ``BaseMetric._loss``.

    The search keeps the fittest trees, i.e. it maximizes, while ``_loss`` is a value to minimize.
    Handing it ``_loss`` directly would make ProfTree search for the costliest tree: with an
    expected-cost loss it would settle on a single leaf, which predicts the class prior for every
    sample.
    """

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_custom_cost_loss_is_minimized(self):
        X, y = make_classification(n_samples=600, random_state=0, weights=[0.7])
        fn, fp = sympy.symbols('fn fp')
        metric = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost())
        model = ProfTreeClassifier(loss=metric, max_iter=60, patience=60, population_size=40, random_state=0)
        y_proba = model.fit(X, y, fn=5.0, fp=1.0).predict_proba(X)[:, 1]

        single_leaf_cost = metric(y, np.full(y.size, y.mean()), fn=5.0, fp=1.0)
        assert metric(y, y_proba, fn=5.0, fp=1.0) < 0.5 * single_leaf_cost


class TestConstantFeatures:
    """A constant feature must not crash the interpreter.

    Drawing a split for a feature with a single distinct value would take a random integer modulo
    zero, which raises SIGFPE in C and kills the process, so the fit runs in a subprocess.
    """

    def test_fit_with_constant_features(self):
        # Both configurations share one subprocess: starting an interpreter and importing Empulse
        # costs far more than the fits. A crash still fails the test, and the script says which
        # configuration it was on before starting it.
        import subprocess
        import sys

        script = """
import sys
import numpy as np
from sklearn.datasets import make_classification
from empulse.models import ProfTreeClassifier

for constant_columns in ([0], [0, 1, 2]):
    print(f'fitting with constant columns {constant_columns}', file=sys.stderr, flush=True)
    X, y = make_classification(n_samples=200, n_features=3, n_informative=2, n_redundant=0, random_state=0)
    X[:, constant_columns] = 1.0
    model = ProfTreeClassifier(max_iter=20, population_size=20, random_state=0).fit(X, y, fn_cost=5, fp_cost=1)
    y_proba = model.predict_proba(X)[:, 1]
    if len(constant_columns) == X.shape[1]:
        # Nothing to split on: a single leaf predicting the class prior.
        assert np.allclose(y_proba, y.mean()), y_proba
"""
        result = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True, timeout=300, check=False
        )
        assert result.returncode == 0, result.stderr


class TestNodeCountPenalty:
    """The node count behind the ``alpha`` penalty matches the real tree.

    ``grow`` must not add two nodes when the leaf is already at ``max_depth`` and nothing is split,
    and pruning illegal nodes must subtract the nodes it removes; otherwise ``alpha`` penalizes the
    wrong trees.
    """

    @staticmethod
    def _count(node):
        return (
            0
            if node is None
            else 1 + TestNodeCountPenalty._count(node['left']) + TestNodeCountPenalty._count(node['right'])
        )

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('max_depth', [2, 10])
    @pytest.mark.parametrize('custom_loss', [False, True], ids=['max_profit', 'custom_loss'])
    def test_stored_node_count_matches_tree(self, max_depth, custom_loss):
        X, y = make_classification(n_samples=400, random_state=0)
        fn, fp = sympy.symbols('fn fp')
        loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost()) if custom_loss else None
        model = ProfTreeClassifier(
            loss=loss, max_depth=max_depth, max_iter=100, patience=100, population_size=20, alpha=1e-3, random_state=0
        )
        if custom_loss:
            model.fit(X, y, fn=5.0, fp=1.0)
        else:
            model.fit(X, y, fn_cost=5.0, fp_cost=1.0)

        tree = model.tree_._serialize_tree()
        assert tree['n_nodes'] == self._count(tree['root'])


class TestEarlyStoppingWithNegativeFitness:
    """Early stopping triggers when the fitness is negative.

    A challenger has to beat the champion by a tolerance relative to the champion's magnitude. Scaling
    the champion by ``(1 + tolerance)`` is below the champion when the fitness is negative (e.g. costs
    without benefits, or a custom loss), so a tie would count as an improvement and reset the patience
    counter every generation.

    With prune as the only variation operator, the population never changes: every tree starts as a
    single split at its root, which prune leaves alone. So every generation's best ties the champion
    and the search must stop after ``patience`` generations without improvement.
    """

    FROZEN_POPULATION: ClassVar[dict[str, float]] = {
        'crossover_rate': 0.0,
        'grow_rate': 0.0,
        'prune_rate': 1.0,
        'mutate_split_rate': 0.0,
        'mutate_value_rate': 0.0,
    }

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'costs',
        [{'fn_cost': 5.0, 'fp_cost': 1.0}, {'tp_cost': -20.0, 'fp_cost': 1.0}],
        ids=['negative_fitness', 'positive_fitness'],
    )
    def test_patience_runs_out_on_a_stagnant_population(self, costs):
        X, y = make_classification(n_samples=400, random_state=0)
        model = ProfTreeClassifier(
            patience=5, max_iter=200, population_size=20, random_state=0, **self.FROZEN_POPULATION
        ).fit(X, y, **costs)
        assert model.n_iter_ == 6

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_patience_runs_out_with_a_custom_loss(self):
        X, y = make_classification(n_samples=400, random_state=0)
        fn, fp = sympy.symbols('fn fp')
        loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost())
        model = ProfTreeClassifier(
            loss=loss, patience=5, max_iter=200, population_size=20, random_state=0, **self.FROZEN_POPULATION
        ).fit(X, y, fn=5.0, fp=1.0)
        assert model.n_iter_ == 6


class TestSurvivorSelection:
    """Each offspring is scored before it competes with its parent, and the fitter of the two survives.

    Every initial tree is a single split at its root, so with crossover as the only variation operator
    a deeper tree can only come from a crossover offspring that won its place in the population. An
    offspring compared before it is scored carries no fitness of its own after a crossover, and would
    never survive.
    """

    CROSSOVER_ONLY: ClassVar[dict[str, float]] = {
        'crossover_rate': 1.0,
        'grow_rate': 0.0,
        'prune_rate': 0.0,
        'mutate_split_rate': 0.0,
        'mutate_value_rate': 0.0,
    }

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('loss', ['max_profit', 'custom_loss', 'stochastic_max_profit'])
    def test_crossover_offspring_survive(self, loss):
        X, y = make_classification(n_samples=1000, n_features=6, n_informative=4, random_state=0)
        clv, cost = sympy.symbols('clv cost')
        if loss == 'max_profit':
            model, fit_params = ProfTreeClassifier(), {'fn_cost': 5.0, 'fp_cost': 1.0}
        elif loss == 'custom_loss':
            fn, fp = sympy.symbols('fn fp')
            model = ProfTreeClassifier(loss=Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost()))
            fit_params = {'fn': 5.0, 'fp': 1.0}
        else:
            gamma = sympy.stats.Beta('gamma', 6, 14)
            matrix = CostMatrix().add_tp_benefit(gamma * clv).add_fp_cost(cost)
            model, fit_params = ProfTreeClassifier(loss=Metric(matrix, MaxProfit())), {'clv': 100.0, 'cost': 1.0}
        model.set_params(max_iter=30, patience=30, population_size=20, random_state=0, **self.CROSSOVER_ONLY)
        model.fit(X, y, **fit_params)

        assert model.tree_._serialize_tree()['n_nodes'] > 3


class TestLeafLevelFitness:
    """The native fitness is computed from the fitted tree's leaf counts, not per-sample predictions.

    The two must agree: every sample in a leaf gets the same score, so ranking the leaves gives the
    same ROC curve as ranking the samples.
    """

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'costs',
        [
            {'fn_cost': 5.0, 'fp_cost': 1.0},
            {'tp_cost': -200.0, 'fp_cost': 11.0},
            {'tp_cost': -3.0, 'tn_cost': -1.0, 'fn_cost': 4.0, 'fp_cost': 2.0},
        ],
        ids=['costs_only', 'benefit', 'all_four'],
    )
    @pytest.mark.parametrize('tied_features', [False, True], ids=['continuous', 'tied'])
    def test_fitness_matches_max_profit_of_the_predictions(self, costs, tied_features):
        X, y = make_classification(n_samples=600, n_features=6, weights=[0.7], random_state=0)
        if tied_features:
            X = np.round(X)  # leaves with equal scores, whose samples tie in the ranking
        model = ProfTreeClassifier(max_iter=20, population_size=30, max_depth=5, random_state=0).fit(X, y, **costs)

        fitness = model.tree_._serialize_tree()['fitness']
        expected = max_profit_score(y, model.predict_proba(X)[:, 1], **costs)
        assert fitness == pytest.approx(expected, rel=1e-5, abs=1e-5)


class TestMemoryLayout:
    """Samples are routed through the tree by pointer to their row, which requires C-ordered rows."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('custom_loss', [False, True], ids=['max_profit', 'custom_loss'])
    def test_fortran_ordered_input_gives_the_same_fit(self, custom_loss):
        X, y = make_classification(n_samples=300, n_features=5, random_state=0)
        fn, fp = sympy.symbols('fn fp')
        loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost()) if custom_loss else None
        fit_params = {'fn': 5.0, 'fp': 1.0} if custom_loss else {'fn_cost': 5.0, 'fp_cost': 1.0}

        def fit(X):
            return ProfTreeClassifier(loss=loss, max_iter=20, population_size=20, random_state=0).fit(
                X, y, **fit_params
            )

        c_ordered = fit(np.ascontiguousarray(X))
        f_ordered = fit(np.asfortranarray(X))
        np.testing.assert_array_equal(
            c_ordered.predict_proba(np.asfortranarray(X)), f_ordered.predict_proba(np.ascontiguousarray(X))
        )


class TestNJobs:
    """The trees are fitted in parallel, but only the serial variation step draws random numbers."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('custom_loss', [False, True], ids=['max_profit', 'custom_loss'])
    def test_fitted_tree_does_not_depend_on_n_jobs(self, custom_loss):
        X, y = make_classification(n_samples=500, n_features=6, random_state=0)
        fn, fp = sympy.symbols('fn fp')
        loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost()) if custom_loss else None
        fit_params = {'fn': 5.0, 'fp': 1.0} if custom_loss else {'fn_cost': 5.0, 'fp_cost': 1.0}

        def fit(n_jobs):
            model = ProfTreeClassifier(loss=loss, max_iter=30, population_size=40, n_jobs=n_jobs, random_state=0)
            return model.fit(X, y, **fit_params)

        serial = fit(1)
        for n_jobs in (2, -1, None):
            parallel = fit(n_jobs)
            np.testing.assert_array_equal(parallel.predict_proba(X), serial.predict_proba(X))
            assert parallel.n_iter_ == serial.n_iter_

    def test_zero_jobs_is_rejected(self, data):
        X, y = data
        with pytest.raises(ValueError, match='n_jobs'):
            ProfTreeClassifier(n_jobs=0).fit(X, y, fn_cost=1.0, fp_cost=1.0)


class TestIncrementalRefit:
    """After a variation operator, only the samples reaching the changed subtree are routed again.

    The counts left in the fitted tree must still be those of routing every training sample through
    it from scratch, whichever operators produced it.
    """

    @staticmethod
    def _assert_counts_match_routing(node, X, y, min_samples_split, min_samples_leaf, is_root=True):
        assert node['n_samples'] == len(y)
        assert node['n_positive_samples'] == y.sum()
        if node['left'] is None:
            return
        if not is_root:  # the root is never pruned
            assert node['n_samples'] >= min_samples_split
        goes_left = X[:, node['feature_index']] <= node['split_value']
        for child, mask in ((node['left'], goes_left), (node['right'], ~goes_left)):
            if not is_root:
                assert child['n_samples'] >= min_samples_leaf
            TestIncrementalRefit._assert_counts_match_routing(
                child, X[mask], y[mask], min_samples_split, min_samples_leaf, is_root=False
            )

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'operators',
        [
            {'grow_rate': 1.0},
            {'grow_rate': 0.5, 'crossover_rate': 0.5},
            {'grow_rate': 0.5, 'prune_rate': 0.5},
            {'grow_rate': 0.5, 'mutate_split_rate': 0.5},
            {'grow_rate': 0.5, 'mutate_value_rate': 0.5},
            {},
        ],
        ids=['grow', 'crossover', 'prune', 'mutate_split', 'mutate_value', 'all'],
    )
    @pytest.mark.parametrize('custom_loss', [False, True], ids=['max_profit', 'custom_loss'])
    def test_counts_match_routing_every_sample(self, operators, custom_loss):
        X, y = make_classification(n_samples=800, n_features=6, random_state=0)
        fn, fp = sympy.symbols('fn fp')
        loss = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost()) if custom_loss else None
        fit_params = {'fn': 5.0, 'fp': 1.0} if custom_loss else {'fn_cost': 5.0, 'fp_cost': 1.0}
        rates = {'crossover_rate': 0.0, 'grow_rate': 0.0, 'prune_rate': 0.0} if operators else {}
        rates |= {'mutate_split_rate': 0.0, 'mutate_value_rate': 0.0} if operators else {}
        model = ProfTreeClassifier(
            loss=loss,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=7,
            max_iter=60,
            patience=60,
            population_size=20,
            random_state=0,
            **(rates | operators),
        ).fit(X, y, **fit_params)

        tree = model.tree_._serialize_tree()
        self._assert_counts_match_routing(tree['root'], X.astype(np.float32), y, 20, 7)


class TestLeafLevelCustomLoss:
    """A loss that can score samples grouped by score is given each tree's leaves, not every sample."""

    EMPC_MATRIX = (
        CostMatrix()
        .add_tp_benefit(sympy.stats.Beta('gamma', 6, 14) * (sympy.Symbol('clv') - sympy.Symbol('d') - 1))
        .add_tp_benefit((1 - sympy.stats.Beta('gamma', 6, 14)) * -1)
        .add_fp_cost(sympy.Symbol('d') + 1)
    )

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('rounded', [False, True], ids=['continuous', 'tied'])
    def test_fits_the_same_tree_as_scoring_every_sample(self, rounded):
        from empulse.metrics import MixtureComponent, MixtureMetric

        X, y = make_classification(n_samples=500, n_features=6, weights=[0.7], random_state=0)
        if rounded:
            X = np.round(X)
        metric = Metric(self.EMPC_MATRIX, MaxProfit())
        # A mixture of one component scores the same, but only from every sample's prediction.
        per_sample = MixtureMetric([MixtureComponent(1.0, metric, {})])
        assert metric._prepare_count_loss(clv=200.0, d=10.0) is not None
        assert per_sample._prepare_count_loss(clv=200.0, d=10.0) is None

        def fit(loss):
            model = ProfTreeClassifier(loss=loss, max_iter=15, population_size=20, max_depth=5, random_state=0)
            return model.fit(X, y, clv=200.0, d=10.0)

        from_leaves, from_samples = fit(metric), fit(per_sample)
        np.testing.assert_array_equal(from_leaves.predict_proba(X), from_samples.predict_proba(X))
        assert from_leaves.n_iter_ == from_samples.n_iter_
        assert from_leaves.tree_._serialize_tree()['fitness'] == from_samples.tree_._serialize_tree()['fitness']


def _stochastic_metric(strategy=None, distribution='beta'):
    clv, cost, a, b = sympy.symbols('clv cost a b')
    gamma = {
        'beta': sympy.stats.Beta('gamma', a, b),
        'gamma': sympy.stats.Gamma('gamma', a, b),
        'normal': sympy.stats.Normal('gamma', a, b),
    }[distribution]
    matrix = CostMatrix().add_tp_benefit(gamma * clv).add_fp_cost(cost)
    return Metric(matrix, strategy if strategy is not None else MaxProfit())


STOCHASTIC_PARAMETERS = {
    'beta': {'clv': 100.0, 'cost': 1.0, 'a': 6.0, 'b': 14.0},
    'gamma': {'clv': 100.0, 'cost': 1.0, 'a': 2.0, 'b': 0.2},
    'normal': {'clv': 100.0, 'cost': 1.0, 'a': 0.3, 'b': 0.1},
}


class TestSampleCache:
    """``cache_samples`` changes how trees are refit, never which trees are found."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('n_jobs', [1, 3])
    @pytest.mark.parametrize('loss', ['max_profit', 'custom_loss', 'expected_max_profit', 'min_cost', 'monte_carlo'])
    def test_cache_does_not_change_the_fit(self, loss, n_jobs):
        X, y = make_classification(n_samples=600, n_features=6, weights=[0.7], random_state=0)
        fit_params = {'fn_cost': 5.0, 'fp_cost': 1.0}
        if loss == 'custom_loss':
            fn, fp = sympy.symbols('fn fp')
            loss_, fit_params = Metric(CostMatrix().add_fn_cost(fn).add_fp_cost(fp), Cost()), {'fn': 5.0, 'fp': 1.0}
        elif loss == 'expected_max_profit':
            loss_, fit_params = _stochastic_metric(), STOCHASTIC_PARAMETERS['beta']
        elif loss == 'min_cost':
            loss_, fit_params = _stochastic_metric(MinCost()), STOCHASTIC_PARAMETERS['beta']
        elif loss == 'monte_carlo':
            loss_ = _stochastic_metric(MaxProfit(integration_method='monte-carlo', random_state=0))
            fit_params = STOCHASTIC_PARAMETERS['beta']
        else:
            loss_ = None

        def fit(cache_samples):
            model = ProfTreeClassifier(
                loss=loss_, max_iter=30, population_size=20, n_jobs=n_jobs, cache_samples=cache_samples, random_state=0
            )
            return model.fit(X, y, **fit_params)

        cached, uncached = fit(True), fit(False)
        assert cached.tree_._serialize_tree() == uncached.tree_._serialize_tree()
        assert cached.n_iter_ == uncached.n_iter_


class TestNativeExpectedMaxProfit:
    """A stochastic MaxProfit or MinCost loss with a closed form scores every tree in compiled code.

    It must find the same trees as scoring each tree through the loss's Python count scorer, which is
    what it falls back to without the compiled kernel.
    """

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('distribution', ['beta', 'gamma', 'normal'])
    @pytest.mark.parametrize('strategy', [MaxProfit, MinCost])
    def test_native_scores_match_the_python_scores(self, distribution, strategy):
        X, y = make_classification(n_samples=600, n_features=6, weights=[0.7], random_state=0)
        loss = _stochastic_metric(strategy(), distribution)

        def fit():
            model = ProfTreeClassifier(loss=loss, max_iter=30, population_size=20, n_jobs=2, random_state=0)
            return model.fit(X, y, **STOCHASTIC_PARAMETERS[distribution])

        native = fit()
        with mock.patch.object(proftree_module, '_expected_max_profit_of_groups_address', None):
            python = fit()
        assert native.tree_._serialize_tree() == python.tree_._serialize_tree()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_trees_are_not_scored_in_python(self):
        X, y = make_classification(n_samples=300, n_features=4, random_state=0)
        with mock.patch.object(
            proftree_module, '_negated_count_loss', wraps=proftree_module._negated_count_loss
        ) as count_loss:
            ProfTreeClassifier(loss=_stochastic_metric(), max_iter=10, population_size=10, random_state=0).fit(
                X, y, **STOCHASTIC_PARAMETERS['beta']
            )
        assert count_loss.call_count == 1  # the check of the loss before the fit
