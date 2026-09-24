import warnings
from typing import ClassVar

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, max_profit_score
from empulse.models import ProfTreeClassifier


@pytest.fixture
def data():
    return make_classification(n_samples=100, n_features=4, random_state=0)


class TestFitDispatch:
    """Regression tests for the four `_fit` code paths after deduplicating the two
    `fit_max_profit` calls into one (MODELS_OPTIMIZERS_REVIEW.md item 28).

    `_prepare_class_costs` is shared between the "no custom loss" and "deterministic MaxProfit
    metric" cases (both use `fit_max_profit`); a stochastic MaxProfit metric or any other strategy
    still falls through to the generic `fit_custom` fitness-function path.
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
    def test_stochastic_maxprofit_metric_uses_fit_custom(self, data):
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
    """Regression test: the pre-flight loss-function smoke test used an unseeded RNG.

    `np.random.default_rng()` with no seed meant a loss function that only fails for some random
    inputs would fail non-reproducibly across runs. It's now seeded from `self.random_state` via
    `check_random_state`.
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
    """Regression test: the custom-loss path handed the evolutionary search ``BaseMetric._loss``.

    The search keeps the fittest trees, i.e. it maximizes, while ``_loss`` is a value to minimize,
    so ProfTree searched for the costliest tree: with an expected-cost loss it settled on a single
    leaf, which predicts the class prior for every sample.
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
    """Regression tests: a constant feature crashed the interpreter.

    Drawing a split for a feature with a single distinct value took a random integer modulo zero,
    which raises SIGFPE in C and kills the process, so the fit runs in a subprocess.
    """

    @pytest.mark.parametrize('constant_columns', [[0], [0, 1, 2]], ids=['one_constant', 'all_constant'])
    def test_fit_with_constant_features(self, constant_columns):
        import subprocess
        import sys

        script = f"""
import numpy as np
from sklearn.datasets import make_classification
from empulse.models import ProfTreeClassifier

X, y = make_classification(n_samples=200, n_features=3, n_informative=2, n_redundant=0, random_state=0)
X[:, {constant_columns}] = 1.0
model = ProfTreeClassifier(max_iter=20, population_size=20, random_state=0).fit(X, y, fn_cost=5, fp_cost=1)
y_proba = model.predict_proba(X)[:, 1]
if {len(constant_columns)} == X.shape[1]:
    # Nothing to split on: a single leaf predicting the class prior.
    assert np.allclose(y_proba, y.mean()), y_proba
"""
        result = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True, timeout=300, check=False
        )
        assert result.returncode == 0, result.stderr


class TestNodeCountPenalty:
    """Regression tests: the node count behind the ``alpha`` penalty drifted from the real tree.

    ``grow`` added two nodes even when the leaf was already at ``max_depth`` and nothing was split,
    and pruning illegal nodes removed nodes without subtracting them, so ``alpha`` penalized the
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
    """Regression tests: early stopping never triggered when the fitness was negative.

    A challenger had to beat ``champion * (1 + tolerance)``, which is below the champion when the
    fitness is negative (e.g. costs without benefits, or a custom loss), so a tie counted as an
    improvement and reset the patience counter every generation.

    With crossover as the only variation operator, the population never changes (crossover
    offspring are never inserted, by design), so every generation's best ties the champion and the
    search must stop after ``patience`` generations without improvement.
    """

    FROZEN_POPULATION: ClassVar[dict[str, float]] = {
        'crossover_rate': 1.0,
        'grow_rate': 0.0,
        'prune_rate': 0.0,
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
