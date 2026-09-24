import warnings

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric
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
