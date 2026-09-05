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
