"""Tests for the sign-flipped sibling strategies.

Each of these pairs computes exactly the same quantity and differs only in how it is reported:

* :class:`~empulse.metrics.Profit` is :class:`~empulse.metrics.Cost` negated,
* :class:`~empulse.metrics.MinCost` is :class:`~empulse.metrics.MaxProfit` negated,
* :class:`~empulse.metrics.EmpiricalMinCost` is
  :class:`~empulse.metrics.EmpiricalMaxProfit` negated.

Choosing between a pair is presentational. The load-bearing property is that
:meth:`~empulse.metrics.BaseMetric._loss` -- what every model actually optimizes -- is *identical*
for both members, because the sibling flips its ``direction`` along with its score. These tests pin
that, and the model-level tests at the bottom confirm it end to end by fitting the same model twice.
"""

import numpy as np
import pytest

from empulse.metrics import (
    Cost,
    CostMatrix,
    EmpiricalMaxProfit,
    EmpiricalMinCost,
    MaxProfit,
    Metric,
    MinCost,
    MixtureComponent,
    MixtureMetric,
    Profit,
)

PAIRS = [
    pytest.param(Cost, Profit, id='Cost-Profit'),
    pytest.param(MaxProfit, MinCost, id='MaxProfit-MinCost'),
    pytest.param(EmpiricalMaxProfit, EmpiricalMinCost, id='EmpiricalMaxProfit-EmpiricalMinCost'),
]

COSTS = {'fp_cost': 1.0, 'fn_cost': 5.0}


@pytest.fixture(scope='module')
def cost_matrix():
    return CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.binomial(1, 0.3, n)
    y_score = np.clip(rng.normal(0.5, 0.2, n) + y * 0.25, 0.01, 0.99)
    return y, y_score


@pytest.mark.parametrize(('parent_cls', 'sibling_cls'), PAIRS)
class TestSiblingIsANegation:
    def test_score_is_negated(self, parent_cls, sibling_cls, cost_matrix, dataset):
        y, y_score = dataset
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling(y, y_score, **COSTS) == pytest.approx(-parent(y, y_score, **COSTS))

    def test_loss_is_identical(self, parent_cls, sibling_cls, cost_matrix, dataset):
        """The property models rely on: both members present the same objective to an estimator."""
        y, y_score = dataset
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling._loss(y, y_score, **COSTS) == pytest.approx(parent._loss(y, y_score, **COSTS))

    def test_directions_are_opposite(self, parent_cls, sibling_cls, cost_matrix):
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling.direction is not parent.direction

    def test_names_differ(self, parent_cls, sibling_cls, cost_matrix):
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling.__name__ != parent.__name__

    def test_optimal_rate_and_threshold_are_unchanged(self, parent_cls, sibling_cls, cost_matrix, dataset):
        """An optimal cutoff is orientation-free -- negating the metric must not move it."""
        y, y_score = dataset
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling.optimal_rate(y, y_score, **COSTS) == pytest.approx(parent.optimal_rate(y, y_score, **COSTS))
        np.testing.assert_allclose(
            sibling.optimal_threshold(y, y_score, **COSTS),
            parent.optimal_threshold(y, y_score, **COSTS),
        )

    def test_boost_objective_dispatch_is_unchanged(self, parent_cls, sibling_cls):
        """CSBoostClassifier branches on this; a sibling must route the same way as its parent."""
        assert sibling_cls().requires_dynamic_boost_objective == parent_cls().requires_dynamic_boost_objective

    def test_instance_dependent_costs(self, parent_cls, sibling_cls, cost_matrix, dataset):
        y, y_score = dataset
        rng = np.random.default_rng(1)
        costs = {'fp_cost': rng.uniform(0.5, 2.0, y.size), 'fn_cost': rng.uniform(3.0, 8.0, y.size)}
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling(y, y_score, **costs) == pytest.approx(-parent(y, y_score, **costs))

    def test_repr_and_latex_smoke(self, parent_cls, sibling_cls, cost_matrix):
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling_cls.__name__ in repr(sibling)
        latex = sibling._repr_latex_()
        assert isinstance(latex, str)
        assert latex.startswith('$')

    def test_mixing_a_pair_raises(self, parent_cls, sibling_cls, cost_matrix):
        """A mixture of a metric and its own negation has no well-defined direction."""
        mixture = MixtureMetric([
            MixtureComponent(weight=0.5, metric=Metric(cost_matrix, parent_cls()), parameters={}),
            MixtureComponent(weight=0.5, metric=Metric(cost_matrix, sibling_cls()), parameters={}),
        ])

        with pytest.raises(ValueError, match='inconsistent optimization directions'):
            _ = mixture.direction


def test_latex_of_a_sibling_differs_from_its_parent(cost_matrix):
    """The rendered formula is negated too, not just the computed value."""
    assert Metric(cost_matrix, Profit())._repr_latex_() != Metric(cost_matrix, Cost())._repr_latex_()


def test_empirical_min_cost_latex_uses_min(cost_matrix):
    assert '\\min_{k' in Metric(cost_matrix, EmpiricalMinCost())._repr_latex_()
    assert '\\max_{k' in Metric(cost_matrix, EmpiricalMaxProfit())._repr_latex_()


def test_min_cost_inherits_max_profit_arguments():
    """MinCost is a MaxProfit, so it takes the same integration arguments."""
    strategy = MinCost(integration_method='quad', n_mc_samples_exp=8, random_state=0, alpha=2.0)

    assert strategy.integration_method == 'quad'
    assert strategy.n_mc_samples == 2**8
    assert strategy.alpha == 2.0
    assert isinstance(strategy, MaxProfit)


class TestModelsTrainIdenticallyOnEitherPhrasing:
    """The point of the sibling strategies: which phrasing you pick cannot change the fitted model."""

    @pytest.fixture(scope='class')
    def data(self):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        return X, y

    def test_cslogit(self, data, cost_matrix):
        from empulse.models import CSLogitClassifier

        X, y = data
        cost_model = CSLogitClassifier(loss=Metric(cost_matrix, Cost())).fit(X, y, **COSTS)
        profit_model = CSLogitClassifier(loss=Metric(cost_matrix, Profit())).fit(X, y, **COSTS)

        np.testing.assert_allclose(cost_model.coef_, profit_model.coef_)

    def test_csboost(self, data, cost_matrix):
        from xgboost import XGBClassifier

        from empulse.models import CSBoostClassifier

        X, y = data
        cost_model = CSBoostClassifier(
            XGBClassifier(n_estimators=10, random_state=0), loss=Metric(cost_matrix, Cost())
        ).fit(X, y, **COSTS)
        profit_model = CSBoostClassifier(
            XGBClassifier(n_estimators=10, random_state=0), loss=Metric(cost_matrix, Profit())
        ).fit(X, y, **COSTS)

        np.testing.assert_allclose(cost_model.predict_proba(X), profit_model.predict_proba(X))

    def test_proftree_takes_the_same_fast_path(self, data, cost_matrix):
        """MinCost is a MaxProfit, so ProfTree uses the Cython fit_max_profit path for both."""
        from empulse.models import ProfTreeClassifier

        X, y = data
        profit_model = ProfTreeClassifier(loss=Metric(cost_matrix, MaxProfit()), max_iter=10, random_state=0).fit(
            X, y, **COSTS
        )
        cost_model = ProfTreeClassifier(loss=Metric(cost_matrix, MinCost()), max_iter=10, random_state=0).fit(
            X, y, **COSTS
        )

        np.testing.assert_allclose(cost_model.predict_proba(X), profit_model.predict_proba(X))
