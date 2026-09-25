"""
The two ranking-based strategies, AUEPC and EmpiricalMaxProfit.

Both rank the samples by predicted score and read the profit off the ranking itself, so they use
per-sample costs as they are and neither can serve as a training objective.

* **EmpiricalMaxProfit** first reduces any stochastic variable to its mean, then takes the argmax
  of the cumulative profit curve over the ranking, mirroring :func:`~empulse.metrics.empb_score`.
  Unlike :class:`~empulse.metrics.MaxProfit`, it maximises outside the integral.
* **AUEPC** integrates the ratio of the ranking's cumulative profit curve to that of an oracle that
  targets the most profitable samples first, mirroring :func:`~empulse.metrics.auepc_score`. Since it
  summarises every targeted fraction, it has no decision rule either.
"""

import itertools

import numpy as np
import pytest
import sympy
from scipy.integrate import trapezoid

from empulse.metrics import AUEPC, CostMatrix, EmpiricalMaxProfit, Metric
from empulse.metrics.churn.stochastic import empb_score
from empulse.metrics.metric._direction import Direction
from empulse.metrics.metric.strategies.auepc_strategy import AUEPCScore
from empulse.metrics.metric.strategies.empirical_max_profit_strategy import EmpiricalMaxProfitScore

from .reference.churn import auepc_score, empb

# --- The shared contract -----------------------------------------------------------------------------


@pytest.fixture(params=[AUEPC, EmpiricalMaxProfit], ids=lambda strategy: strategy.__name__)
def empirical_strategy(request):
    return request.param


def test_direction_is_maximize(empirical_churn_cost_matrix, empirical_strategy):
    cost_matrix, _ = empirical_churn_cost_matrix
    assert Metric(cost_matrix, empirical_strategy()).direction == Direction.MAXIMIZE


def test_instance_dependent_costs(empirical_churn_dataset, empirical_strategy):
    """Both strategies support instance-dependent (array-like) costs."""
    y, y_score, clv = empirical_churn_dataset
    fp = sympy.symbols('fp')
    metric = Metric(CostMatrix().add_tp_benefit(50.0).add_fp_cost(fp), empirical_strategy())
    assert np.isfinite(metric(y, y_score, fp=np.abs(clv) / 10))


def test_does_not_support_model_training(empirical_churn_cost_matrix, empirical_churn_dataset, empirical_strategy):
    """Both evaluate a ranking through an argmax or a ratio of curves, so no training objective exists."""
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    metric = Metric(cost_matrix, empirical_strategy())

    with pytest.raises(NotImplementedError):
        metric._logit_objective(
            features=np.eye(len(y)),
            y_true=y,
            C=1.0,
            l1_ratio=0.0,
            fit_intercept=True,
            clv=clv,
            delta=0.05,
            f=15,
        )
    with pytest.raises(NotImplementedError):
        metric._gradient_boost_objective(y, y_score, clv=clv, delta=0.05, f=15)
    with pytest.raises(NotImplementedError):
        metric._prepare_boost_objective(y, clv=clv, delta=0.05, f=15)


def test_repr_and_latex_smoke(empirical_churn_cost_matrix, empirical_strategy):
    cost_matrix, _ = empirical_churn_cost_matrix
    metric = Metric(cost_matrix, empirical_strategy())
    assert empirical_strategy.__name__ in repr(metric)
    latex = metric._repr_latex_()
    assert isinstance(latex, str)
    assert latex.startswith('$')


# --- EmpiricalMaxProfit ------------------------------------------------------------------------------


def test_empirical_max_profit_matches_native_function(empirical_churn_cost_matrix, empirical_churn_dataset):
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    result = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected = empb_score(
        y, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=incentive_fraction, contact_cost=contact_cost
    )

    assert result == pytest.approx(expected, rel=1e-6)


def test_empirical_max_profit_optimal_rate_matches_native_function(
    empirical_churn_cost_matrix, empirical_churn_dataset
):
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    rate = metric.optimal_rate(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected_profit, expected_rate = empb(
        y, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=incentive_fraction, contact_cost=contact_cost
    )

    assert rate == pytest.approx(expected_rate, rel=1e-6)
    # sanity: the score should also match the profit returned alongside the threshold by empb().
    score = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    assert score == pytest.approx(expected_profit, rel=1e-6)


def test_empirical_max_profit_optimal_threshold_is_consistent_with_optimal_rate(
    empirical_churn_cost_matrix, empirical_churn_dataset
):
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    rate = metric.optimal_rate(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    threshold = metric.optimal_threshold(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    # The threshold should mark (approximately) the same fraction of samples as "positive" as
    # the optimal rate, since it is derived from the empirical score distribution.
    predicted_positive_frac = np.mean(y_score >= threshold)
    assert predicted_positive_frac == pytest.approx(rate, abs=1 / len(y))


def test_empirical_max_profit_class_dependent_clv(empirical_churn_cost_matrix, empirical_churn_dataset):
    """A scalar (class-dependent) clv should work and match the native function."""
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, _clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15
    clv = 150.0

    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    result = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected = empb_score(
        y,
        y_score,
        clv=np.full(len(y), clv),
        alpha=6,
        beta=14,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
    )

    assert result == pytest.approx(expected, rel=1e-6)


def test_empirical_max_profit_score_class_matches_hand_rolled_delta(empirical_churn_dataset):
    """EmpiricalMaxProfitScore should match a hand-rolled cumulative-profit-argmax implementation."""
    y, y_score, _clv = empirical_churn_dataset
    tp, fp = sympy.symbols('tp fp')

    score_fn = EmpiricalMaxProfitScore(tp_benefit=tp, tn_benefit=sympy.Integer(0), fp_cost=fp, fn_cost=sympy.Integer(0))
    tp_val, fp_val = 100.0, 20.0
    result = score_fn(y, y_score, tp=tp_val, fp=fp_val)

    delta = np.where(y == 1, tp_val, -fp_val)
    sorted_indices = np.argsort(y_score)[::-1]
    cumulative_profits = np.cumsum(delta[sorted_indices])
    cumulative_profits = np.insert(cumulative_profits, 0, 0.0)
    expected = float(np.max(cumulative_profits))

    assert result == pytest.approx(expected)


class TestEmpiricalMaxProfitTiedScores:
    """No threshold can split samples with equal scores, so ties must be targeted as one group.

    The curve used to be accumulated one sample at a time, so its maximum could fall inside a tie
    group: the result then depended on the order the tied samples happened to be in (84.0 or 9.0
    for the same data below, depending on the row order).
    """

    Y_TRUE = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    Y_SCORE = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.2, 0.2])
    CLV = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 150.0, 120.0, 300.0])

    def _permuted(self, seed):
        order = np.random.default_rng(seed).permutation(self.Y_TRUE.size)
        return self.Y_TRUE[order], self.Y_SCORE[order], self.CLV[order]

    @pytest.mark.parametrize('method', ['__call__', 'optimal_rate', 'optimal_threshold'])
    def test_result_does_not_depend_on_row_order(self, method):
        results = {getattr(empb_score, method)(y, s, clv=clv) for y, s, clv in map(self._permuted, range(20))}
        assert len(results) == 1

    def test_score_is_the_best_group_boundary(self):
        """Hand-rolled: profits after targeting nobody, the 0.9 group, the 0.5 group, and everyone."""
        tp, fp = sympy.symbols('tp fp')
        score_fn = EmpiricalMaxProfitScore(
            tp_benefit=tp, tn_benefit=sympy.Integer(0), fp_cost=fp, fn_cost=sympy.Integer(0)
        )
        delta = np.where(self.Y_TRUE == 1, 100.0, -20.0)
        boundaries = [0.0, delta[5], delta[:6].sum(), delta.sum()]
        assert score_fn(self.Y_TRUE, self.Y_SCORE, tp=100.0, fp=20.0) == pytest.approx(max(boundaries))

    def test_constant_scores_target_everyone_or_nobody(self):
        """With one score for every sample the only policies are to target nobody or everyone."""
        y, clv = self.Y_TRUE, self.CLV
        constant = np.full(y.size, 0.5)
        # empb_score's defaults: accept rate ~ Beta(6, 14) (mean 0.3), incentive 5% of clv, contact 15.
        accept, incentive, contact = 0.3, 0.05, 15.0
        per_sample = np.where(
            y == 1,
            accept * ((1 - incentive) * clv - contact) - (1 - accept) * contact,
            -(incentive * clv + contact),
        )
        assert empb_score(y, constant, clv=clv) == pytest.approx(max(0.0, per_sample.sum()))
        assert empb_score.optimal_rate(y, constant, clv=clv) == (1.0 if per_sample.sum() > 0 else 0.0)

    def test_optimal_threshold_targets_the_optimal_rate(self):
        """The rate is always at a group boundary, so the threshold reproduces it exactly."""
        rate = empb_score.optimal_rate(self.Y_TRUE, self.Y_SCORE, clv=self.CLV)
        threshold = empb_score.optimal_threshold(self.Y_TRUE, self.Y_SCORE, clv=self.CLV)
        assert np.mean(threshold <= self.Y_SCORE) == pytest.approx(rate)


# --- AUEPC -------------------------------------------------------------------------------------------


def test_auepc_matches_native_function(empirical_churn_cost_matrix, empirical_churn_dataset):
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, AUEPC(normalize=True))
    result = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected = auepc_score(
        y, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=incentive_fraction, contact_cost=contact_cost
    )

    assert result == pytest.approx(expected, rel=1e-6)


def test_auepc_matches_native_function_unnormalized(empirical_churn_cost_matrix, empirical_churn_dataset):
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, AUEPC(normalize=False))
    result = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected = auepc_score(
        y,
        y_score,
        clv=clv,
        alpha=6,
        beta=14,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        normalize=False,
    )

    assert result == pytest.approx(expected, rel=1e-6)


def test_auepc_perfect_ranking_scores_one(empirical_churn_cost_matrix, empirical_churn_dataset):
    """A model whose scores match the oracle's profit-maximizing ranking should score ~1.0."""
    cost_matrix, _syms = empirical_churn_cost_matrix
    y, _, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15
    accept_rate = 6 / (6 + 14)

    tp_benefit = accept_rate * (1 - incentive_fraction) * clv - contact_cost
    fp_cost = incentive_fraction * clv + contact_cost
    oracle_score = np.where(y == 1, tp_benefit, -fp_cost)

    metric = Metric(cost_matrix, AUEPC(normalize=True))
    result = metric(y, oracle_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    assert result == pytest.approx(1.0, abs=1e-8)


def test_auepc_uninformative_ranking_scores_lower(empirical_churn_cost_matrix, empirical_churn_dataset):
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, AUEPC(normalize=True))
    informative = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    rng = np.random.default_rng(1)
    random_score = rng.normal(size=y.shape)
    uninformative = metric(y, random_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    assert uninformative < informative


def test_auepc_score_class_matches_hand_rolled_delta(empirical_churn_dataset):
    """AUEPCScore should match a hand-rolled implementation of the delta-ranking algorithm."""
    y, y_score, _clv = empirical_churn_dataset
    tp, fp = sympy.symbols('tp fp')

    score_fn = AUEPCScore(tp_benefit=tp, tn_benefit=sympy.Integer(0), fp_cost=fp, fn_cost=sympy.Integer(0))
    tp_val, fp_val = 100.0, 20.0
    result = score_fn(y, y_score, tp=tp_val, fp=fp_val)

    delta = np.where(y == 1, tp_val, -fp_val)
    perfect_order = np.argsort(delta)[::-1]
    perfect_profits = np.cumsum(delta[perfect_order])
    model_order = np.argsort(y_score)[::-1]
    profits = np.cumsum(delta[model_order])
    n = y.shape[0]
    stop_index = int(np.argmax(perfect_profits < 0)) if np.any(perfect_profits < 0) else n
    expected = float(trapezoid(profits[:stop_index] / perfect_profits[:stop_index], dx=1 / n))
    expected /= (stop_index - 1) / n

    assert result == pytest.approx(expected)


def test_auepc_has_no_decision_rule(empirical_churn_cost_matrix, empirical_churn_dataset):
    """Unlike EmpiricalMaxProfit, AUEPC summarises every targeted fraction, so there is no one to pick."""
    cost_matrix, _ = empirical_churn_cost_matrix
    y, y_score, clv = empirical_churn_dataset
    metric = Metric(cost_matrix, AUEPC())

    with pytest.raises(NotImplementedError):
        metric.optimal_threshold(y, y_score, clv=clv, delta=0.05, f=15)
    with pytest.raises(NotImplementedError):
        metric.optimal_rate(y, y_score, clv=clv, delta=0.05, f=15)


def _auepc_score_function(normalize=True):
    tp, fp = sympy.symbols('tp fp')
    return AUEPCScore(
        tp_benefit=tp, tn_benefit=sympy.Integer(0), fp_cost=fp, fn_cost=sympy.Integer(0), normalize=normalize
    )


@pytest.mark.parametrize(('normalize', 'expected'), [(True, -10.0), (False, 0.0)])
def test_auepc_single_profitable_point(normalize, expected):
    """The oracle curve turns negative right after its first point, so there is no area to average over.

    Normalizing used to divide by ``(stop_index - 1) / n == 0``. With a single point, the mean ratio
    is that point's own ratio: the model's top sample is a negative (-1000) where the oracle's is a
    positive (+100).
    """
    y = np.array([1, 0, 0, 0])
    y_score = np.array([0.6, 0.9, 0.8, 0.7])
    result = _auepc_score_function(normalize)(y, y_score, tp=100.0, fp=1000.0)
    assert result == pytest.approx(expected)


def test_auepc_stops_where_oracle_profit_reaches_exactly_zero():
    """An oracle curve that returns exactly to zero must stop there, not divide by that zero.

    Deltas are [10, 10, -20, -20], so the oracle curve is [10, 20, 0, -20]. The model's curve is
    [10, -10, ...], giving ratios [1, -0.5] over the profitable range and a mean (trapezoid) of 0.25.
    """
    y = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.7, 0.8, 0.6])
    result = _auepc_score_function()(y, y_score, tp=10.0, fp=20.0)
    assert result == pytest.approx(0.25)


def test_auepc_no_profitable_sample_scores_zero():
    """When even the oracle cannot make a profit, there is no curve to integrate."""
    y = np.array([0, 0, 1])
    y_score = np.array([0.9, 0.8, 0.7])
    result = _auepc_score_function()(y, y_score, tp=0.0, fp=5.0)
    assert result == 0.0
    assert not np.signbit(result)


class TestAUEPCTiedScores:
    """Within a group of tied scores the curve is the expected profit under random tie-breaking.

    The curve used to be accumulated one sample at a time in whatever order the tied samples
    happened to be in, so the same data scored 1.0 or -0.66 depending on its row order.
    """

    Y_TRUE = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    Y_SCORE = np.array([0.9, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3, 0.1])

    def _per_sample_auepc(self, y, y_score, tp, fp):
        """The old algorithm: one sample at a time, in the order given (stable within ties)."""
        delta = np.where(y == 1, tp, -fp)
        perfect_profits = np.cumsum(np.sort(delta)[::-1])
        profits = np.cumsum(delta[np.argsort(-y_score, kind='stable')])
        n = y.size
        stop_index = int(np.argmax(perfect_profits <= 0)) if np.any(perfect_profits <= 0) else n
        score = float(trapezoid(profits[:stop_index] / perfect_profits[:stop_index], dx=1 / n))
        return score / ((stop_index - 1) / n)

    def test_equals_the_average_over_every_tie_breaking_order(self):
        """AUEPC is linear in the curve, so scoring the expected curve is the expected score."""
        tp, fp = 100.0, 20.0
        groups = [np.flatnonzero(score == self.Y_SCORE) for score in np.unique(self.Y_SCORE)]
        scores = []
        for orders in itertools.product(*(itertools.permutations(group) for group in groups)):
            order = np.concatenate(orders)
            scores.append(self._per_sample_auepc(self.Y_TRUE[order], self.Y_SCORE[order], tp, fp))

        result = _auepc_score_function()(self.Y_TRUE, self.Y_SCORE, tp=tp, fp=fp)
        assert result == pytest.approx(np.mean(scores))

    def test_result_does_not_depend_on_row_order(self):
        score_fn = _auepc_score_function()
        results = []
        for seed in range(20):
            order = np.random.default_rng(seed).permutation(self.Y_TRUE.size)
            results.append(score_fn(self.Y_TRUE[order], self.Y_SCORE[order], tp=100.0, fp=20.0))
        assert results == pytest.approx([results[0]] * len(results))
