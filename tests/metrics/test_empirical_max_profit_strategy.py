"""Tests for the EmpiricalMaxProfit metric strategy.

Unlike :class:`~empulse.metrics.MaxProfit`, which finds the profit-maximizing threshold *inside*
the integral over any stochastic variables (via a closed-form profit function of the population's
true/false positive rates), EMPB (and thus ``EmpiricalMaxProfit``) first simplifies any stochastic
variable to its mean, and only then searches for the profit-maximizing threshold, empirically: by
ranking samples by predicted score and taking the argmax of the resulting cumulative profit curve.
This mirrors :func:`~empulse.metrics.empb`/:func:`~empulse.metrics.empb_score`.

Like AUEPC, EmpiricalMaxProfit does not support use as a model training objective.

These tests check:

* EmpiricalMaxProfit (built from the churn cost matrix shape) reproduces
  :func:`~empulse.metrics.empb_score` and the threshold fraction returned by
  :func:`~empulse.metrics.empb`.
* ``optimal_rate`` matches the fraction returned by :func:`~empulse.metrics.empb`.
* ``optimal_threshold`` is consistent with ``optimal_rate`` via the empirical score distribution.
* Instance-dependent costs (array-like ``clv``) work, matching the native function.
* Class-dependent (scalar) costs also work.
* logit_objective / gradient_boost_objective / prepare_boost_objective are (deliberately)
  unsupported, since the profit-maximizing threshold is a piecewise-constant argmax, not
  differentiable in the predicted scores.
* direction is MAXIMIZE, and repr/latex smoke test passes.
"""

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import CostMatrix, EmpiricalMaxProfit, Metric
from empulse.metrics.churn.stochastic import empb_score
from empulse.metrics.metric.strategies.empirical_max_profit_strategy import EmpiricalMaxProfitScore

from .reference.churn import empb


@pytest.fixture(scope='module')
def churn_cost_matrix():
    gamma = sympy.stats.Beta('gamma', 6, 14)
    delta, f, clv = sympy.symbols('delta f clv')
    # The contact cost is incurred whenever a churner is contacted, regardless of whether they
    # accept the incentive offer, so it must be added as a separate, gamma-independent term
    # (matching the canonical churn cost matrix documented in CostMatrix and used by B2BoostClassifier).
    return (
        CostMatrix()
        .add_tp_benefit(gamma * ((1 - delta) * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f),
        {'delta': delta, 'f': f, 'clv': clv},
    )


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    clv = rng.gamma(2, 100, size=n)
    y_score = rng.normal(size=n) + y * 1.5
    return y, y_score, clv


def test_empirical_max_profit_matches_native_function(churn_cost_matrix, dataset):
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    result = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected = empb_score(
        y, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=incentive_fraction, contact_cost=contact_cost
    )

    assert result == pytest.approx(expected, rel=1e-6)


def test_empirical_max_profit_optimal_rate_matches_native_function(churn_cost_matrix, dataset):
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
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


def test_empirical_max_profit_optimal_threshold_is_consistent_with_optimal_rate(churn_cost_matrix, dataset):
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    rate = metric.optimal_rate(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    threshold = metric.optimal_threshold(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    # The threshold should mark (approximately) the same fraction of samples as "positive" as
    # the optimal rate, since it is derived from the empirical score distribution.
    predicted_positive_frac = np.mean(y_score >= threshold)
    assert predicted_positive_frac == pytest.approx(rate, abs=1 / len(y))


def test_empirical_max_profit_class_dependent_clv(churn_cost_matrix, dataset):
    """A scalar (class-dependent) clv should work and match the native function."""
    cost_matrix, _ = churn_cost_matrix
    y, y_score, _clv = dataset
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


def test_empirical_max_profit_instance_dependent_costs(dataset):
    """EmpiricalMaxProfit should support instance-dependent (array-like) costs beyond clv alone."""
    y, y_score, clv = dataset
    fp = sympy.symbols('fp')
    cost_matrix = CostMatrix().add_tp_benefit(50.0).add_fp_cost(fp)
    metric = Metric(cost_matrix, EmpiricalMaxProfit())

    fp_cost = np.abs(clv) / 10  # instance-dependent
    result = metric(y, y_score, fp=fp_cost)

    assert np.isfinite(result)


def test_empirical_max_profit_score_class_matches_hand_rolled_delta(dataset):
    """EmpiricalMaxProfitScore should match a hand-rolled cumulative-profit-argmax implementation."""
    y, y_score, _clv = dataset
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


def test_empirical_max_profit_direction_is_maximize(churn_cost_matrix):
    cost_matrix, _ = churn_cost_matrix
    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    from empulse.metrics.metric.common import Direction

    assert metric.direction == Direction.MAXIMIZE


def test_empirical_max_profit_does_not_support_model_training(churn_cost_matrix, dataset):
    """The profit-maximizing threshold is a piecewise-constant argmax, so training hooks are unsupported."""
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
    metric = Metric(cost_matrix, EmpiricalMaxProfit())

    with pytest.raises(NotImplementedError):
        metric._logit_objective(
            features=np.eye(len(y)),
            y_true=y,
            C=1.0,
            l1_ratio=0.0,
            soft_threshold=False,
            fit_intercept=True,
            clv=clv,
            delta=0.05,
            f=15,
        )
    with pytest.raises(NotImplementedError):
        metric._gradient_boost_objective(y, y_score, clv=clv, delta=0.05, f=15)
    with pytest.raises(NotImplementedError):
        metric._prepare_boost_objective(y, clv=clv, delta=0.05, f=15)


def test_empirical_max_profit_repr_and_latex_smoke(churn_cost_matrix):
    cost_matrix, _ = churn_cost_matrix
    metric = Metric(cost_matrix, EmpiricalMaxProfit())
    assert 'EmpiricalMaxProfit' in repr(metric)
    latex = metric._repr_latex_()
    assert isinstance(latex, str)
    assert latex.startswith('$')
