"""Tests for the AUEPC metric strategy.

AUEPC ranks samples by predicted score and integrates the ratio of the classifier's cumulative
"targeting profit" curve against an oracle curve that maximizes cumulative profit at every
targeted fraction. Unlike Cost/LogCost/MaxProfit, it is not meant to support model training --
just computing the metric dynamically from the cost matrix, mirroring
:func:`~empulse.metrics.auepc_score`.

These tests check:

* AUEPC (built from the same cost matrix shape as the churn use case) reproduces
  :func:`~empulse.metrics.auepc_score` for realistic (non-degenerate) parameterizations.
* A perfect ranking scores 1.0 (normalize=True); a less informative ranking scores lower.
* Instance-dependent costs work.
* logit_objective / gradient_boost_objective / prepare_boost_objective / optimal_threshold /
  optimal_rate are (deliberately) unsupported, since AUEPC evaluates a whole ranking rather than
  a per-sample outcome.
"""

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import AUEPC, CostMatrix, Metric
from empulse.metrics.churn.stochastic import auepc_score
from empulse.metrics.metric.strategies.auepc_strategy import AUEPCScore


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


def test_auepc_matches_native_function(churn_cost_matrix, dataset):
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, AUEPC(normalize=True))
    result = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)
    expected = auepc_score(
        y, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=incentive_fraction, contact_cost=contact_cost
    )

    assert result == pytest.approx(expected, rel=1e-6)


def test_auepc_matches_native_function_unnormalized(churn_cost_matrix, dataset):
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
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


def test_auepc_perfect_ranking_scores_one(churn_cost_matrix, dataset):
    """A model whose scores match the oracle's profit-maximizing ranking should score ~1.0."""
    cost_matrix, _syms = churn_cost_matrix
    y, _, clv = dataset
    incentive_fraction, contact_cost = 0.05, 15
    accept_rate = 6 / (6 + 14)

    tp_benefit = accept_rate * (1 - incentive_fraction) * clv - contact_cost
    fp_cost = incentive_fraction * clv + contact_cost
    oracle_score = np.where(y == 1, tp_benefit, -fp_cost)

    metric = Metric(cost_matrix, AUEPC(normalize=True))
    result = metric(y, oracle_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    assert result == pytest.approx(1.0, abs=1e-8)


def test_auepc_uninformative_ranking_scores_lower(churn_cost_matrix, dataset):
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
    incentive_fraction, contact_cost = 0.05, 15

    metric = Metric(cost_matrix, AUEPC(normalize=True))
    informative = metric(y, y_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    rng = np.random.default_rng(1)
    random_score = rng.normal(size=y.shape)
    uninformative = metric(y, random_score, clv=clv, delta=incentive_fraction, f=contact_cost)

    assert uninformative < informative


def test_auepc_direction_is_maximize(churn_cost_matrix):
    cost_matrix, _ = churn_cost_matrix
    metric = Metric(cost_matrix, AUEPC())
    from empulse.metrics.metric.common import Direction

    assert metric.direction == Direction.MAXIMIZE


def test_auepc_score_class_matches_hand_rolled_delta(dataset):
    """AUEPCScore should match a hand-rolled implementation of the delta-ranking algorithm."""
    y, y_score, _clv = dataset
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
    expected = float(np.trapezoid(profits[:stop_index] / perfect_profits[:stop_index], dx=1 / n))
    expected /= (stop_index - 1) / n

    assert result == pytest.approx(expected)


def test_auepc_instance_dependent_costs(dataset):
    """AUEPC should support instance-dependent (array-like) costs."""
    y, y_score, clv = dataset
    fp = sympy.symbols('fp')
    cost_matrix = CostMatrix().add_tp_benefit(50.0).add_fp_cost(fp)
    metric = Metric(cost_matrix, AUEPC())

    fp_cost = np.abs(clv) / 10  # instance-dependent
    result = metric(y, y_score, fp=fp_cost)

    assert np.isfinite(result)


def test_auepc_does_not_support_model_training(churn_cost_matrix, dataset):
    """AUEPC evaluates a whole ranking, not a per-sample outcome, so training hooks are unsupported."""
    cost_matrix, _ = churn_cost_matrix
    y, y_score, clv = dataset
    metric = Metric(cost_matrix, AUEPC())

    with pytest.raises(NotImplementedError):
        metric.optimal_threshold(y, y_score, clv=clv, delta=0.05, f=15)
    with pytest.raises(NotImplementedError):
        metric.optimal_rate(y, y_score, clv=clv, delta=0.05, f=15)
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


def test_auepc_repr_and_latex_smoke(churn_cost_matrix):
    cost_matrix, _ = churn_cost_matrix
    metric = Metric(cost_matrix, AUEPC())
    assert 'AUEPC' in repr(metric)
    latex = metric._repr_latex_()
    assert isinstance(latex, str)
    assert latex.startswith('$')
