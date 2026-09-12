"""Tests for the orientation contract shared by every :class:`BaseMetric`.

``__call__`` returns the metric in its natural orientation (described by ``direction``), while
``_loss`` is always minimized. Models optimize ``_loss`` and never branch on ``direction``, so the
sign relation between the two is pinned here.
"""

import numpy as np
import pytest

from empulse.metrics import (
    AUEPC,
    Cost,
    CostMatrix,
    EmpiricalMaxProfit,
    EmpiricalMinCost,
    LogCost,
    MaxProfit,
    Metric,
    MinCost,
    MixtureComponent,
    MixtureMetric,
    Profit,
    Savings,
    auepc_score,
    empa_score,
    empb_score,
    empc_score,
    empcs_score,
    expected_cost_loss,
    expected_cost_loss_acquisition,
    expected_cost_loss_churn,
    expected_log_cost_loss,
    expected_savings_score,
    max_profit_score,
    mpa_score,
    mpc_score,
    mpcs_score,
)
from empulse.metrics.metric.common import Direction

Y_TRUE = np.array([0, 1, 0, 1, 1, 0, 1, 0])
Y_SCORE = np.array([0.1, 0.9, 0.3, 0.8, 0.4, 0.2, 0.7, 0.6])

CLASS_COSTS = {'fp_cost': 1.0, 'fn_cost': 5.0, 'tp_cost': 0.0, 'tn_cost': 0.0}
CHURN_PARAMS = {'clv': 200.0, 'incentive_cost': 10.0, 'contact_cost': 1.0}
CREDIT_PARAMS = {'roi': 0.2644, 'default_rate': 0.4, 'success_rate': 0.55}


@pytest.mark.parametrize(
    ('strategy', 'expected'),
    [
        (Cost(), Direction.MINIMIZE),
        (LogCost(), Direction.MINIMIZE),
        (Savings(), Direction.MAXIMIZE),
        (MaxProfit(), Direction.MAXIMIZE),
        (EmpiricalMaxProfit(), Direction.MAXIMIZE),
        (AUEPC(), Direction.MAXIMIZE),
        (Profit(), Direction.MAXIMIZE),
        (MinCost(), Direction.MINIMIZE),
        (EmpiricalMinCost(), Direction.MINIMIZE),
    ],
)
def test_strategy_direction(strategy, expected):
    """Each strategy's direction is part of its public contract."""
    assert strategy.direction is expected

    cost_matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')
    assert Metric(cost_matrix, strategy).direction is expected


@pytest.mark.parametrize(
    ('metric', 'expected'),
    [
        (expected_cost_loss, Direction.MINIMIZE),
        (expected_log_cost_loss, Direction.MINIMIZE),
        (expected_cost_loss_churn, Direction.MINIMIZE),
        (expected_cost_loss_acquisition, Direction.MINIMIZE),
        (expected_savings_score, Direction.MAXIMIZE),
        (max_profit_score, Direction.MAXIMIZE),
        (empc_score, Direction.MAXIMIZE),
        (mpc_score, Direction.MAXIMIZE),
        (empa_score, Direction.MAXIMIZE),
        (mpa_score, Direction.MAXIMIZE),
        (mpcs_score, Direction.MAXIMIZE),
        (empcs_score, Direction.MAXIMIZE),
        (empb_score, Direction.MAXIMIZE),
        (auepc_score, Direction.MAXIMIZE),
    ],
)
def test_prebuilt_metric_direction(metric, expected):
    """Every prebuilt metric object reports the direction of the strategy it was built from."""
    assert metric.direction is expected


@pytest.mark.parametrize(
    ('metric', 'parameters'),
    [
        (expected_cost_loss, CLASS_COSTS),
        (expected_log_cost_loss, CLASS_COSTS),
        (expected_savings_score, CLASS_COSTS),
        (max_profit_score, CLASS_COSTS),
        (empc_score, CHURN_PARAMS),
        (mpc_score, CHURN_PARAMS),
        (empcs_score, CREDIT_PARAMS),
    ],
)
def test_loss_is_minimized(metric, parameters):
    """``_loss`` equals the score for a MINIMIZE metric and its negation for a MAXIMIZE one."""
    score = metric(Y_TRUE, Y_SCORE, **parameters)
    loss = metric._loss(Y_TRUE, Y_SCORE, **parameters)

    sign = -1.0 if metric.direction is Direction.MAXIMIZE else 1.0
    assert loss == pytest.approx(sign * score)


def test_loss_ranks_a_maximize_metric_the_right_way():
    """A better model must produce a lower ``_loss``, even though it produces a higher score."""
    good = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.1, 0.9, 0.2])
    bad = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.9, 0.1, 0.8])

    assert expected_savings_score(Y_TRUE, good, **CLASS_COSTS) > expected_savings_score(Y_TRUE, bad, **CLASS_COSTS)
    assert expected_savings_score._loss(Y_TRUE, good, **CLASS_COSTS) < expected_savings_score._loss(
        Y_TRUE, bad, **CLASS_COSTS
    )


def test_loss_on_mixture_metric():
    """A mixture inherits ``_loss`` from ``BaseMetric``; it negates the combined score once."""
    score = empcs_score(Y_TRUE, Y_SCORE, **CREDIT_PARAMS)
    assert empcs_score._loss(Y_TRUE, Y_SCORE, **CREDIT_PARAMS) == pytest.approx(-score)


def test_loss_on_direction_inconsistent_mixture_raises():
    """``_loss`` is undefined when components disagree on direction, and must not guess a sign."""
    cost_matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')
    mixture = MixtureMetric([
        MixtureComponent(weight=0.5, metric=Metric(cost_matrix, Cost()), parameters={}),
        MixtureComponent(weight=0.5, metric=Metric(cost_matrix, Savings()), parameters={}),
    ])

    with pytest.raises(ValueError, match='inconsistent optimization directions'):
        mixture._loss(Y_TRUE, Y_SCORE, fp_cost=1.0, fn_cost=5.0)
