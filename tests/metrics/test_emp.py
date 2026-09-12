"""
Cross-checks between prebuilt metrics that should agree by construction.

This file used to compare ``reference.max_profit.max_profit`` against ``reference.churn.mpc`` --
reference against reference, so it could never catch a regression in the shipped package. Both
sides now call the public metrics.
"""

import pytest

from empulse.metrics import max_profit_score, mpc_score


@pytest.mark.parametrize(
    ('clv', 'incentive_cost', 'contact_cost', 'accept_rate'),
    [
        (200, 10, 1, 0.3),
        (500, 25, 5, 0.5),
        (1000, 100, 10, 0.1),
    ],
    ids=['default_like', 'mid', 'expensive_incentive'],
)
def test_mpc_score_is_max_profit_with_the_churn_cost_matrix(clv, incentive_cost, contact_cost, accept_rate):
    """
    MPC is just the generic max-profit measure evaluated on the churn cost matrix.

    A true positive earns the retained CLV, net of the incentive and the contact, but only when the
    customer accepts; a false positive costs the incentive plus the contact. Deriving those two
    numbers by hand and feeding them to ``max_profit_score`` must reproduce ``mpc_score`` exactly.
    """
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    # `max_profit_score` is parameterised in costs, and a benefit is a negative cost.
    tp_cost = -(clv * (accept_rate * (1 - (incentive_cost / clv)) - (contact_cost / clv)))
    fp_cost = incentive_cost + contact_cost

    generic = max_profit_score(y_true, y_score, tp_cost=tp_cost, fp_cost=fp_cost)
    churn = mpc_score(
        y_true,
        y_score,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert generic == pytest.approx(churn)


@pytest.mark.parametrize(
    ('clv', 'incentive_cost', 'contact_cost', 'accept_rate'),
    [
        (200, 10, 1, 0.3),
        (500, 25, 5, 0.5),
    ],
    ids=['default_like', 'mid'],
)
def test_mpc_optimal_rate_is_max_profit_optimal_rate(clv, incentive_cost, contact_cost, accept_rate):
    """The same equivalence must hold for the rate at which the maximum is attained."""
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    tp_cost = -(clv * (accept_rate * (1 - (incentive_cost / clv)) - (contact_cost / clv)))
    fp_cost = incentive_cost + contact_cost

    generic = max_profit_score.optimal_rate(y_true, y_score, tp_cost=tp_cost, fp_cost=fp_cost)
    churn = mpc_score.optimal_rate(
        y_true,
        y_score,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert generic == pytest.approx(churn)
