"""
Cross-checks between prebuilt metrics that should agree by construction.

This file used to compare ``reference.max_profit.max_profit`` against ``reference.churn.mpc`` --
reference against reference, so it could never catch a regression in the shipped package. Both
sides now call the public metrics.
"""

import warnings

import numpy as np
import pytest

from empulse.metrics import empc_score, empcs_score, max_profit_score, mpc_score, mpcs_score


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


@pytest.mark.parametrize(
    ('metric', 'label', 'expected_score', 'expected_rate'),
    [
        # Each targeted churner accepts with probability 0.3 and is then worth the CLV net of the
        # incentive and the contact; otherwise the contact is lost.
        (mpc_score, 1, 0.3 * (200 - 10 - 1) - 0.7 * 1, 1.0),
        (mpc_score, 0, 0.0, 0.0),
        # The same for the expected MPC: the accept rate follows Beta(6, 14), whose mean is 0.3.
        (empc_score, 1, 0.3 * (200 - 10 - 1) - 0.7 * 1, None),
        (empc_score, 0, 0.0, None),
        # Rejecting a defaulter saves the fraction of the loan that would be lost, 0.275 by default.
        (mpcs_score, 1, 0.275, 1.0),
        (mpcs_score, 0, 0.0, 0.0),
        # The lost fraction is 0 w.p. 0.55, 1 w.p. 0.1 and uniform on [0, 1] otherwise: 0.1 + 0.35 / 2.
        (empcs_score, 1, 0.1 + 0.35 * 0.5, None),
        (empcs_score, 0, 0.0, None),
    ],
)
def test_max_profit_of_a_single_class(metric, label, expected_score, expected_rate):
    """With only positives, targeting everyone is best; with only negatives, targeting no one is."""
    y_true = np.full(4, label)
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)  # no 0/0 on the way
        assert metric(y_true, y_score) == pytest.approx(expected_score, rel=1e-9)
        if expected_rate is not None:
            assert metric.optimal_rate(y_true, y_score) == expected_rate
