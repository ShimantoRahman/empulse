import pytest

from empulse.metrics import max_profit, mpc


def test_mpc_replication():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    clv = 200
    d = 10
    f = 1
    gamma = 0.3
    tp_benefit = clv * (gamma * (1 - (d / clv)) - (f / clv))
    fp_cost = d + f

    mp_score, mp_threshold = max_profit(y_true, y_pred, tp_benefit=tp_benefit, fp_cost=fp_cost)
    mpc_score, mpc_threshold = mpc(y_true, y_pred, clv=clv, incentive_cost=d, contact_cost=f, accept_rate=gamma)
    assert mp_score == pytest.approx(mpc_score)
    assert mp_threshold == pytest.approx(mpc_threshold)
