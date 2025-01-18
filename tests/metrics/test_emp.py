import pytest

from empulse.metrics import emp, empc, max_profit, mpc


def test_empc_replication():
    from scipy.stats import beta

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    clv = 200
    d = 10
    f = 1
    tp_benefit = (-f, clv * (1 - (d / clv) - (f / clv)))

    fp_cost = d + f

    def weighted_pdf(b0, b1, c0, c1, b0_step, b1_step, c0_step, c1_step):
        gamma = (b0 + f) / (clv - d)
        gamma_step = b0_step / (clv - d)
        return beta.pdf(gamma, a=6, b=14) * gamma_step

    emp_score, emp_threshold = emp(
        y_true, y_pred, weighted_pdf=weighted_pdf, tp_benefit=tp_benefit, fp_cost=fp_cost, n_buckets=1_000
    )
    empc_score, empc_threshold = empc(y_true, y_pred, clv=clv, incentive_cost=d, contact_cost=f)
    assert emp_score == pytest.approx(empc_score, abs=1e-3)
    assert emp_threshold == pytest.approx(empc_threshold, abs=1e-3)


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
