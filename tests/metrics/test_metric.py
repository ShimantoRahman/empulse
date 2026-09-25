"""
:class:`~empulse.metrics.Metric` built from a cost matrix, checked against the reference
implementations in ``tests/metrics/reference/``: its scores, objectives, aliases and defaults, and
how it prints.
"""

import numpy as np
import pytest
import sympy
from sympy.stats import Normal, Uniform

from empulse.metrics import (
    Cost,
    CostMatrix,
    MaxProfit,
    Metric,
    MinCost,
    Profit,
    Savings,
    expected_cost_loss,
    expected_cost_loss_churn,
    expected_savings_score,
    max_profit_score,
)
from empulse.metrics._loss import cy_boost_grad_hess

from .reference.churn import empc, make_objective_churn, mpc

METRIC_STRATEGIES = [
    Cost(),
    MaxProfit(),
    Savings(),
    Profit(),
    MinCost(),
]


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, gamma_alpha, gamma_beta',
    [(100, 10, 1, 6, 14), (200, 20, 2, 8, 16), (150, 15, 1.5, 10, 20)],
)
def test_metric_vs_empc_score(
    customer_lifetime_value,
    incentive_cost,
    contact_cost,
    gamma_alpha,
    gamma_beta,
    y_true_and_prediction,
    integration_method,
    stochastic_churn_cost_matrix,
):
    y, y_proba = y_true_and_prediction
    profit_func = Metric(
        stochastic_churn_cost_matrix, MaxProfit(integration_method=integration_method, random_state=12)
    )

    metric_val = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, beta=gamma_beta, alpha=gamma_alpha
    )
    metric_rate = profit_func.optimal_rate(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, beta=gamma_beta, alpha=gamma_alpha
    )
    empc_val, empc_rate = empc(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        beta=gamma_beta,
        alpha=gamma_alpha,
    )
    if integration_method == 'monte-carlo':  # Monte Carlo methods have higher variance
        assert pytest.approx(metric_val, rel=1e-2) == empc_val
        assert pytest.approx(metric_rate, rel=1e-2) == empc_rate
    elif integration_method == 'quasi-monte-carlo':  # Quasi-Monte Carlo methods have lower variance
        assert pytest.approx(metric_val, rel=1e-4) == empc_val
        assert pytest.approx(metric_rate, rel=1e-4) == empc_rate
    else:
        assert pytest.approx(metric_val) == empc_val
        assert pytest.approx(metric_rate) == empc_rate


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, accept_rate',
    [(100, 10, 1, 0.3), (200, 20, 2, 0.2), (150, 15, 1.5, 0.4)],
)
def test_metric_vs_mpc_score(
    customer_lifetime_value, incentive_cost, contact_cost, accept_rate, y_true_and_prediction, churn_cost_matrix
):
    y, y_proba = y_true_and_prediction
    profit_func = Metric(churn_cost_matrix, MaxProfit())

    metric_val = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    metric_rate = profit_func.optimal_rate(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    mpc_val, mpc_rate = mpc(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_val) == mpc_val
    assert pytest.approx(metric_rate) == mpc_rate


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, accept_rate',
    [(100, 10, 1, 0.3), (200, 20, 2, 0.2), (150, 15, 1.5, 0.4)],
)
def test_metric_vs_mpc_score_inversed(
    customer_lifetime_value, incentive_cost, contact_cost, accept_rate, y_true_and_prediction
):
    y, y_proba = y_true_and_prediction
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    cost_matrix = CostMatrix().add_tp_cost(-gamma * (clv - d - f)).add_tp_cost(-(1 - gamma) * -f).add_fp_benefit(-d - f)
    profit_func = Metric(cost_matrix, MaxProfit())

    metric_val = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    metric_rate = profit_func.optimal_rate(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    mpc_val, mpc_rate = mpc(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_val) == mpc_val
    assert pytest.approx(metric_rate) == mpc_rate


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_fraction, contact_cost, accept_rate',
    [(100, 0.05, 1, 0.3), (200, 0.1, 2, 0.2), (150, 0.15, 1.5, 0.4)],
)
def test_metric_vs_expected_loss(
    customer_lifetime_value,
    incentive_fraction,
    contact_cost,
    accept_rate,
    y_true_and_prediction,
    delta_churn_cost_matrix,
):
    y, y_proba = y_true_and_prediction
    profit_func = Metric(delta_churn_cost_matrix, Cost())

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, delta=incentive_fraction, f=contact_cost, gamma=accept_rate
    )
    cost_result = expected_cost_loss_churn(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == cost_result


@pytest.mark.parametrize('baseline', ['zero_one', 'one', 'zero', 'prior', np.ones(100) * 0.3])
@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, accept_rate',
    [(100, 5, 1, 0.3), (200, 10, 2, 0.2), (150, 15, 1.5, 0.4)],
)
def test_metric_vs_savings(
    customer_lifetime_value,
    incentive_cost,
    contact_cost,
    accept_rate,
    y_true_and_prediction,
    churn_cost_matrix,
    baseline,
):
    y, y_proba = y_true_and_prediction
    cost_matrix = CostMatrix().add_tp_cost('tp').add_tn_cost('tn').add_fp_cost('fp').add_fn_cost('fn')
    profit_func = Metric(cost_matrix, Savings())

    tp_cost = (
        -accept_rate * (customer_lifetime_value - incentive_cost - contact_cost) + (1 - accept_rate) * contact_cost
    )
    fp_cost = incentive_cost + contact_cost
    metric_result = profit_func(y, y_proba, tp=tp_cost, fp=fp_cost, fn=contact_cost, tn=contact_cost, baseline=baseline)

    cost_result = expected_savings_score(
        y, y_proba, tp_cost=tp_cost, fp_cost=fp_cost, fn_cost=contact_cost, tn_cost=contact_cost, baseline=baseline
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_savings(y_true_and_prediction, bank_upsell_cost_matrix):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    profit_func = Metric(bank_upsell_cost_matrix, Savings())

    metric_result = profit_func(
        y,
        y_proba,
        contact_cost=contact_cost,
        interest_rate=interest_rate,
        deposit_fraction=deposit_fraction,
        balance=balance,
    )
    savings_result = expected_savings_score(
        y,
        y_proba,
        tp_cost=contact_cost,
        fp_cost=contact_cost,
        fn_cost=interest_rate * deposit_fraction * balance,
    )
    assert pytest.approx(metric_result) == savings_result


def test_metric_upsell_cost(y_true_and_prediction, bank_upsell_cost_matrix):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    profit_func = Metric(bank_upsell_cost_matrix, Cost())

    metric_result = profit_func(
        y,
        y_proba,
        contact_cost=contact_cost,
        interest_rate=interest_rate,
        deposit_fraction=deposit_fraction,
        balance=balance,
    )
    cost_result = expected_cost_loss(
        y,
        y_proba,
        tp_cost=contact_cost,
        fp_cost=contact_cost,
        fn_cost=interest_rate * deposit_fraction * balance,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_cost_inverse(y_true_and_prediction):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    c, r, d, b = sympy.symbols('c r d b')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(-c)
        .add_fp_benefit(-c)
        .add_fn_benefit(-r * d * b)
        .alias('contact_cost', c)
        .alias('interest_rate', r)
        .alias('deposit_fraction', d)
        .alias('balance', b)
    )
    profit_func = Metric(cost_matrix, Cost())

    metric_result = profit_func(
        y,
        y_proba,
        contact_cost=contact_cost,
        interest_rate=interest_rate,
        deposit_fraction=deposit_fraction,
        balance=balance,
    )
    cost_result = expected_cost_loss(
        y,
        y_proba,
        tp_cost=contact_cost,
        fp_cost=contact_cost,
        fn_cost=interest_rate * deposit_fraction * balance,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_cost_matrix(y_true_and_prediction, bank_upsell_cost_matrix):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    profit_func = Metric(bank_upsell_cost_matrix, Cost())

    metric_result = profit_func(
        y,
        y_proba,
        contact_cost=contact_cost,
        interest_rate=interest_rate,
        deposit_fraction=deposit_fraction,
        balance=balance,
    )
    cost_result = expected_cost_loss(
        y,
        y_proba,
        tp_cost=contact_cost,
        fp_cost=contact_cost,
        fn_cost=interest_rate * deposit_fraction * balance,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_max_profit(y_true_and_prediction, bank_upsell_cost_matrix):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction, balance = 1, 0.02463333, 0.25, 100
    profit_func = Metric(bank_upsell_cost_matrix, MaxProfit())

    metric_result = profit_func(
        y,
        y_proba,
        contact_cost=contact_cost,
        interest_rate=interest_rate,
        deposit_fraction=deposit_fraction,
        balance=balance,
    )
    profit_result = max_profit_score(
        y,
        y_proba,
        tp_cost=contact_cost,
        fp_cost=contact_cost,
        fn_cost=interest_rate * deposit_fraction * balance,
    )
    assert pytest.approx(metric_result) == profit_result


def test_metric_str_symbols(y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit('a')
        .add_tp_cost('k')
        .add_tn_benefit('b')
        .add_tn_cost('b')
        .add_fp_benefit('c')
        .add_fp_cost('c')
        .add_fn_benefit('d')
        .add_fn_cost('d')
    )
    profit_func = Metric(cost_matrix, Cost())

    metric_result = profit_func(y, y_proba, a=1, k=1)
    assert pytest.approx(metric_result) == 0.0


def test_metric_arraylikes(y_true_and_prediction, delta_churn_cost_matrix):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    profit_func = Metric(delta_churn_cost_matrix, Cost())

    clvs = [customer_lifetime_value] * len(y)
    deltas = [incentive_fraction] * len(y)
    fs = [contact_cost] * len(y)
    gammas = [accept_rate] * len(y)
    metric_result = profit_func(
        y,
        y_proba,
        clv=clvs,
        delta=deltas,
        f=fs,
        gamma=gammas,
    )
    cost_result = expected_cost_loss_churn(
        y,
        y_proba,
        clv=clvs,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == cost_result


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
def test_metric_uniform_dist(y_true_and_prediction, integration_method, uniform_dist_matrix):
    customer_lifetime_value, incentive_cost, contact_cost = 100, 10, 1
    y, y_proba = y_true_and_prediction
    profit_func = Metric(uniform_dist_matrix, MaxProfit(integration_method=integration_method, random_state=12))

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, beta=1, alpha=0
    )
    if integration_method == 'monte-carlo':
        assert pytest.approx(metric_result, rel=1e-2) == 21.14892314814815
    else:
        assert pytest.approx(metric_result) == 21.14892314814815


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
def test_metric_uniform_dist_no_params(y_true_and_prediction, integration_method):
    customer_lifetime_value, incentive_cost, contact_cost = 100, 10, 1
    y, y_proba = y_true_and_prediction
    clv, d, f = sympy.symbols('clv d f')
    gamma = Uniform('gamma', 0, 1)
    cost_matrix = (
        CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost('d + f')
    )
    profit_func = Metric(cost_matrix, MaxProfit(integration_method=integration_method, random_state=12))

    metric_result = profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost)
    if integration_method == 'monte-carlo':
        assert pytest.approx(metric_result, rel=1e-2) == 21.14892314814815
    else:
        assert pytest.approx(metric_result) == 21.14892314814815


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
def test_metric_normal_dist(y_true_and_prediction, integration_method):
    accept_rate, incentive_cost, contact_cost = 0.3, 10, 1
    y, y_proba = y_true_and_prediction
    gamma, d, f, mu, sigma = sympy.symbols('gamma d f mu sigma')
    clv = Normal('clv', mu, sigma)
    cost_matrix = (
        CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost('d + f')
    )
    profit_func = Metric(cost_matrix, MaxProfit(integration_method=integration_method, random_state=12))

    metric_result = profit_func(y, y_proba, gamma=accept_rate, d=incentive_cost, f=contact_cost, mu=100, sigma=10)
    if integration_method == 'monte-carlo':
        assert pytest.approx(metric_result, rel=1e-2) == 12.150199167337625
    else:
        assert pytest.approx(metric_result) == 12.150199167337625


def test_metric_alias(y_true_and_prediction, delta_churn_cost_matrix):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    delta_churn_cost_matrix.alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
    profit_func = Metric(delta_churn_cost_matrix, Cost())

    metric_result = profit_func(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    cost_result = expected_cost_loss_churn(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_alias_symbols(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .alias('incentive_fraction', delta)
        .alias('contact_cost', f)
        .alias('accept_rate', gamma)
    )
    profit_func = Metric(cost_matrix, Cost())
    metric_result = profit_func(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    cost_result = expected_cost_loss_churn(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_set_default(y_true_and_prediction, delta_churn_cost_matrix):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    (
        delta_churn_cost_matrix.alias({
            'incentive_fraction': 'delta',
            'contact_cost': 'f',
            'accept_rate': 'gamma',
        }).set_default(incentive_fraction=0.05, contact_cost=1, accept_rate=0.3, clv=100)
    )
    profit_func = Metric(delta_churn_cost_matrix, Cost())
    metric_result = profit_func(y, y_proba)
    cost_result = expected_cost_loss_churn(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == cost_result


def test_unsupported_integration_method():
    with pytest.raises(ValueError, match=r'Integration method unsupported is not supported. Supported values are'):
        Metric(CostMatrix(), MaxProfit(integration_method='unsupported'))  # type: ignore


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
def test_missing_arguments(y_true_and_prediction, integration_method, uniform_dist_matrix):
    customer_lifetime_value, incentive_fraction, contact_cost = 100, 0.05, 1
    y, y_proba = y_true_and_prediction
    profit_func = Metric(uniform_dist_matrix, MaxProfit(integration_method=integration_method, random_state=12))

    with pytest.raises(ValueError, match=r'Metric expected a value for clv, did not receive it.'):
        profit_func(y, y_proba, d=incentive_fraction, f=contact_cost, alpha=6, beta=14)
    with pytest.raises(ValueError, match=r'Metric expected a value for d, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, f=contact_cost, alpha=6, beta=14)
    with pytest.raises(ValueError, match=r'Metric expected a value for f, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, alpha=6, beta=14)
    with pytest.raises(ValueError, match=r'Metric expected a value for alpha, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, f=contact_cost, beta=14)
    with pytest.raises(ValueError, match=r'Metric expected a value for beta, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, f=contact_cost, alpha=6)


@pytest.mark.parametrize('strategy', METRIC_STRATEGIES)
def test_missing_arguments_deterministic(y_true_and_prediction, strategy, churn_cost_matrix):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    profit_func = Metric(churn_cost_matrix, strategy)

    with pytest.raises(ValueError, match=r'Metric expected a value for clv, did not receive it.'):
        profit_func(y, y_proba, d=incentive_fraction, f=contact_cost, gamma=accept_rate)
    with pytest.raises(ValueError, match=r'Metric expected a value for d, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, f=contact_cost, gamma=accept_rate)
    with pytest.raises(ValueError, match=r'Metric expected a value for f, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, gamma=accept_rate)
    with pytest.raises(ValueError, match=r'Metric expected a value for gamma, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, f=contact_cost)


def test_objective_aec_gradient_boost(y_true_and_prediction, delta_churn_cost_matrix):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    profit_func = Metric(delta_churn_cost_matrix, Cost())

    grad_const = profit_func._prepare_boost_objective(
        y,
        clv=customer_lifetime_value,
        delta=incentive_fraction,
        f=contact_cost,
        gamma=accept_rate,
    ).reshape(-1)
    gradient, hessian = cy_boost_grad_hess(y, y_proba, grad_const)

    objective = make_objective_churn(
        model='xgboost',
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    gradient_true, hessian_true = objective(y, y_proba)
    assert np.allclose(gradient, gradient_true)
    assert np.allclose(hessian, hessian_true)


def test_repr_metric(uniform_dist_matrix):
    profit_func = Metric(uniform_dist_matrix, MaxProfit())

    assert repr(profit_func) == (
        'Metric(cost_matrix=CostMatrix(tp_cost=f*(1 - gamma) - (clv - d - f)*gamma, '
        'tn_cost=0, fp_cost=d + f, fn_cost=0), '
        "strategy=MaxProfit(direction=Direction.MAXIMIZE, integration_method='auto', "
        'n_mc_samples=65536, random_state=Generator(PCG64)))'
    )


def test_repr_latex_max_profit(uniform_dist_matrix):
    """MaxProfit renders the profit being maximized: the TP benefit less the FP cost."""
    profit_func = Metric(uniform_dist_matrix, MaxProfit())
    assert profit_func._repr_latex_() == (
        '$\\displaystyle \\int\\limits_{\\alpha}^{\\beta} \\begin{cases} \\frac{F_{0} \\pi_{0} \\left(- f '
        '\\left(1 - \\gamma\\right) + \\left(\\mathrm{clv} - d - f\\right) \\gamma\\right) - F_{1} \\pi_{1} '
        '\\left(d + f\\right)}{- \\alpha + \\beta} & \\text{for}\\: \\beta \\geq \\gamma \\wedge \\alpha '
        '\\leq \\gamma \\\\0 & \\text{otherwise} \\end{cases}\\, d\\gamma$'
    )


def test_repr_latex_min_cost(uniform_dist_matrix):
    """MinCost renders the same quantity negated: every outcome's cost, added up."""
    cost_func = Metric(uniform_dist_matrix, MinCost())
    assert cost_func._repr_latex_() == (
        '$\\displaystyle \\int\\limits_{\\alpha}^{\\beta} \\begin{cases} \\frac{F_{0} \\pi_{0} \\left(f '
        '\\left(1 - \\gamma\\right) - \\left(\\mathrm{clv} - d - f\\right) \\gamma\\right) + F_{1} \\pi_{1} '
        '\\left(d + f\\right)}{- \\alpha + \\beta} & \\text{for}\\: \\beta \\geq \\gamma \\wedge \\alpha '
        '\\leq \\gamma \\\\0 & \\text{otherwise} \\end{cases}\\, d\\gamma$'
    )


def test_repr_latex_savings(churn_cost_matrix):
    savings_func = Metric(churn_cost_matrix, Savings())
    assert savings_func._repr_latex_() == (
        '$\\displaystyle \\frac{\\sum_{i=0}^{N} \\left(s_{i} y_{i} \\left(f_{i} \\left(1 - '
        '\\gamma_{i}\\right) - \\gamma_{i} \\left(\\mathrm{clv}_{i} - d_{i} - f_{i}\\right)\\right) + s_{i} '
        '\\left(1 - y_{i}\\right) \\left(d_{i} + f_{i}\\right)\\right)}{N \\min\\left(\\mathrm{Cost}_{0}, '
        '\\mathrm{Cost}_{1}\\right)}$'
    )


def test_repr_latex_cost(churn_cost_matrix):
    cost_func = Metric(churn_cost_matrix, Cost())
    assert cost_func._repr_latex_() == (
        '$\\displaystyle \\frac{\\sum_{i=0}^{N} \\left(s_{i} y_{i} \\left(f_{i} \\left(1 - '
        '\\gamma_{i}\\right) - \\gamma_{i} \\left(\\mathrm{clv}_{i} - d_{i} - f_{i}\\right)\\right) + s_{i} '
        '\\left(1 - y_{i}\\right) \\left(d_{i} + f_{i}\\right)\\right)}{N}$'
    )
