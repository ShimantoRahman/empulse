import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sympy.stats import Beta, Normal, Uniform

from empulse.metrics import (
    Metric,
    empc_score,
    expected_cost_loss,
    expected_cost_loss_churn,
    expected_savings_score,
    make_objective_aec,
    make_objective_churn,
    max_profit_score,
    mpc_score,
)


@pytest.fixture(scope='module')
def y_true_and_prediction():
    X, y = make_classification(random_state=12)
    lr = LogisticRegression()
    lr.fit(X, y)
    y_proba = lr.predict_proba(X)[:, 1]
    return y, y_proba


@pytest.mark.parametrize('integration_method', Metric.INTEGRATION_METHODS)
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
):
    y, y_proba = y_true_and_prediction
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Beta('gamma', alpha, beta)
    profit_func = (
        Metric('max profit')
        .set_integration_method(integration_method)
        .set_random_state(12)
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, beta=gamma_beta, alpha=gamma_alpha
    )
    empc_result = empc_score(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        beta=gamma_beta,
        alpha=gamma_alpha,
    )
    if integration_method == 'monte-carlo':  # Monte Carlo methods have higher variance
        assert pytest.approx(metric_result, rel=1e-2) == empc_result
    elif integration_method == 'quasi-monte-carlo':  # Quasi-Monte Carlo methods have lower variance
        assert pytest.approx(metric_result, rel=1e-4) == empc_result
    else:
        assert pytest.approx(metric_result) == empc_result


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, accept_rate',
    [(100, 10, 1, 0.3), (200, 20, 2, 0.2), (150, 15, 1.5, 0.4)],
)
def test_metric_vs_mpc_score(customer_lifetime_value, incentive_cost, contact_cost, accept_rate, y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    profit_func = (
        Metric('max profit')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    mpc_result = mpc_score(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == mpc_result


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, accept_rate',
    [(100, 10, 1, 0.3), (200, 20, 2, 0.2), (150, 15, 1.5, 0.4)],
)
def test_metric_vs_mpc_score_inversed(
    customer_lifetime_value, incentive_cost, contact_cost, accept_rate, y_true_and_prediction
):
    y, y_proba = y_true_and_prediction
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    profit_func = (
        Metric('max profit')
        .add_tp_cost(-gamma * (clv - d - f))
        .add_tp_cost(-(1 - gamma) * -f)
        .add_fp_benefit(-d - f)
        .build()
    )

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    mpc_result = mpc_score(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    assert pytest.approx(metric_result) == mpc_result


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_fraction, contact_cost, accept_rate',
    [(100, 0.05, 1, 0.3), (200, 0.1, 2, 0.2), (150, 0.15, 1.5, 0.4)],
)
def test_metric_vs_expected_loss(
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate, y_true_and_prediction
):
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .build()
    )

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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


@pytest.mark.parametrize(
    'customer_lifetime_value, incentive_cost, contact_cost, accept_rate',
    [(100, 5, 1, 0.3), (200, 10, 2, 0.2), (150, 15, 1.5, 0.4)],
)
def test_metric_vs_savings(customer_lifetime_value, incentive_cost, contact_cost, accept_rate, y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    profit_func = (
        Metric('savings')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .build()
    )

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, gamma=accept_rate
    )
    tp_cost = (
        -accept_rate * (customer_lifetime_value - incentive_cost - contact_cost) + (1 - accept_rate) * contact_cost
    )
    fp_cost = incentive_cost + contact_cost
    cost_result = expected_savings_score(
        y,
        y_proba,
        tp_cost=tp_cost,
        fp_cost=fp_cost,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_savings(y_true_and_prediction):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    c, r, d, b = sympy.symbols('c r d b')
    profit_func = (
        Metric('savings')
        .add_tp_cost(c)
        .add_fp_cost(c)
        .add_fn_cost(r * d * b)
        .alias('contact_cost', c)
        .alias('interest_rate', r)
        .alias('deposit_fraction', d)
        .alias('balance', b)
        .build()
    )

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


def test_metric_upsell_cost(y_true_and_prediction):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    c, r, d, b = sympy.symbols('c r d b')
    profit_func = (
        Metric('cost')
        .add_tp_cost(c)
        .add_fp_cost(c)
        .add_fn_cost(r * d * b)
        .alias('contact_cost', c)
        .alias('interest_rate', r)
        .alias('deposit_fraction', d)
        .alias('balance', b)
        .build()
    )

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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_cost_inverse(y_true_and_prediction):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    c, r, d, b = sympy.symbols('c r d b')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(-c)
        .add_fp_benefit(-c)
        .add_fn_benefit(-r * d * b)
        .alias('contact_cost', c)
        .alias('interest_rate', r)
        .alias('deposit_fraction', d)
        .alias('balance', b)
        .build()
    )

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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_cost_inverse_matrix(y_true_and_prediction):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction = 1, 0.02463333, 0.25
    balance = np.arange(len(y))
    c, r, d, b = sympy.symbols('c r d b')
    profit_func = (
        Metric('cost')
        .add_tn_cost(c)
        .add_fn_cost(c)
        .add_fp_cost(r * d * b)
        .alias('contact_cost', c)
        .alias('interest_rate', r)
        .alias('deposit_fraction', d)
        .alias('balance', b)
        .build()
    )

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
        tn_cost=contact_cost,
        fn_cost=contact_cost,
        fp_cost=interest_rate * deposit_fraction * balance,
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_upsell_max_profit(y_true_and_prediction):
    """Case from Bank Telemarketing Upsell Campaign dataset"""
    y, y_proba = y_true_and_prediction
    contact_cost, interest_rate, deposit_fraction, balance = 1, 0.02463333, 0.25, 100
    c, r, d, b = sympy.symbols('c r d b')
    profit_func = (
        Metric('max profit')
        .add_tp_cost(c)
        .add_fp_cost(c)
        .add_fn_cost(r * d * b)
        .alias('contact_cost', c)
        .alias('interest_rate', r)
        .alias('deposit_fraction', d)
        .alias('balance', b)
        .build()
    )

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
        tp_benefit=-contact_cost,
        fp_cost=contact_cost,
        fn_cost=interest_rate * deposit_fraction * balance,
    )
    assert pytest.approx(metric_result) == profit_result


def test_metric_str_symbols(y_true_and_prediction):
    y, y_proba = y_true_and_prediction
    profit_func = (
        Metric('cost')
        .add_tp_benefit('a')
        .add_tp_cost('k')
        .add_tn_benefit('b')
        .add_tn_cost('b')
        .add_fp_benefit('c')
        .add_fp_cost('c')
        .add_fn_benefit('d')
        .add_fn_cost('d')
        .build()
    )

    metric_result = profit_func(y, y_proba, a=1, k=1)
    assert pytest.approx(metric_result) == 0.0


def test_metric_arraylikes(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .build()
    )
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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


@pytest.mark.parametrize('integration_method', Metric.INTEGRATION_METHODS)
def test_metric_uniform_dist(y_true_and_prediction, integration_method):
    customer_lifetime_value, incentive_cost, contact_cost = 100, 10, 1
    y, y_proba = y_true_and_prediction
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    profit_func = (
        Metric('max profit')
        .set_integration_method(integration_method)
        .set_random_state(12)
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )

    metric_result = profit_func(
        y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost, beta=1, alpha=0
    )
    if integration_method == 'monte-carlo':
        assert pytest.approx(metric_result, rel=1e-2) == 21.14892314814815
    else:
        assert pytest.approx(metric_result) == 21.14892314814815


@pytest.mark.parametrize('integration_method', Metric.INTEGRATION_METHODS)
def test_metric_uniform_dist_no_params(y_true_and_prediction, integration_method):
    customer_lifetime_value, incentive_cost, contact_cost = 100, 10, 1
    y, y_proba = y_true_and_prediction
    clv, d, f = sympy.symbols('clv d f')
    gamma = Uniform('gamma', 0, 1)
    profit_func = (
        Metric('max profit')
        .set_integration_method(integration_method)
        .set_random_state(12)
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )

    metric_result = profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_cost, f=contact_cost)
    if integration_method == 'monte-carlo':
        assert pytest.approx(metric_result, rel=1e-2) == 21.14892314814815
    else:
        assert pytest.approx(metric_result) == 21.14892314814815


@pytest.mark.parametrize('integration_method', Metric.INTEGRATION_METHODS)
def test_metric_normal_dist(y_true_and_prediction, integration_method):
    accept_rate, incentive_cost, contact_cost = 0.3, 10, 1
    y, y_proba = y_true_and_prediction
    gamma, d, f, mu, sigma = sympy.symbols('gamma d f mu sigma')
    clv = Normal('clv', mu, sigma)
    profit_func = (
        Metric('max profit')
        .set_integration_method(integration_method)
        .set_random_state(12)
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )

    metric_result = profit_func(y, y_proba, gamma=accept_rate, d=incentive_cost, f=contact_cost, mu=100, sigma=10)
    if integration_method == 'monte-carlo':
        assert pytest.approx(metric_result, rel=1e-2) == 12.150199167337625
    else:
        assert pytest.approx(metric_result) == 12.150199167337625


def test_metric_alias(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .build()
    )
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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_alias_symbols(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .alias('incentive_fraction', delta)
        .alias('contact_cost', f)
        .alias('accept_rate', gamma)
        .build()
    )
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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


def test_metric_alias_wrong_types(y_true_and_prediction):
    clv = sympy.symbols('clv')
    profit_func = Metric('cost').add_tp_benefit(clv)
    with pytest.raises(ValueError, match=r'Either a dictionary or both an alias and a symbol should be provided'):
        profit_func.alias(None)  # type: ignore


def test_metric_set_default(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_fraction=0.05, contact_cost=1, accept_rate=0.3, clv=100)
        .build()
    )
    metric_result = profit_func(y, y_proba)
    cost_result = expected_cost_loss_churn(
        y,
        y_proba,
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


def test_calling_before_building():
    clv = sympy.symbols('clv')
    profit_func = Metric('max profit').add_tp_benefit(clv)
    with pytest.raises(
        ValueError, match=r'The metric function has not been built. Call the build method before calling the metric'
    ):
        profit_func(np.array([1]), np.array([1]), d=0.05, f=1, gamma=0.3)


def test_unsupported_kind():
    clv = sympy.symbols('clv')
    with pytest.raises(ValueError, match=r'Kind unsupported is not supported. Supported values are'):
        Metric('unsupported').add_tp_benefit(clv).build()  # type: ignore


def test_unsupported_integration_method():
    clv = sympy.symbols('clv')
    with pytest.raises(ValueError, match=r'Integration method unsupported is not supported. Supported values are'):
        Metric('max profit').set_integration_method('unsupported').add_tp_benefit(clv).build()  # type: ignore


@pytest.mark.parametrize('integration_method', Metric.INTEGRATION_METHODS)
def test_missing_arguments(y_true_and_prediction, integration_method):
    customer_lifetime_value, incentive_fraction, contact_cost = 100, 0.05, 1
    y, y_proba = y_true_and_prediction
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    profit_func = (
        Metric('max profit')
        .set_integration_method(integration_method)
        .set_random_state(12)
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )
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


@pytest.mark.parametrize('kind', Metric.METRIC_TYPES)
def test_missing_arguments_deterministic(y_true_and_prediction, kind):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    profit_func = (
        Metric(kind).add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost('d + f').build()
    )
    with pytest.raises(ValueError, match=r'Metric expected a value for clv, did not receive it.'):
        profit_func(y, y_proba, d=incentive_fraction, f=contact_cost, gamma=accept_rate)
    with pytest.raises(ValueError, match=r'Metric expected a value for d, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, f=contact_cost, gamma=accept_rate)
    with pytest.raises(ValueError, match=r'Metric expected a value for f, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, gamma=accept_rate)
    with pytest.raises(ValueError, match=r'Metric expected a value for gamma, did not receive it.'):
        profit_func(y, y_proba, clv=customer_lifetime_value, d=incentive_fraction, f=contact_cost)


def test_random_var_cost_loss():
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    cost_func = Metric('cost').add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(d + f)
    with pytest.raises(NotImplementedError, match=r'Random variables are not supported for the cost metric.'):
        cost_func.build()


def test_random_var_savings_score():
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    savings_func = (
        Metric('savings').add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(d + f)
    )
    with pytest.raises(NotImplementedError, match=r'Random variables are not supported for the savings metric.'):
        savings_func.build()


def test_objective_aec_gradient_boost(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction

    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .build()
    )
    gradient, hessian = profit_func.gradient_boost_objective(
        y,
        y_proba,
        clv=customer_lifetime_value,
        delta=incentive_fraction,
        f=contact_cost,
        gamma=accept_rate,
    )
    objective = make_objective_churn(
        model='xgboost',
        clv=customer_lifetime_value,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
    )
    gradient_true, hessian_true = objective(y_proba, y)
    assert np.allclose(gradient, gradient_true)
    assert np.allclose(hessian, hessian_true)


def test_objective_aec_logit():
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    X, y = make_classification(random_state=12)
    weights = np.zeros(X.shape[1])

    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    profit_func = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - delta * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f)
        .build()
    )
    metric, gradient = profit_func.logit_objective(
        X,
        weights,
        y,
        clv=customer_lifetime_value,
        delta=incentive_fraction,
        f=contact_cost,
        gamma=accept_rate,
    )
    tp_benefit = accept_rate * (customer_lifetime_value - incentive_fraction * customer_lifetime_value - contact_cost)
    tp_benefit += (1 - accept_rate) * -contact_cost
    fp_cost = incentive_fraction * customer_lifetime_value + contact_cost
    objective = make_objective_aec(
        model='cslogit',
        tp_cost=-tp_benefit,
        fp_cost=fp_cost,
    )
    metric_true, gradient_true = objective(X, weights, y)
    assert pytest.approx(metric) == metric_true
    assert np.allclose(gradient, gradient_true)


@pytest.mark.parametrize('kind', ['max profit', 'savings'])
def test_objective_logit_unsupported(kind):
    clv = sympy.symbols('clv')
    metric = Metric(kind).add_tp_benefit(clv).build()
    with pytest.raises(NotImplementedError, match=r'Gradient of the logit function is not defined for this kind'):
        metric.logit_objective(np.array([1]), np.array([1]), np.array([1]))


@pytest.mark.parametrize('kind', ['max profit', 'savings'])
def test_objective_boost_unsupported(kind):
    clv = sympy.symbols('clv')
    metric = Metric(kind).add_tp_benefit(clv).build()
    with pytest.raises(
        NotImplementedError,
        match=r'Gradient and Hessian of the gradient boosting function is not defined for this kind',
    ):
        metric.gradient_boost_objective(np.array([1]), np.array([1]))


def test_repr_metric():
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    profit_func = (
        Metric('max profit')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )
    assert repr(profit_func) == (
        "Metric(kind='max profit', integration_method='auto', n_mc_samples=65536, random_state=RandomState(MT19937), "
        'tp_benefit=-f*(1 - gamma) + (clv - d - f)*gamma, tn_benefit=0, fp_cost=d + f, fn_cost=0)'
    )


def test_repr_latex_max_profit():
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    profit_func = (
        Metric('max profit')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost('d + f')
        .build()
    )
    assert profit_func._repr_latex_() == (
        '$\\displaystyle \\int\\limits_{\\alpha}^{\\beta} \\begin{cases} \\frac{F_{0} \\pi_{0} \\left(f \\left(1 - '
        '\\gamma\\right) - \\left(clv - d - f\\right) \\gamma\\right) - F_{1} \\pi_{1} \\left(d + f\\right)}{- \\alpha '
        '+ \\beta} & \\text{for}\\: \\beta \\geq \\gamma \\wedge \\alpha \\leq \\gamma \\\\0 & \\text{otherwise} '
        '\\end{cases}\\, d\\gamma$'
    )


def test_repr_latex_savings():
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    savings_func = (
        Metric('savings')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .build()
    )
    assert savings_func._repr_latex_() == (
        '$\\displaystyle \\frac{\\sum_{i=0}^{N} \\left(s_{i} y_{i} \\left(f_{i} \\left(1 - \\gamma_{i}\\right) - '
        '\\gamma_{i} \\left(clv_{i} - d_{i} - f_{i}\\right)\\right) + s_{i} \\left(1 - y_{i}\\right) \\left(d_{i} + '
        'f_{i}\\right)\\right)}{N \\min\\left(Cost_{0}, Cost_{1}\\right)}$'
    )


def test_repr_latex_cost():
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    cost_func = (
        Metric('cost').add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(d + f).build()
    )
    assert cost_func._repr_latex_() == (
        '$\\displaystyle \\frac{\\sum_{i=0}^{N} \\left(s_{i} y_{i} \\left(f_{i} \\left(1 - \\gamma_{i}\\right) - '
        '\\gamma_{i} \\left(clv_{i} - d_{i} - f_{i}\\right)\\right) + s_{i} \\left(1 - y_{i}\\right) \\left(d_{i} + '
        'f_{i}\\right)\\right)}{N}$'
    )


def test_metric_context_manager(y_true_and_prediction):
    customer_lifetime_value, incentive_fraction, contact_cost, accept_rate = 100, 0.05, 1, 0.3
    y, y_proba = y_true_and_prediction
    clv, delta, f, gamma = sympy.symbols('clv delta f gamma')
    with Metric('cost') as profit_func:
        profit_func.add_tp_benefit(gamma * (clv - delta * clv - f))
        profit_func.add_tp_benefit((1 - gamma) * -f)
        profit_func.add_fp_cost(delta * clv + f)
        profit_func.alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})

    assert profit_func._built

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
        normalize=True,
    )
    assert pytest.approx(metric_result) == cost_result


# # Extract sympy distributions from _sympy_dist_to_scipy
# sympy_distributions = list(Metric._sympy_dist_to_scipy.keys())
#
# @pytest.mark.parametrize('sympy_dist', sympy_distributions)
# def test_sympy_distributions(sympy_dist):
#     clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
#     gamma = sympy.stats.crv_types.rv('gamma', sympy_dist, (alpha, beta))
#
#     profit_func = (
#         Metric('max profit')
#         .add_tp_benefit(gamma * (clv - d - f))
#         .add_tp_benefit((1 - gamma) * -f)
#         .add_fp_cost('d + f')
#         .build()
#     )
#
#     y_true = [1, 0, 1, 0, 1]
#     y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
#
#     # Test with some default values
#     result = profit_func(y_true, y_proba, clv=100, d=10, f=1, alpha=0.5, beta=1)
#     assert isinstance(result, float)
