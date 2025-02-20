import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sympy.stats import Beta, Normal, Uniform

from empulse.metrics import Metric, empc_score, expected_cost_loss_churn, expected_savings_score, mpc_score


@pytest.fixture(scope='module')
def y_true_and_prediction():
    X, y = make_classification(random_state=12)
    lr = LogisticRegression()
    lr.fit(X, y)
    y_proba = lr.predict_proba(X)[:, 1]
    return y, y_proba


@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
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
        Metric('max profit', integration_method=integration_method, random_state=12)
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


@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_metric_uniform_dist(y_true_and_prediction, integration_method):
    customer_lifetime_value, incentive_cost, contact_cost = 100, 10, 1
    y, y_proba = y_true_and_prediction
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = Uniform('gamma', alpha, beta)
    profit_func = (
        Metric('max profit', integration_method=integration_method, random_state=12)
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


@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_metric_uniform_dist_no_params(y_true_and_prediction, integration_method):
    customer_lifetime_value, incentive_cost, contact_cost = 100, 10, 1
    y, y_proba = y_true_and_prediction
    clv, d, f = sympy.symbols('clv d f')
    gamma = Uniform('gamma', 0, 1)
    profit_func = (
        Metric('max profit', integration_method=integration_method, random_state=12)
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


@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_metric_normal_dist(y_true_and_prediction, integration_method):
    accept_rate, incentive_cost, contact_cost = 0.3, 10, 1
    y, y_proba = y_true_and_prediction
    gamma, d, f, mu, sigma = sympy.symbols('gamma d f mu sigma')
    clv = Normal('clv', mu, sigma)
    profit_func = (
        Metric('max profit', integration_method=integration_method, random_state=12)
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
