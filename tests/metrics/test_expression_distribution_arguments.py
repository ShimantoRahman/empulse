"""A distribution's arguments may be expressions of parameters, such as ``Beta('v', 2 * a, b)``."""

import warnings

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import CostMatrix, MaxProfit, Metric

from ._helpers import RANKING_FEATURES, RANKING_Y_SCORE, RANKING_Y_TRUE, logit_and_count_evaluations

A, B, CLV, D = sympy.symbols('a b clv d')
PARAMETERS = {'a': 3.0, 'b': 5.0}


# Each distribution with expression arguments, and the same distribution with their values filled in.
DISTRIBUTIONS = {
    'beta': (sympy.stats.Beta('v', 2 * A, B), sympy.stats.Beta('v', 6, 5)),
    'gamma': (sympy.stats.Gamma('v', A + 1, B / 10), sympy.stats.Gamma('v', 4, 0.5)),
    # The support of a uniform variable is given by its parameters.
    'uniform': (sympy.stats.Uniform('v', A / 10, A * B / 25), sympy.stats.Uniform('v', 0.3, 0.6)),
    'normal': (sympy.stats.Normal('v', A / 10, 1 / B), sympy.stats.Normal('v', 0.3, 0.2)),
}


def _metric(variable, integration_method, polynomial, extra=0):
    benefit = (variable * CLV if polynomial else sympy.sqrt(variable) * CLV) + extra
    cost_matrix = CostMatrix().add_tp_benefit(benefit).add_fp_cost(D).set_default(d=10, clv=100)
    return Metric(cost_matrix, MaxProfit(integration_method=integration_method, n_mc_samples_exp=10, random_state=0))


def _real_cases(distributions):
    """Each distribution with a polynomial profit, and with a square-root one where that is real."""
    for name in distributions:
        yield pytest.param(name, True, id=f'polynomial-{name}')
        if name != 'normal':  # the square root of a normal variable is not real
            yield pytest.param(name, False, id=f'sqrt-{name}')


@pytest.mark.parametrize(('distribution', 'polynomial'), _real_cases(DISTRIBUTIONS))
@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_expression_arguments_match_their_values(distribution, integration_method, polynomial):
    expression, value = DISTRIBUTIONS[distribution]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = logit_and_count_evaluations(_metric(value, integration_method, polynomial), {})
        actual = logit_and_count_evaluations(_metric(expression, integration_method, polynomial), PARAMETERS)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_expression_arguments_in_the_gradient_objective():
    expression, value = DISTRIBUTIONS['gamma']
    weights = np.full(RANKING_FEATURES.shape[1], 0.1)
    arguments = {
        'features': RANKING_FEATURES,
        'y_true': RANKING_Y_TRUE,
        'C': 1.0,
        'l1_ratio': 1.0,
        'fit_intercept': True,
    }
    actual = _metric(expression, 'auto', polynomial=True)._logit_objective(**arguments, **PARAMETERS)
    expected = _metric(value, 'auto', polynomial=True)._logit_objective(**arguments)
    assert actual.logit_loss(weights) == pytest.approx(expected.logit_loss(weights), rel=1e-12)
    np.testing.assert_allclose(actual.logit_gradient(weights), expected.logit_gradient(weights), rtol=1e-9)


@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_the_symbols_of_expression_arguments_are_required(integration_method):
    metric = _metric(DISTRIBUTIONS['beta'][0], integration_method, polynomial=True)
    with pytest.raises(ValueError, match='expected a value for a,'):
        metric(RANKING_Y_TRUE, RANKING_Y_SCORE, b=5.0)


def test_expression_arguments_are_checked_by_the_distribution():
    metric = _metric(DISTRIBUTIONS['beta'][0], 'auto', polynomial=True)
    with pytest.raises(ValueError, match='Invalid parameters for the Beta distribution'):
        metric(RANKING_Y_TRUE, RANKING_Y_SCORE, a=-1.0, b=5.0)


@pytest.mark.parametrize(('distribution', 'polynomial'), _real_cases(DISTRIBUTIONS))
@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_a_distribution_parameter_can_also_appear_in_the_profit(distribution, integration_method, polynomial):
    expression, value = DISTRIBUTIONS[distribution]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = logit_and_count_evaluations(
            _metric(value, integration_method, polynomial, extra=-PARAMETERS['a']), {}
        )
        actual = logit_and_count_evaluations(_metric(expression, integration_method, polynomial, extra=-A), PARAMETERS)
    assert actual == pytest.approx(expected, rel=1e-9)
