"""Symbols declared with assumptions (e.g. positive=True) behave exactly like plain symbols."""

import warnings

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.metrics.metric._symbolic import _subs_by_name

from ._helpers import RANKING_FEATURES, RANKING_Y_TRUE, logit_and_count_evaluations

DISTRIBUTIONS = {
    # The support of a uniform variable is given by its parameters.
    'uniform': (sympy.stats.Uniform, {'a': 0.2, 'b': 0.7}),
    'gamma': (sympy.stats.Gamma, {'a': 2.0, 'b': 0.1}),
    'normal': (sympy.stats.Normal, {'a': 0.3, 'b': 0.1}),
}


def _metric(distribution, integration_method, polynomial, **assumptions):
    a, b, clv, d = sympy.symbols('a b clv d', **assumptions)
    variable = distribution('v', a, b)
    benefit = variable * clv if polynomial else sympy.sqrt(variable) * clv
    cost_matrix = CostMatrix().add_tp_benefit(benefit).add_fp_cost(d).set_default(d=10, clv=100)
    return Metric(cost_matrix, MaxProfit(integration_method=integration_method, n_mc_samples_exp=10, random_state=0))


def _real_cases(distributions):
    """Each distribution with a polynomial profit, and with a square-root one where that is real."""
    for name in distributions:
        yield pytest.param(name, True, id=f'polynomial-{name}')
        if name != 'normal':  # the square root of a normal variable is not real
            yield pytest.param(name, False, id=f'sqrt-{name}')


@pytest.mark.parametrize(('distribution', 'polynomial'), _real_cases(DISTRIBUTIONS))
@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
def test_max_profit_with_assumption_symbols_matches_plain_symbols(distribution, integration_method, polynomial):
    make_distribution, parameters = DISTRIBUTIONS[distribution]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plain = logit_and_count_evaluations(_metric(make_distribution, integration_method, polynomial), parameters)
        assumed = logit_and_count_evaluations(
            _metric(make_distribution, integration_method, polynomial, positive=True, real=True), parameters
        )
    assert assumed == pytest.approx(plain, rel=1e-12)


def test_max_profit_gradient_objective_with_assumption_symbols_matches_plain_symbols():
    make_distribution, parameters = DISTRIBUTIONS['gamma']
    weights = np.full(RANKING_FEATURES.shape[1], 0.1)
    values = [
        _metric(make_distribution, 'auto', polynomial=True, **assumptions)
        ._logit_objective(
            features=RANKING_FEATURES, y_true=RANKING_Y_TRUE, C=1.0, l1_ratio=1.0, fit_intercept=True, **parameters
        )
        .logit_loss(weights)
        for assumptions in ({}, {'positive': True})
    ]
    assert values[1] == pytest.approx(values[0], rel=1e-12)


def test_subs_by_name_matches_symbols_whatever_their_assumptions():
    a, b = sympy.Symbol('a', positive=True), sympy.Symbol('b', integer=True)
    assert _subs_by_name(2 * a + b + sympy.Symbol('c'), {'a': 1, 'b': 3}) == 5 + sympy.Symbol('c')
    # Including the parameters of a random variable's distribution.
    variable = sympy.stats.Uniform('v', a, 2 * a)
    assert sympy.stats.E(_subs_by_name(variable, {'a': 2})) == 3
