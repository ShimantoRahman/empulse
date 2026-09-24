"""Symbols declared with assumptions (e.g. positive=True) behave exactly like plain symbols."""

import warnings

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.metrics.metric._symbolic import _subs_by_name

RNG = np.random.default_rng(0)
Y_TRUE = (RNG.random(200) < 0.3).astype(int)
Y_SCORE = RNG.random(200) + 0.4 * Y_TRUE * RNG.random(200)
FEATURES = np.hstack((np.ones((200, 1)), RNG.normal(size=(200, 3))))

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


def _evaluations(metric, parameters):
    unique_scores, groups = np.unique(Y_SCORE, return_inverse=True)
    n_positive = np.bincount(groups, weights=Y_TRUE).astype(np.int64)
    n_negative = np.bincount(groups, weights=1 - Y_TRUE).astype(np.int64)
    objective = metric._logit_value_objective(
        features=FEATURES, y_true=Y_TRUE, C=1.0, l1_ratio=1.0, fit_intercept=True, **parameters
    )
    return [
        metric(Y_TRUE, Y_SCORE, **parameters),
        metric.optimal_rate(Y_TRUE, Y_SCORE, **parameters),
        metric._prepare_count_loss(**parameters)(unique_scores, n_positive, n_negative),
        objective.logit_loss(np.full(FEATURES.shape[1], 0.1)),
    ]


@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('integration_method', ['auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'])
@pytest.mark.parametrize('polynomial', [True, False], ids=['polynomial', 'sqrt'])
def test_max_profit_with_assumption_symbols_matches_plain_symbols(distribution, integration_method, polynomial):
    if distribution == 'normal' and not polynomial:
        pytest.skip('The square root of a normal variable is not real.')
    make_distribution, parameters = DISTRIBUTIONS[distribution]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plain = _evaluations(_metric(make_distribution, integration_method, polynomial), parameters)
        assumed = _evaluations(
            _metric(make_distribution, integration_method, polynomial, positive=True, real=True), parameters
        )
    assert assumed == pytest.approx(plain, rel=1e-12)


def test_max_profit_gradient_objective_with_assumption_symbols_matches_plain_symbols():
    make_distribution, parameters = DISTRIBUTIONS['gamma']
    weights = np.full(FEATURES.shape[1], 0.1)
    values = [
        _metric(make_distribution, 'auto', polynomial=True, **assumptions)
        ._logit_objective(features=FEATURES, y_true=Y_TRUE, C=1.0, l1_ratio=1.0, fit_intercept=True, **parameters)
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
