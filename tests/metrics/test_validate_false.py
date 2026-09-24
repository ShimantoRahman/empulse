"""``validate=False`` skips the label checks, but never changes a result."""

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import (
    AUEPC,
    Cost,
    CostMatrix,
    EmpiricalMaxProfit,
    EmpiricalMinCost,
    LogCost,
    MaxProfit,
    Metric,
    MinCost,
    MixtureComponent,
    MixtureMetric,
    Profit,
    Savings,
)

RNG = np.random.default_rng(0)
Y_TRUE = (RNG.random(100) < 0.3).astype(int)
Y_SCORE = np.clip(RNG.random(100) + 0.3 * Y_TRUE, 0.01, 0.99)
CLV, D = sympy.symbols('clv d')


def _metric(strategy, stochastic=False):
    benefit = sympy.stats.Beta('gamma', 6, 14) * CLV if stochastic else CLV / 4
    cost_matrix = CostMatrix().add_tp_benefit(benefit).add_fp_cost(D).add_fn_cost(CLV / 10)
    return Metric(cost_matrix.set_default(clv=100, d=5), strategy)


METRICS = {
    'cost': _metric(Cost()),
    'profit': _metric(Profit()),
    'savings': _metric(Savings()),
    'log_cost': _metric(LogCost()),
    'max_profit': _metric(MaxProfit()),
    'expected_max_profit': _metric(MaxProfit(), stochastic=True),
    'min_cost': _metric(MinCost()),
    'empirical_max_profit': _metric(EmpiricalMaxProfit()),
    'empirical_min_cost': _metric(EmpiricalMinCost()),
    'auepc': _metric(AUEPC()),
    'mixture': MixtureMetric([
        MixtureComponent(0.5, _metric(MaxProfit()), {}),
        MixtureComponent(0.5, _metric(Cost()), {}),
    ]),
}


def _outcome(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except NotImplementedError:
        return 'not implemented'


@pytest.mark.parametrize('name', METRICS)
@pytest.mark.parametrize('method', ['__call__', 'optimal_threshold', 'optimal_rate'])
def test_validate_false_gives_the_same_result(name, method):
    metric = METRICS[name]
    call = getattr(metric, method)
    for y_true, y_score in [(Y_TRUE, Y_SCORE), (list(Y_TRUE), list(Y_SCORE)), (Y_TRUE[:, None], Y_SCORE[:, None])]:
        validated = _outcome(call, y_true, y_score)
        unvalidated = _outcome(call, y_true, y_score, validate=False)
        np.testing.assert_array_equal(unvalidated, validated)


@pytest.mark.parametrize('name', METRICS)
@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_validate_false_still_rejects_scores_that_are_not_finite(name, bad):
    y_score = Y_SCORE.copy()
    y_score[3] = bad
    kind = 'NaN' if np.isnan(bad) else 'Inf'
    with pytest.raises(ValueError, match=f'should not contain {kind} values'):
        METRICS[name](Y_TRUE, y_score, validate=False)


def test_validate_false_still_rejects_inputs_of_different_lengths():
    with pytest.raises(ValueError, match='same length'):
        METRICS['max_profit'](Y_TRUE, Y_SCORE[:-1], validate=False)


def test_validate_true_still_checks_the_labels():
    with pytest.raises(ValueError, match='binary'):
        METRICS['max_profit'](Y_TRUE * 2, Y_SCORE)
