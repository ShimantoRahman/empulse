"""
``BaseMetric._prepare_count_loss``: the loss of samples grouped by score, prepared once per fit.

A model whose predictions take few distinct values, like the leaves of a decision tree, scores them
from each value's numbers of positive and negative samples instead of predicting every sample.
"""

from unittest import mock

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, MinCost, MixtureComponent, MixtureMetric
from empulse.metrics.metric import metric as metric_module

clv, d, f = sympy.symbols('clv d f')
gamma = sympy.stats.Beta('gamma', 6, 14)
EMPC_MATRIX = (
    CostMatrix()
    .add_tp_benefit(gamma * (clv - d - f))
    .add_tp_benefit((1 - gamma) * -f)
    .add_fp_cost(d + f)
    .set_default(d=10, f=1)
)
TWO_VARIABLES = (
    CostMatrix().add_tp_benefit(sympy.stats.Beta('a', 2, 3) * clv).add_fp_cost(sympy.stats.Uniform('b', 1, 5))
)


def grouped(y_true, y_score):
    scores, group = np.unique(y_score, return_inverse=True)
    n_positive = np.bincount(group, weights=y_true).astype(np.int64)
    return scores, n_positive, np.bincount(group).astype(np.int64) - n_positive


@pytest.fixture
def tree_like_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 600)
    y_score = rng.random(15).astype(np.float32)[rng.integers(0, 15, 600)]  # one score per "leaf"
    return y_true, y_score


@pytest.mark.parametrize(
    ('metric', 'parameters'),
    [
        (Metric(EMPC_MATRIX, MaxProfit()), {'clv': 200.0}),
        (Metric(EMPC_MATRIX, MinCost()), {'clv': 200.0}),
        (Metric(EMPC_MATRIX, MaxProfit()), {'clv': np.linspace(100, 300, 600)}),
        (Metric(EMPC_MATRIX, MaxProfit(integration_method='quad')), {'clv': 200.0}),
        (
            Metric(EMPC_MATRIX, MaxProfit(integration_method='monte-carlo', n_mc_samples_exp=10, random_state=0)),
            {'clv': 200.0},
        ),
        (
            Metric(EMPC_MATRIX, MaxProfit(integration_method='quasi-monte-carlo', n_mc_samples_exp=10, random_state=0)),
            {'clv': 200.0},
        ),
        (
            Metric(CostMatrix().add_tp_benefit(sympy.stats.Uniform('g', 0, 1) * clv).add_fp_cost(5), MaxProfit()),
            {'clv': 50.0},
        ),
        (Metric(TWO_VARIABLES, MaxProfit()), {'clv': 100.0}),
    ],
    ids=['piecewise', 'min_cost', 'instance_dependent', 'quad', 'monte_carlo', 'quasi_monte_carlo', 'uniform', 'two'],
)
def test_count_loss_equals_the_loss_of_the_samples(metric, parameters, tree_like_predictions):
    y_true, y_score = tree_like_predictions
    count_loss = metric._prepare_count_loss(n_samples=y_true.size, **parameters)
    assert count_loss is not None
    assert count_loss(*grouped(y_true, y_score)) == metric._loss(y_true, y_score, **parameters)


@pytest.mark.parametrize(
    'metric',
    [
        Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(d), MaxProfit()),
        Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(d), Cost()),
        MixtureMetric([MixtureComponent(1.0, Metric(EMPC_MATRIX, MaxProfit()), {})]),
    ],
    ids=['deterministic_max_profit', 'cost', 'mixture'],
)
def test_count_loss_is_none_when_every_sample_is_needed(metric):
    assert metric._prepare_count_loss(clv=200.0, d=10.0) is None


def test_count_loss_resolves_its_parameters_once(tree_like_predictions):
    """Aliases, defaults and parameter checks are resolved when the loss is prepared, not per call."""
    y_true, y_score = tree_like_predictions
    metric = Metric(EMPC_MATRIX.alias('customer_lifetime_value', clv), MaxProfit())
    count_loss = metric._prepare_count_loss(n_samples=y_true.size, customer_lifetime_value=200.0)
    groups = grouped(y_true, y_score)

    with mock.patch.object(
        metric_module.Metric, '_prepare_parameters', side_effect=AssertionError('parameters resolved again')
    ):
        for _ in range(3):
            count_loss(*groups)


def test_count_loss_checks_the_parameters_when_prepared():
    metric = Metric(EMPC_MATRIX, MaxProfit())
    with pytest.raises(ValueError, match='clv'):
        metric._prepare_count_loss()
