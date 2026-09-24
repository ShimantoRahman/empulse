"""Optimizers that use only the objective's value get an objective that computes the value alone."""

import numpy as np
import pytest
import sympy
import sympy.stats
from scipy.special import expit
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, MixtureComponent, MixtureMetric
from empulse.metrics.metric.strategies.max_profit_strategy.common import MaxProfitLogitValueObjective
from empulse.models import CSLogitClassifier, ProfLogitClassifier
from empulse.optimizers import GeneticAlgorithmOptimizer, LBFGSBOptimizer, MemeticOptimizer, ScipyOptimizer

CLV, D, F = sympy.symbols('clv d f')


def _churn(gamma):
    return Metric(
        CostMatrix()
        .add_tp_benefit(gamma * (CLV - D - F))
        .add_tp_benefit((1 - gamma) * -F)
        .add_fp_cost(D + F)
        .set_default(d=10, f=1, clv=200),
        MaxProfit(),
    )


DETERMINISTIC = _churn(sympy.Rational(3, 10))
BETA = _churn(sympy.stats.Beta('gamma', 6, 14))
# The gradient objective supports only deterministic metrics and distributions with positive support.
UNIFORM = _churn(sympy.stats.Uniform('gamma', 0.1, 0.5))


@pytest.fixture(scope='module')
def data():
    X, y = make_classification(n_samples=300, n_features=4, weights=[0.7], random_state=0)
    return np.hstack((np.ones((X.shape[0], 1)), X)), y


def test_optimizers_state_whether_they_use_the_gradient():
    assert GeneticAlgorithmOptimizer().requires_gradient is False
    assert MemeticOptimizer().requires_gradient is True
    assert LBFGSBOptimizer().requires_gradient is True
    assert ScipyOptimizer(method='CG').requires_gradient is True
    assert ScipyOptimizer(method='Nelder-Mead', use_jacobian=False).requires_gradient is False


@pytest.mark.parametrize('metric', [DETERMINISTIC, BETA, UNIFORM], ids=['deterministic', 'beta', 'uniform'])
def test_value_objective_is_the_negated_metric_plus_the_penalty(data, metric):
    X, y = data
    weights = np.random.default_rng(0).normal(size=X.shape[1])
    objective = metric._logit_value_objective(features=X, y_true=y, C=0.1, l1_ratio=0.5, fit_intercept=True)
    assert isinstance(objective, MaxProfitLogitValueObjective)
    expected = -metric(y, expit(X @ weights)) + objective.penalty.value(weights)
    assert objective.logit_loss(weights) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(NotImplementedError, match='only the value'):
        objective.logit_gradient(weights)


@pytest.mark.parametrize('metric', [DETERMINISTIC, BETA], ids=['deterministic', 'beta'])
def test_value_objective_matches_the_value_of_the_gradient_objective(data, metric):
    """Both objectives score the same model the same way and add the same penalty."""
    X, y = data
    weights = np.random.default_rng(1).normal(size=X.shape[1])
    arguments = {'features': X, 'y_true': y, 'C': 0.1, 'l1_ratio': 0.5, 'fit_intercept': True}
    value = metric._logit_value_objective(**arguments).logit_loss(weights)
    assert value == pytest.approx(metric._logit_objective(**arguments).logit_loss(weights), rel=1e-12)


def test_cost_metrics_keep_their_objective_for_value_only_optimizers(data):
    """Their objective already computes the value on its own, so it is used as it is."""
    X, y = data
    fp, fn = sympy.symbols('fp fn')
    metric = Metric(CostMatrix().add_fp_cost(fp).add_fn_cost(fn).set_default(fp=1, fn=5), Cost())
    arguments = {'features': X, 'y_true': y, 'C': 1.0, 'l1_ratio': 1.0, 'fit_intercept': True}
    assert type(metric._logit_value_objective(**arguments)) is type(metric._logit_objective(**arguments))


def test_mixture_combines_the_value_objectives_of_its_components(data):
    X, y = data
    weights = np.random.default_rng(2).normal(size=X.shape[1])
    mixture = MixtureMetric([MixtureComponent(0.25, DETERMINISTIC, {}), MixtureComponent(0.75, UNIFORM, {})])
    objective = mixture._logit_value_objective(features=X, y_true=y, C=np.inf, l1_ratio=0.0, fit_intercept=True)
    y_score = expit(X @ weights)
    expected = -(0.25 * DETERMINISTIC(y, y_score) + 0.75 * UNIFORM(y, y_score))
    assert objective.logit_loss(weights) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ('optimizer', 'builder'),
    [
        (GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=0), '_logit_value_objective'),
        (ScipyOptimizer(method='Nelder-Mead', use_jacobian=False, max_iter=20), '_logit_value_objective'),
        (ScipyOptimizer(method='L-BFGS-B', max_iter=2), '_logit_objective'),
    ],
    ids=['genetic', 'nelder-mead', 'l-bfgs-b'],
)
def test_models_build_the_objective_their_optimizer_needs(data, monkeypatch, optimizer, builder):
    X, y = data
    metric = _churn(sympy.Rational(3, 10))
    built = []
    for name in ('_logit_objective', '_logit_value_objective'):
        original = getattr(metric, name)

        def spy(*args, name=name, original=original, **kwargs):
            built.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(metric, name, spy)
    ProfLogitClassifier(loss=metric, optimizer=optimizer).fit(X[:, 1:], y)
    assert built == [builder]


def test_proflogit_fits_metrics_without_a_gradient(data):
    """A stochastic variable without positive support has no gradient objective, but has a value."""
    X, y = data
    optimizer = GeneticAlgorithmOptimizer(max_iter=5, population_size=10, random_state=0)
    model = ProfLogitClassifier(loss=UNIFORM, optimizer=optimizer).fit(X[:, 1:], y)
    objective = UNIFORM._logit_value_objective(features=X, y_true=y, C=1.0, l1_ratio=1.0, fit_intercept=True)
    expected = -UNIFORM(y, model.predict_proba(X[:, 1:])[:, 1]) + objective.penalty.value(model.result_.x)
    assert model.result_.fun == pytest.approx(expected, rel=1e-12)
    with pytest.raises(NotImplementedError):
        CSLogitClassifier(loss=UNIFORM, optimizer=LBFGSBOptimizer(max_iter=2)).fit(X[:, 1:], y)
