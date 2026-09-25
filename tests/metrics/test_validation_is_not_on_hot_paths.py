"""
Parameter validation must run once per fit, never once per training iteration.

Checking a parameter's domain touches array data, so it is O(n_samples) -- unlike the rest of
``Metric._prepare_parameters``, which only inspects shapes. Running it inside a training loop would
make instance-dependent training scale with the iteration count for no benefit: the values are
assembled once in ``CostSensitiveClassifier.fit`` and never change during the fit.

These tests count validating calls rather than timing them. A count is deterministic, is unaffected
by machine speed, and names the exact path that regressed -- and the counts below are the ones that
actually matter, because they are the paths measured to re-enter the metric per iteration:

============================  ==========================================
path                          calls before the opt-outs were added
============================  ==========================================
CSBoost + LogCost/MaxProfit   one per boosting round
CSBoost + CatBoost            one per evaluation period
ProfTree + a non-MaxProfit    one per candidate tree per generation
============================  ==========================================

The last section checks the other side of ``validate=False``: skipping the label checks must never
change a result, and the checks that guard against garbage input still run.
"""

import functools
from typing import Any
from unittest import mock

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
from empulse.metrics.metric import metric as metric_module
from empulse.models import CSBaggingClassifier, CSBoostClassifier, CSForestClassifier, ProfTreeClassifier

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')


@pytest.fixture(scope='module')
def training_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1500, n_features=8, random_state=0)
    return X, y, np.random.default_rng(0).uniform(100, 500, 1500)


def count_validating_calls(model: Any, X: Any, y: Any, **fit_params: Any) -> int:
    """Fit `model` and return how many times it validated its parameters."""
    calls = 0
    original = metric_module.Metric._prepare_parameters

    def counting(self: Any, *, n_samples: Any = None, validate: bool = True, **kwargs: Any) -> Any:
        nonlocal calls
        if validate:
            calls += 1
        return original(self, n_samples=n_samples, validate=validate, **kwargs)

    metric_module.Metric._prepare_parameters = counting
    try:
        model.fit(X, y, **fit_params)
    finally:
        metric_module.Metric._prepare_parameters = original
    return calls


def instance_dependent_metric(strategy: Any) -> Metric:
    return Metric(CostMatrix().add_tp_benefit('c').add_fp_cost('d'), strategy)


@pytest.mark.parametrize('strategy', [LogCost(), MaxProfit()], ids=['LogCost', 'MaxProfit'])
@pytest.mark.parametrize('n_estimators', [5, 25], ids=['5_rounds', '25_rounds'])
def test_boosting_validates_once_regardless_of_rounds(training_data, strategy, n_estimators):
    """
    These two strategies need a gradient recomputed every round, so they re-enter the metric.

    Parametrising over the round count is the point: if validation were inside the loop the count
    would track `n_estimators` instead of staying at one.
    """
    xgboost = pytest.importorskip('xgboost')
    X, y, clv = training_data
    model = CSBoostClassifier(
        xgboost.XGBClassifier(n_estimators=n_estimators, max_depth=3, n_jobs=1),
        loss=instance_dependent_metric(strategy),
    )
    assert count_validating_calls(model, X, y, c=clv, d=10.0) == 1


@pytest.mark.parametrize('n_estimators', [5, 25], ids=['5_rounds', '25_rounds'])
def test_catboost_validates_once_regardless_of_rounds(training_data, n_estimators):
    """CatBoost re-slices the cost arrays and re-enters the metric on every evaluation period."""
    catboost = pytest.importorskip('catboost')
    X, y, clv = training_data
    model = CSBoostClassifier(
        catboost.CatBoostClassifier(n_estimators=n_estimators, depth=3, verbose=False, allow_writing_files=False),
        loss=instance_dependent_metric(Cost()),
    )
    assert count_validating_calls(model, X, y, c=clv, d=10.0) == 1


@pytest.mark.parametrize(('population_size', 'generations'), [(10, 3), (20, 6)], ids=['small', 'larger'])
def test_proftree_validates_once_regardless_of_population(training_data, population_size, generations):
    """
    With a non-MaxProfit strategy ProfTree scores every candidate through the Python metric.

    With `MaxProfit` it takes a compiled fitness path instead and never reaches the metric at all,
    which is why this is parametrised on the strategy that does.
    """
    X, y, clv = training_data
    model = ProfTreeClassifier(
        max_iter=generations,
        population_size=population_size,
        random_state=42,
        loss=instance_dependent_metric(Cost()),
    )
    assert count_validating_calls(model, X, y, c=clv, d=10.0) == 1


@pytest.mark.parametrize(('population_size', 'generations'), [(10, 3), (20, 6)], ids=['small', 'larger'])
def test_proftree_prepares_a_stochastic_max_profit_once(training_data, population_size, generations):
    """
    A stochastic `MaxProfit` metric scores each candidate tree from its leaves' class counts.

    The parameters are resolved once, when that leaf-level loss is prepared, rather than once per
    candidate tree per generation as the per-sample path does.
    """
    X, y, clv = training_data
    calls = 0
    original = metric_module.Metric._prepare_parameters

    def counting(self: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    gamma = sympy.stats.Beta('gamma', 6, 14)
    loss = Metric(CostMatrix().add_tp_benefit(gamma * sympy.Symbol('c')).add_fp_cost('d'), MaxProfit())
    model = ProfTreeClassifier(max_iter=generations, population_size=population_size, random_state=42, loss=loss)
    with mock.patch.object(metric_module.Metric, '_prepare_parameters', counting):
        model.fit(X, y, c=clv, d=10.0)
    # Once to validate the fit's parameters, once to prepare the leaf-level loss.
    assert calls == 2


@pytest.mark.parametrize('n_estimators', [3, 6], ids=['3_estimators', '6_estimators'])
def test_csforest_validates_once_regardless_of_estimator_count(training_data, n_estimators):
    """The out-of-bag weighting loop scores every tree through the metric, and must not revalidate."""
    X, y, clv = training_data
    model = CSForestClassifier(
        n_estimators=n_estimators, max_depth=2, random_state=42, loss=instance_dependent_metric(Cost())
    )
    assert count_validating_calls(model, X, y, c=clv, d=10.0) == 1


@pytest.mark.parametrize('n_estimators', [3, 6], ids=['3_estimators', '6_estimators'])
def test_csbagging_validates_once_per_sub_fit(training_data, n_estimators):
    """
    Bagging fits a full cost-sensitive estimator per bag, so each bag is its own boundary.

    One validation for the outer fit plus one per sub-estimator is the correct cost: every sub-fit
    receives a different subset of the cost arrays. What must not happen is the count tracking
    anything finer than the number of sub-fits.
    """
    X, y, clv = training_data
    model = CSBaggingClassifier(n_estimators=n_estimators, random_state=42, loss=instance_dependent_metric(Cost()))
    assert count_validating_calls(model, X, y, c=clv, d=10.0) == n_estimators + 1


def test_a_constrained_symbol_is_checked_during_fit(training_data):
    """A user-declared constraint must be enforced when a model fits on the metric."""
    X, y, _ = training_data
    cost_matrix = CostMatrix().add_tp_benefit('c').add_fp_cost('d').constrain('d', 0, 1)
    model = CSBoostClassifier(loss=Metric(cost_matrix, Cost()))
    with pytest.raises(ValueError, match='d should lay between 0 and 1'):
        model.fit(X, y, c=np.ones(len(y)), d=5.0)


class TestCompilationIsNotOnHotPaths:
    """
    Compiling a cost expression to a numpy function (``sympy.lambdify``) is also O(expression
    size), not free, and used to happen on every ``Metric._evaluate_costs``/``MaxProfit.
    _evaluate_class_costs`` call rather than once at ``build()``/``__init__`` time -- the same
    shape of regression the validation tests above guard against, just for compilation instead
    of validation. These count calls to ``sympy.lambdify`` directly across repeated calls on the
    *same* metric/strategy instance, rather than through a specific model's training loop: the
    compiled functions are cached on the instance, so the count that matters is calls per
    instance, independent of which model (or how many times) ends up calling into it.
    """

    def test_metric_evaluate_costs_compiles_once(self):
        metric = instance_dependent_metric(Cost())
        with mock.patch('sympy.lambdify', wraps=sympy.lambdify) as spy:
            for _ in range(10):
                metric._evaluate_costs(c=1.0, d=2.0)
        assert spy.call_count == 0  # already compiled by Metric.__init__

    def test_metric_evaluate_costs_replace_stochastic_compiles_once(self):
        alpha, beta = sympy.symbols('alpha beta')
        gamma = sympy.stats.Beta('gamma', alpha, beta)
        metric = Metric(CostMatrix().add_tp_benefit(gamma * 10).add_fp_cost('d'), Cost())
        # The first replace_stochastic=True call builds and caches the mean-substituted
        # expressions' compiled functions (PicklableLambda's constant-expression fast path skips
        # sympy.lambdify entirely for tn_cost/fn_cost here, since neither has a term -- so the
        # exact count from this first call is an implementation detail, not what's under test).
        metric._evaluate_costs(alpha=2.0, beta=5.0, d=1.0, replace_stochastic=True)
        with mock.patch('sympy.lambdify', wraps=sympy.lambdify) as spy:
            for _ in range(10):
                metric._evaluate_costs(alpha=2.0, beta=5.0, d=1.0, replace_stochastic=True)
        assert spy.call_count == 0  # cached by the first call above, not recompiled since

    def test_max_profit_evaluate_class_costs_compiles_once(self):
        metric = instance_dependent_metric(MaxProfit())
        parameters = {'c': 1.0, 'd': 2.0}
        with mock.patch('sympy.lambdify', wraps=sympy.lambdify) as spy:
            for _ in range(10):
                metric.strategy._evaluate_class_costs(parameters)
        assert spy.call_count == 0  # already compiled by MaxProfit.build()

    def test_compiled_expressions_do_not_rederive_their_symbols(self):
        """Filtering a call's parameters to an expression's free symbols reuses the symbol names."""
        from empulse.metrics.metric._compile import _filter_parameters, _safe_lambdify, _safe_run_lambda

        c, d = sympy.symbols('c d')
        expression = 2 * c + d
        function = _safe_lambdify(expression)
        assert _safe_run_lambda(function, expression, c=1.0, d=2.0, unrelated=3.0) == 4.0
        with mock.patch.object(type(expression), 'free_symbols', new_callable=mock.PropertyMock) as free_symbols:
            for _ in range(10):
                assert _safe_run_lambda(function, expression, c=1.0, d=2.0, unrelated=3.0) == 4.0
        assert free_symbols.call_count == 0

        # An unhashable expression cannot be cached, but is still filtered.
        matrix = sympy.Matrix([c, d])
        assert _filter_parameters(matrix, {'c': 1.0, 'd': 2.0, 'unrelated': 3.0}) == {'c': 1.0, 'd': 2.0}


# --- validate=False gives the same results as validating ---------------------------------------------


_VF_RNG = np.random.default_rng(0)
VF_Y_TRUE = (_VF_RNG.random(100) < 0.3).astype(int)
VF_Y_SCORE = np.clip(_VF_RNG.random(100) + 0.3 * VF_Y_TRUE, 0.01, 0.99)
_CLV, _D = sympy.symbols('clv d')


def _vf_metric(strategy, stochastic=False):
    benefit = sympy.stats.Beta('gamma', 6, 14) * _CLV if stochastic else _CLV / 4
    cost_matrix = CostMatrix().add_tp_benefit(benefit).add_fp_cost(_D).add_fn_cost(_CLV / 10)
    return Metric(cost_matrix.set_default(clv=100, d=5), strategy)


_VF_METRIC_FACTORIES = {
    'cost': lambda: _vf_metric(Cost()),
    'profit': lambda: _vf_metric(Profit()),
    'savings': lambda: _vf_metric(Savings()),
    'log_cost': lambda: _vf_metric(LogCost()),
    'max_profit': lambda: _vf_metric(MaxProfit()),
    'expected_max_profit': lambda: _vf_metric(MaxProfit(), stochastic=True),
    'min_cost': lambda: _vf_metric(MinCost()),
    'empirical_max_profit': lambda: _vf_metric(EmpiricalMaxProfit()),
    'empirical_min_cost': lambda: _vf_metric(EmpiricalMinCost()),
    'auepc': lambda: _vf_metric(AUEPC()),
    'mixture': lambda: MixtureMetric([
        MixtureComponent(0.5, _vf_metric(MaxProfit()), {}),
        MixtureComponent(0.5, _vf_metric(Cost()), {}),
    ]),
}


@functools.cache
def _vf_metrics(name):
    return _VF_METRIC_FACTORIES[name]()


def _outcome(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except NotImplementedError:
        return 'not implemented'


@pytest.mark.parametrize('name', _VF_METRIC_FACTORIES)
@pytest.mark.parametrize('method', ['__call__', 'optimal_threshold', 'optimal_rate'])
def test_validate_false_gives_the_same_result(name, method):
    metric = _vf_metrics(name)
    call = getattr(metric, method)
    for y_true, y_score in [
        (VF_Y_TRUE, VF_Y_SCORE),
        (list(VF_Y_TRUE), list(VF_Y_SCORE)),
        (VF_Y_TRUE[:, None], VF_Y_SCORE[:, None]),
    ]:
        validated = _outcome(call, y_true, y_score)
        unvalidated = _outcome(call, y_true, y_score, validate=False)
        np.testing.assert_array_equal(unvalidated, validated)


@pytest.mark.parametrize('name', _VF_METRIC_FACTORIES)
@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_validate_false_still_rejects_scores_that_are_not_finite(name, bad):
    y_score = VF_Y_SCORE.copy()
    y_score[3] = bad
    kind = 'NaN' if np.isnan(bad) else 'Inf'
    with pytest.raises(ValueError, match=f'should not contain {kind} values'):
        _vf_metrics(name)(VF_Y_TRUE, y_score, validate=False)


def test_validate_false_still_rejects_inputs_of_different_lengths():
    with pytest.raises(ValueError, match='same length'):
        _vf_metrics('max_profit')(VF_Y_TRUE, VF_Y_SCORE[:-1], validate=False)


def test_validate_true_still_checks_the_labels():
    with pytest.raises(ValueError, match='binary'):
        _vf_metrics('max_profit')(VF_Y_TRUE * 2, VF_Y_SCORE)
