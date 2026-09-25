"""
What every :class:`~empulse.metrics.MetricStrategy` promises, whichever one it is.

* **Pickling**: strategies and the callables they compile must survive a round trip, since
  scikit-learn pickles estimators (and so their losses) for parallel work and persistence.
* **Capabilities**: each strategy advertises what it supports, and the methods behind them exist.
* **Direction**: whether a metric is maximised, and the sign ``BaseMetric._loss`` derives from it.
* **Sign-flipped siblings**: ``Profit``, ``MinCost`` and ``EmpiricalMinCost`` are presentation only;
  they negate their parent's score and hand an estimator exactly the same loss.
"""

import pickle

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import (
    AUEPC,
    Capability,
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
    auepc_score,
    empa_score,
    empb_score,
    empc_score,
    empcs_score,
    expected_cost_loss,
    expected_cost_loss_acquisition,
    expected_cost_loss_churn,
    expected_log_cost_loss,
    expected_savings_score,
    max_profit_score,
    mpa_score,
    mpc_score,
    mpcs_score,
)
from empulse.metrics.metric._compile import PicklableLambda
from empulse.metrics.metric._direction import Direction
from empulse.metrics.metric.strategies.cost_strategy import (
    CostBoostGradientConst,
    CostLoss,
    CostOptimalRate,
    CostOptimalThreshold,
)
from empulse.metrics.metric.strategies.max_profit_strategy.deterministic import (
    MaxProfitRateDeterministic,
    MaxProfitScoreDeterministic,
)
from empulse.metrics.metric.strategies.max_profit_strategy.gradient_piecewise import (
    MaxProfitBoostGradientPiecewise,
    MaxProfitLogitGradientPiecewise,
)
from empulse.metrics.metric.strategies.max_profit_strategy.max_profit_strategy import (
    _build_profit_function,
    _build_rate_function,
)
from empulse.metrics.metric.strategies.max_profit_strategy.piecewise import (
    MaxProfitRatePiecewise,
    _build_max_profit_rate_piecewise,
    _build_max_profit_score_piecewise,
)
from empulse.metrics.metric.strategies.metric_strategy import _CAPABILITY_METHOD_NAMES, MetricStrategy
from empulse.metrics.metric.strategies.savings_strategy import SavingsScore

# --- Pickling ----------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    rng = np.random.default_rng(42)
    fn_cost = rng.random(y.size)
    fp_cost = 5
    return X, y, fn_cost, fp_cost


def _make_deterministic_profit_function():
    clv, contact_cost = sympy.symbols('clv contact_cost')
    profit_function = _build_profit_function(
        tp_benefit=clv, tn_benefit=sympy.Integer(0), fp_cost=contact_cost, fn_cost=sympy.Integer(0)
    )
    return profit_function, [clv, contact_cost]


def _make_gamma_score_function():
    clv = sympy.stats.Gamma('clv', 2, 1)
    contact_cost = sympy.symbols('contact_cost')
    profit_function = _build_profit_function(
        tp_benefit=clv, tn_benefit=sympy.Integer(0), fp_cost=contact_cost, fn_cost=sympy.Integer(0)
    )
    return _build_max_profit_score_piecewise(profit_function, clv, [contact_cost])


def _make_piecewise_score_function(random_var):
    """Build a piecewise score function for a given sympy random variable."""
    contact_cost = sympy.symbols('contact_cost')
    profit_function = _build_profit_function(
        tp_benefit=random_var, tn_benefit=sympy.Integer(0), fp_cost=contact_cost, fn_cost=sympy.Integer(0)
    )
    return _build_max_profit_score_piecewise(profit_function, random_var, [contact_cost])


def _make_piecewise_rate_function(random_var):
    """Build a piecewise rate function for a given sympy random variable."""
    contact_cost = sympy.symbols('contact_cost')
    profit_function = _build_profit_function(
        tp_benefit=random_var, tn_benefit=sympy.Integer(0), fp_cost=contact_cost, fn_cost=sympy.Integer(0)
    )
    rate_function = _build_rate_function()
    return _build_max_profit_rate_piecewise(profit_function, rate_function, random_var, [contact_cost])


def _make_max_profit_rate_piecewise():
    """Directly instantiate MaxProfitRatePiecewise (bypasses the routing in _build_max_profit_rate_piecewise)."""
    clv = sympy.stats.Gamma('clv', 2, 1)
    contact_cost = sympy.symbols('contact_cost')
    profit_function = _build_profit_function(
        tp_benefit=clv, tn_benefit=sympy.Integer(0), fp_cost=contact_cost, fn_cost=sympy.Integer(0)
    )
    return MaxProfitRatePiecewise(profit_function, _build_rate_function(), clv, [contact_cost])


_COST_EXPRS = sympy.symbols('tp tn fp fn', positive=True)


def test_picklable_lambda_constant_expression():
    """PicklableLambda wrapping a constant expression should survive a pickle round-trip and return the same value."""
    expr = sympy.Integer(42)
    pl = PicklableLambda(expr)
    result_before = pl()

    restored = pickle.loads(pickle.dumps(pl))
    assert restored() == result_before


def test_picklable_lambda_single_variable():
    """PicklableLambda with one free variable should survive pickle and evaluate correctly."""
    x = sympy.Symbol('x')
    pl = PicklableLambda(x**2 + 1)

    restored = pickle.loads(pickle.dumps(pl))
    assert restored(x=3) == pl(x=3) == 10


def test_picklable_lambda_multiple_variables():
    """PicklableLambda with several free variables should survive pickle and evaluate correctly."""
    x, y = sympy.symbols('x y')
    pl = PicklableLambda(x * y + x)

    restored = pickle.loads(pickle.dumps(pl))
    assert restored(x=2, y=3) == pl(x=2, y=3) == 8


def test_picklable_lambda_explicit_variable_order():
    """PicklableLambda respects an explicit variable list for positional calls."""
    x, y = sympy.symbols('x y')
    pl = PicklableLambda(x - y, variables=[x, y])

    restored = pickle.loads(pickle.dumps(pl))
    assert restored(5, 3) == pl(5, 3) == 2


def test_picklable_lambda_numpy_array():
    """PicklableLambda evaluates correctly on numpy arrays after pickling."""
    x = sympy.Symbol('x')
    pl = PicklableLambda(2 * x)

    arr = np.array([1.0, 2.0, 3.0])
    restored = pickle.loads(pickle.dumps(pl))
    np.testing.assert_array_equal(restored(x=arr), pl(x=arr))


def test_picklable_lambda_multiple_pickle_cycles():
    """PicklableLambda remains functional after multiple sequential pickle round-trips."""
    x = sympy.Symbol('x')
    pl = PicklableLambda(x + 1)

    for _ in range(3):
        pl = pickle.loads(pickle.dumps(pl))

    assert pl(x=4) == 5


@pytest.mark.parametrize(
    'instance_factory',
    [
        pytest.param(lambda: CostLoss(*_COST_EXPRS), id='CostLoss'),
        pytest.param(lambda: CostBoostGradientConst(*_COST_EXPRS), id='CostBoostGradientConst'),
        pytest.param(lambda: CostOptimalThreshold(*_COST_EXPRS), id='CostOptimalThreshold'),
        pytest.param(lambda: CostOptimalRate(*_COST_EXPRS), id='CostOptimalRate'),
        pytest.param(lambda: SavingsScore(*_COST_EXPRS), id='SavingsScore'),
        pytest.param(
            lambda: MaxProfitScoreDeterministic(*_make_deterministic_profit_function()),
            id='MaxProfitScoreDeterministic',
        ),
        pytest.param(
            lambda: MaxProfitRateDeterministic(*_make_deterministic_profit_function()),
            id='MaxProfitRateDeterministic',
        ),
        pytest.param(_make_gamma_score_function, id='MaxProfitScorePiecewiseGamma'),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.Pareto('clv', 1, 3)),
            id='MaxProfitScorePiecewisePareto',
        ),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.Triangular('clv', 0, 10, 5)),
            id='MaxProfitScorePiecewiseTriangular',
        ),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.Exponential('clv', 1)),
            id='MaxProfitScorePiecewiseExponential',
        ),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.ChiSquared('clv', 4)),
            id='MaxProfitScorePiecewiseChi2',
        ),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.LogNormal('clv', 0, 1)),
            id='MaxProfitScorePiecewiseLogNormal',
        ),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.Beta('clv', 2, 5)),
            id='MaxProfitScorePiecewiseBeta',
        ),
        pytest.param(
            lambda: _make_piecewise_score_function(sympy.stats.Weibull('clv', 1, 2)),
            id='MaxProfitScorePiecewiseWeibull',
        ),
        pytest.param(
            lambda: _make_piecewise_rate_function(sympy.stats.Gamma('clv', 2, 1)),
            id='ExactMaxProfitRatePiecewise',
        ),
        pytest.param(_make_max_profit_rate_piecewise, id='MaxProfitRatePiecewise'),
    ],
)
def test_strategy_classes_are_picklable(instance_factory):
    """Test that strategy helper classes using _safe_lambdify are picklable."""
    instance = instance_factory()
    pickled = pickle.dumps(instance)
    restored = pickle.loads(pickled)
    assert restored is not None


def test_max_profit_boost_gradient_piecewise_is_picklable(dataset):
    """Test that MaxProfitBoostGradientPiecewise is picklable."""
    _, y, _, _ = dataset
    score_fn = _make_gamma_score_function()
    instance = MaxProfitBoostGradientPiecewise(
        score_function=score_fn,
        y_true=y,
        parameters={'contact_cost': 1.0},
    )
    pickled = pickle.dumps(instance)
    restored = pickle.loads(pickled)
    assert restored is not None


def test_max_profit_logit_gradient_piecewise_is_picklable(dataset):
    """Test that MaxProfitLogitGradientPiecewise is picklable."""
    X, y, _, _ = dataset
    score_fn = _make_gamma_score_function()
    instance = MaxProfitLogitGradientPiecewise(
        score_function=score_fn,
        features=X,
        y_true=y,
        C=1.0,
        l1_ratio=0.0,
        fit_intercept=True,
        alpha=1.0,
        parameters={'contact_cost': 1.0},
    )
    pickled = pickle.dumps(instance)
    restored = pickle.loads(pickled)
    assert restored is not None


# --- Capabilities ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'strategy_factory, expected',
    [
        pytest.param(Cost, False, id='Cost'),
        pytest.param(Savings, False, id='Savings'),
        pytest.param(MaxProfit, True, id='MaxProfit'),
        pytest.param(LogCost, True, id='LogCost'),
        pytest.param(Profit, False, id='Profit'),
        pytest.param(MinCost, True, id='MinCost'),
    ],
)
def test_requires_dynamic_boost_objective(strategy_factory, expected):
    """MaxProfit and LogCost need per-round gradients; Cost and Savings use a precomputed constant."""
    strategy = strategy_factory()
    assert strategy.requires_dynamic_boost_objective is expected


def test_requires_dynamic_boost_objective_survives_rename():
    """Renaming a strategy (e.g. via Metric.__name__) must not change its objective capability.

    Regression test: CSBoostClassifier used to dispatch on `strategy.name`, which
    `Metric.__name__`'s setter overwrites - every prebuilt metric (empc_score, mpc_score, ...)
    renames its strategy this way, so they all silently took the wrong branch.
    """
    strategy = MaxProfit()
    strategy.name = 'my_custom_metric'
    assert strategy.requires_dynamic_boost_objective is True


class TestCapabilities:
    """`Capability` replaces `isinstance(strategy, SomeConcreteStrategy)` model-side checks."""

    @pytest.mark.parametrize(
        'strategy_factory, expected',
        [
            pytest.param(
                Cost,
                frozenset({
                    Capability.COST_ONLY_DECISION,
                    Capability.OPTIMAL_THRESHOLD,
                    Capability.OPTIMAL_RATE,
                    Capability.LOGIT_OBJECTIVE,
                    Capability.PRECOMPUTED_BOOST_OBJECTIVE,
                }),
                id='Cost',
            ),
            pytest.param(
                Profit,
                frozenset({
                    Capability.COST_ONLY_DECISION,
                    Capability.OPTIMAL_THRESHOLD,
                    Capability.OPTIMAL_RATE,
                    Capability.LOGIT_OBJECTIVE,
                    Capability.PRECOMPUTED_BOOST_OBJECTIVE,
                }),
                id='Profit',
            ),
            pytest.param(
                Savings,
                frozenset({
                    Capability.COST_ONLY_DECISION,
                    Capability.OPTIMAL_THRESHOLD,
                    Capability.OPTIMAL_RATE,
                    Capability.LOGIT_OBJECTIVE,
                    Capability.PRECOMPUTED_BOOST_OBJECTIVE,
                }),
                id='Savings',
            ),
            pytest.param(
                LogCost,
                frozenset({
                    Capability.COST_ONLY_DECISION,
                    Capability.OPTIMAL_THRESHOLD,
                    Capability.OPTIMAL_RATE,
                    Capability.LOGIT_OBJECTIVE,
                    Capability.BOOST_OBJECTIVE,
                }),
                id='LogCost',
            ),
            pytest.param(
                MaxProfit,
                frozenset({
                    Capability.CLASS_COSTS,
                    Capability.OPTIMAL_THRESHOLD,
                    Capability.OPTIMAL_RATE,
                    Capability.LOGIT_OBJECTIVE,
                    Capability.BOOST_OBJECTIVE,
                }),
                id='MaxProfit-deterministic',
            ),
            pytest.param(
                MinCost,
                frozenset({
                    Capability.CLASS_COSTS,
                    Capability.OPTIMAL_THRESHOLD,
                    Capability.OPTIMAL_RATE,
                    Capability.LOGIT_OBJECTIVE,
                    Capability.BOOST_OBJECTIVE,
                }),
                id='MinCost-deterministic',
            ),
            pytest.param(
                EmpiricalMaxProfit,
                frozenset({Capability.OPTIMAL_THRESHOLD, Capability.OPTIMAL_RATE}),
                id='EmpiricalMaxProfit',
            ),
            pytest.param(
                EmpiricalMinCost,
                frozenset({Capability.OPTIMAL_THRESHOLD, Capability.OPTIMAL_RATE}),
                id='EmpiricalMinCost',
            ),
            pytest.param(AUEPC, frozenset(), id='AUEPC'),
        ],
    )
    def test_bundled_strategy_capability_sets(self, strategy_factory, expected):
        """Each bundled strategy's exact capability set, built with a plain deterministic cost matrix."""
        cost_matrix = CostMatrix().add_tp_benefit('a').add_fp_cost('b')
        metric = Metric(cost_matrix, strategy_factory())
        assert metric.strategy.capabilities == expected

    def test_max_profit_drops_logit_and_boost_for_a_non_positive_distribution(self):
        """MaxProfit's capabilities depend on the built score function, not just the class."""
        mu, sigma = sympy.symbols('mu sigma')
        normal = sympy.stats.Normal('normal', mu, sigma)
        cost_matrix = CostMatrix().add_tp_benefit(normal).add_fp_cost('b')
        metric = Metric(cost_matrix, MaxProfit())
        assert Capability.LOGIT_OBJECTIVE not in metric.strategy.capabilities
        assert Capability.BOOST_OBJECTIVE not in metric.strategy.capabilities
        assert Capability.CLASS_COSTS in metric.strategy.capabilities

    def test_unbuilt_max_profit_has_no_logit_or_boost_capability(self):
        """`capabilities` must not crash on an unbuilt strategy (before `Metric.__init__` calls `build`)."""
        strategy = MaxProfit()
        assert Capability.LOGIT_OBJECTIVE not in strategy.capabilities
        assert Capability.BOOST_OBJECTIVE not in strategy.capabilities

    @pytest.mark.parametrize(
        'strategy_factory',
        [Cost, Profit, Savings, LogCost, MaxProfit, MinCost, EmpiricalMaxProfit, EmpiricalMinCost, AUEPC],
    )
    def test_capability_presence_matches_method_override(self, strategy_factory):
        """
        For every bundled strategy, a capability is present iff its method is overridden.

        This is the test that would catch `_capabilities_from_overrides`'s method-name mapping
        drifting from the actual method names on `MetricStrategy`: if a capability's entry in
        `_CAPABILITY_METHOD_NAMES` is wrong or stale, this fails even though
        `test_bundled_strategy_capability_sets` above (built from the same source) would not.
        """
        cost_matrix = CostMatrix().add_tp_benefit('a').add_fp_cost('b')
        metric = Metric(cost_matrix, strategy_factory())
        strategy = metric.strategy
        for capability, method_name in _CAPABILITY_METHOD_NAMES.items():
            overridden = getattr(type(strategy), method_name) is not getattr(MetricStrategy, method_name)
            assert (capability in strategy.capabilities) is overridden, (
                f'{strategy_factory.__name__}: {capability} in capabilities is {capability in strategy.capabilities}'
                f', but {method_name} is {"" if overridden else "not "}overridden'
            )


# --- Direction and the sign of the loss --------------------------------------------------------------


Y_TRUE = np.array([0, 1, 0, 1, 1, 0, 1, 0])
Y_SCORE = np.array([0.1, 0.9, 0.3, 0.8, 0.4, 0.2, 0.7, 0.6])
CLASS_COSTS = {'fp_cost': 1.0, 'fn_cost': 5.0, 'tp_cost': 0.0, 'tn_cost': 0.0}
CHURN_PARAMS = {'clv': 200.0, 'incentive_cost': 10.0, 'contact_cost': 1.0}
CREDIT_PARAMS = {'roi': 0.2644, 'default_rate': 0.4, 'success_rate': 0.55}


@pytest.mark.parametrize(
    ('strategy', 'expected'),
    [
        (Cost(), Direction.MINIMIZE),
        (LogCost(), Direction.MINIMIZE),
        (Savings(), Direction.MAXIMIZE),
        (MaxProfit(), Direction.MAXIMIZE),
        (EmpiricalMaxProfit(), Direction.MAXIMIZE),
        (AUEPC(), Direction.MAXIMIZE),
        (Profit(), Direction.MAXIMIZE),
        (MinCost(), Direction.MINIMIZE),
        (EmpiricalMinCost(), Direction.MINIMIZE),
    ],
)
def test_strategy_direction(strategy, expected):
    """Each strategy's direction is part of its public contract."""
    assert strategy.direction is expected

    cost_matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')
    assert Metric(cost_matrix, strategy).direction is expected


@pytest.mark.parametrize(
    ('metric', 'expected'),
    [
        (expected_cost_loss, Direction.MINIMIZE),
        (expected_log_cost_loss, Direction.MINIMIZE),
        (expected_cost_loss_churn, Direction.MINIMIZE),
        (expected_cost_loss_acquisition, Direction.MINIMIZE),
        (expected_savings_score, Direction.MAXIMIZE),
        (max_profit_score, Direction.MAXIMIZE),
        (empc_score, Direction.MAXIMIZE),
        (mpc_score, Direction.MAXIMIZE),
        (empa_score, Direction.MAXIMIZE),
        (mpa_score, Direction.MAXIMIZE),
        (mpcs_score, Direction.MAXIMIZE),
        (empcs_score, Direction.MAXIMIZE),
        (empb_score, Direction.MAXIMIZE),
        (auepc_score, Direction.MAXIMIZE),
    ],
)
def test_prebuilt_metric_direction(metric, expected):
    """Every prebuilt metric object reports the direction of the strategy it was built from."""
    assert metric.direction is expected


@pytest.mark.parametrize(
    ('metric', 'parameters'),
    [
        (expected_cost_loss, CLASS_COSTS),
        (expected_log_cost_loss, CLASS_COSTS),
        (expected_savings_score, CLASS_COSTS),
        (max_profit_score, CLASS_COSTS),
        (empc_score, CHURN_PARAMS),
        (mpc_score, CHURN_PARAMS),
        (empcs_score, CREDIT_PARAMS),
    ],
)
def test_loss_is_minimized(metric, parameters):
    """``_loss`` equals the score for a MINIMIZE metric and its negation for a MAXIMIZE one."""
    score = metric(Y_TRUE, Y_SCORE, **parameters)
    loss = metric._loss(Y_TRUE, Y_SCORE, **parameters)

    sign = -1.0 if metric.direction is Direction.MAXIMIZE else 1.0
    assert loss == pytest.approx(sign * score)


def test_loss_ranks_a_maximize_metric_the_right_way():
    """A better model must produce a lower ``_loss``, even though it produces a higher score."""
    good = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.1, 0.9, 0.2])
    bad = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.9, 0.1, 0.8])

    assert expected_savings_score(Y_TRUE, good, **CLASS_COSTS) > expected_savings_score(Y_TRUE, bad, **CLASS_COSTS)
    assert expected_savings_score._loss(Y_TRUE, good, **CLASS_COSTS) < expected_savings_score._loss(
        Y_TRUE, bad, **CLASS_COSTS
    )


def test_loss_on_mixture_metric():
    """A mixture inherits ``_loss`` from ``BaseMetric``; it negates the combined score once."""
    score = empcs_score(Y_TRUE, Y_SCORE, **CREDIT_PARAMS)
    assert empcs_score._loss(Y_TRUE, Y_SCORE, **CREDIT_PARAMS) == pytest.approx(-score)


def test_loss_on_direction_inconsistent_mixture_raises():
    """``_loss`` is undefined when components disagree on direction, and must not guess a sign."""
    cost_matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')
    mixture = MixtureMetric([
        MixtureComponent(weight=0.5, metric=Metric(cost_matrix, Cost()), parameters={}),
        MixtureComponent(weight=0.5, metric=Metric(cost_matrix, Savings()), parameters={}),
    ])

    with pytest.raises(ValueError, match='inconsistent optimization directions'):
        mixture._loss(Y_TRUE, Y_SCORE, fp_cost=1.0, fn_cost=5.0)


# --- Sign-flipped siblings ---------------------------------------------------------------------------


PAIRS = [
    pytest.param(Cost, Profit, id='Cost-Profit'),
    pytest.param(MaxProfit, MinCost, id='MaxProfit-MinCost'),
    pytest.param(EmpiricalMaxProfit, EmpiricalMinCost, id='EmpiricalMaxProfit-EmpiricalMinCost'),
]


COSTS = {'fp_cost': 1.0, 'fn_cost': 5.0}


@pytest.fixture(scope='module')
def cost_matrix():
    return CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')


@pytest.fixture(scope='module')
def sibling_dataset():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.binomial(1, 0.3, n)
    y_score = np.clip(rng.normal(0.5, 0.2, n) + y * 0.25, 0.01, 0.99)
    return y, y_score


@pytest.mark.parametrize(('parent_cls', 'sibling_cls'), PAIRS)
class TestSiblingIsANegation:
    def test_score_is_negated(self, parent_cls, sibling_cls, cost_matrix, sibling_dataset):
        y, y_score = sibling_dataset
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling(y, y_score, **COSTS) == pytest.approx(-parent(y, y_score, **COSTS))

    def test_loss_is_identical(self, parent_cls, sibling_cls, cost_matrix, sibling_dataset):
        """The property models rely on: both members present the same objective to an estimator."""
        y, y_score = sibling_dataset
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling._loss(y, y_score, **COSTS) == pytest.approx(parent._loss(y, y_score, **COSTS))

    def test_directions_are_opposite(self, parent_cls, sibling_cls, cost_matrix):
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling.direction is not parent.direction

    def test_names_differ(self, parent_cls, sibling_cls, cost_matrix):
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling.__name__ != parent.__name__

    def test_optimal_rate_and_threshold_are_unchanged(self, parent_cls, sibling_cls, cost_matrix, sibling_dataset):
        """An optimal cutoff is orientation-free -- negating the metric must not move it."""
        y, y_score = sibling_dataset
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling.optimal_rate(y, y_score, **COSTS) == pytest.approx(parent.optimal_rate(y, y_score, **COSTS))
        np.testing.assert_allclose(
            sibling.optimal_threshold(y, y_score, **COSTS),
            parent.optimal_threshold(y, y_score, **COSTS),
        )

    def test_boost_objective_dispatch_is_unchanged(self, parent_cls, sibling_cls):
        """CSBoostClassifier branches on this; a sibling must route the same way as its parent."""
        assert sibling_cls().requires_dynamic_boost_objective == parent_cls().requires_dynamic_boost_objective

    def test_instance_dependent_costs(self, parent_cls, sibling_cls, cost_matrix, sibling_dataset):
        y, y_score = sibling_dataset
        rng = np.random.default_rng(1)
        costs = {'fp_cost': rng.uniform(0.5, 2.0, y.size), 'fn_cost': rng.uniform(3.0, 8.0, y.size)}
        parent = Metric(cost_matrix, parent_cls())
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling(y, y_score, **costs) == pytest.approx(-parent(y, y_score, **costs))

    def test_repr_and_latex_smoke(self, parent_cls, sibling_cls, cost_matrix):
        sibling = Metric(cost_matrix, sibling_cls())

        assert sibling_cls.__name__ in repr(sibling)
        latex = sibling._repr_latex_()
        assert isinstance(latex, str)
        assert latex.startswith('$')

    def test_mixing_a_pair_raises(self, parent_cls, sibling_cls, cost_matrix):
        """A mixture of a metric and its own negation has no well-defined direction."""
        mixture = MixtureMetric([
            MixtureComponent(weight=0.5, metric=Metric(cost_matrix, parent_cls()), parameters={}),
            MixtureComponent(weight=0.5, metric=Metric(cost_matrix, sibling_cls()), parameters={}),
        ])

        with pytest.raises(ValueError, match='inconsistent optimization directions'):
            _ = mixture.direction


def test_latex_of_a_sibling_differs_from_its_parent(cost_matrix):
    """The rendered formula is negated too, not just the computed value."""
    assert Metric(cost_matrix, Profit())._repr_latex_() != Metric(cost_matrix, Cost())._repr_latex_()


def test_empirical_min_cost_latex_uses_min(cost_matrix):
    assert '\\min_{k' in Metric(cost_matrix, EmpiricalMinCost())._repr_latex_()
    assert '\\max_{k' in Metric(cost_matrix, EmpiricalMaxProfit())._repr_latex_()


def test_min_cost_inherits_max_profit_arguments():
    """MinCost is a MaxProfit, so it takes the same integration arguments."""
    strategy = MinCost(integration_method='quad', n_mc_samples_exp=8, random_state=0, alpha=2.0)

    assert strategy.integration_method == 'quad'
    assert strategy.n_mc_samples == 2**8
    assert strategy.alpha == 2.0
    assert isinstance(strategy, MaxProfit)
