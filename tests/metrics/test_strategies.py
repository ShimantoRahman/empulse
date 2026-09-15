import itertools
import pickle

import numpy as np
import pytest
import scipy.stats as st
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, LogCost, MaxProfit, Metric, MinCost, Profit, Savings
from empulse.metrics.metric.common import PicklableLambda
from empulse.metrics.metric.strategies.cost_strategy import (
    CostBoostGradientConst,
    CostLoss,
    CostOptimalRate,
    CostOptimalThreshold,
)
from empulse.metrics.metric.strategies.max_profit_strategy.common import _convex_hull
from empulse.metrics.metric.strategies.max_profit_strategy.deterministic import (
    MaxProfitRateDeterministic,
    MaxProfitScoreDeterministic,
)
from empulse.metrics.metric.strategies.max_profit_strategy.envelope import PolynomialEnvelope
from empulse.metrics.metric.strategies.max_profit_strategy.gradient_piecewise import (
    MaxProfitBoostGradientPiecewise,
    MaxProfitLogitGradientPiecewise,
)
from empulse.metrics.metric.strategies.max_profit_strategy.max_profit_strategy import (
    _build_profit_function,
    _build_rate_function,
    _support_all_distributions,
)
from empulse.metrics.metric.strategies.max_profit_strategy.piecewise import (
    MaxProfitRatePiecewise,
    _build_max_profit_rate_piecewise,
    _build_max_profit_score_piecewise,
    _evaluate_coefficient_matrix,
    compute_piecewise_bounds,
)
from empulse.metrics.metric.strategies.max_profit_strategy.quadrature import MaxProfitScoreQuad
from empulse.metrics.metric.strategies.max_profit_strategy.quasi_monte_carlo import (
    MaxProfitScoreQuasiMonteCarlo,
    _sympy_dist_to_scipy,
    _sympy_dist_to_scipy_params,
)
from empulse.metrics.metric.strategies.savings_strategy import SavingsScore


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


class TestPartitionSupport:
    """
    Tests for the upper-envelope partition across three profit-function shapes.

    All tests share a 3-point convex hull:
      (TPR=0, FPR=0) -> (TPR=0.5, FPR=0.2) -> (TPR=1, FPR=1)
    with pi_0 = pi_1 = 0.5 and contact_cost = 2.0, on the support [0, inf) used by the
    Gamma / Exponential distributions.

    A region is labelled with the hull vertex that maximises profit on it, which is what the
    envelope decides directly. Earlier versions inferred the label from the sort order of the
    boundary values instead, which only agrees with the envelope while the optimum moves
    monotonically along the hull.
    """

    TPRS = np.array([0.0, 0.5, 1.0])
    FPRS = np.array([0.0, 0.2, 1.0])
    PI_0 = 0.5
    PI_1 = 0.5
    CONTACT_COST = 2.0
    RANDOM_VAR_BOUNDS = (0.0, np.inf)
    DIST_PARAMS: dict = {}  # ruff: ignore[mutable-class-default]

    def _coefficients(self, degree, sign):
        """Profit ``sign * clv**degree * pi_0 * F_0 - contact_cost * pi_1 * F_1`` per hull vertex."""
        coefficients = np.zeros((len(self.TPRS), degree + 1))
        coefficients[:, 0] = -self.CONTACT_COST * self.PI_1 * self.FPRS
        coefficients[:, degree] += sign * self.PI_0 * self.TPRS
        return coefficients

    def _call(self, coefficients):
        return compute_piecewise_bounds(
            PolynomialEnvelope(coefficients),
            self.TPRS,
            self.FPRS,
            self.RANDOM_VAR_BOUNDS,
            self.DIST_PARAMS,
        )

    def test_linear_profit(self):
        """
        The optimum walks up the hull as clv grows: a bigger benefit per true positive justifies
        catching more of them.

        Indifference between two vertices sits at
          clv* = contact_cost * pi_1 * (F_1 - F_3) / (pi_0 * (F_0 - F_2))
        which is 0.8 for the first hull segment and 3.2 for the second.
        """
        partition = self._call(self._coefficients(degree=1, sign=1))

        assert partition.lower_bound == pytest.approx(0.0)
        assert partition.upper_bound == np.inf
        assert partition.bounds == pytest.approx([0.0, 0.8, 3.2, np.inf])
        assert partition.tprs == pytest.approx([0.0, 0.5, 1.0])
        assert partition.fprs == pytest.approx([0.0, 0.2, 1.0])

    def test_stochastic_cost_reverses_the_traversal(self):
        """
        When the stochastic variable prices the *cost* instead of the benefit, the optimum walks
        back down the hull, so the regions are labelled in reverse hull order.

        Profit ``contact_cost * pi_0 * F_0 - clv * pi_1 * F_1`` is indifferent at
          ``clv* = contact_cost * pi_0 * (F_0 - F_2) / (pi_1 * (F_1 - F_3))``
        which is 5.0 for the first hull segment and 1.25 for the second -- decreasing, where the
        benefit-priced version increases.
        """
        coefficients = np.zeros((len(self.TPRS), 2))
        coefficients[:, 0] = self.CONTACT_COST * self.PI_0 * self.TPRS
        coefficients[:, 1] = -self.PI_1 * self.FPRS
        partition = self._call(coefficients)

        assert partition.bounds == pytest.approx([0.0, 1.25, 5.0, np.inf])
        assert partition.tprs == pytest.approx([1.0, 0.5, 0.0])
        assert partition.fprs == pytest.approx([1.0, 0.2, 0.0])

    def test_uniformly_dominated_hull_collapses_to_one_region(self):
        """
        A profit that only ever loses money is maximised by classifying nothing positive.

        ``-clv * pi_0 * F_0 - contact_cost * pi_1 * F_1`` is negative for every vertex but the
        origin on a non-negative support, so the envelope reports a single region rather than
        inventing boundaries where the curves never cross.
        """
        partition = self._call(self._coefficients(degree=1, sign=-1))

        assert partition.bounds == pytest.approx([0.0, np.inf])
        assert partition.tprs == pytest.approx([0.0])
        assert partition.fprs == pytest.approx([0.0])

    def test_quadratic_profit_uses_only_the_in_support_root(self):
        """
        A quadratic indifference equation has a +/- root pair; only the positive one is a boundary.

        ``clv**2 = contact_cost * pi_1 * (F_1 - F_3) / (pi_0 * (F_0 - F_2))`` gives 0.8 and 3.2, so
        the boundaries are their square roots. The negative roots fall outside the support and must
        be dropped, not clipped onto it: clipping used to leave degenerate zero-width regions
        whose labels then displaced the real ones.
        """
        partition = self._call(self._coefficients(degree=2, sign=1))

        assert partition.bounds == pytest.approx([0.0, np.sqrt(0.8), np.sqrt(3.2), np.inf])
        assert partition.tprs == pytest.approx([0.0, 0.5, 1.0])
        assert partition.fprs == pytest.approx([0.0, 0.2, 1.0])

    def test_regions_tile_the_support(self):
        """Every partition covers the support exactly once, with one label per region."""
        for degree in (1, 2, 3, 4, 5):
            partition = self._call(self._coefficients(degree=degree, sign=1))
            assert partition.bounds[0] == pytest.approx(0.0)
            assert partition.bounds[-1] == np.inf
            assert len(partition.bounds) == len(partition.tprs) + 1
            assert len(partition.tprs) == len(partition.fprs) == len(partition.vertex_indices)
            assert all(a <= b for a, b in itertools.pairwise(partition.bounds))

    def test_dominated_vertex_is_dropped(self):
        """
        When a vertex never wins, it contributes no region rather than an empty one.

        Here the middle vertex is dominated everywhere: the envelope should report two regions,
        not three with one of zero width.
        """
        coefficients = np.array([[0.0, 1.0], [-10.0, 1.0], [-1.0, 3.0]])
        partition = compute_piecewise_bounds(PolynomialEnvelope(coefficients), self.TPRS, self.FPRS, (0.0, np.inf), {})

        assert 1 not in partition.vertex_indices.tolist()
        assert len(partition.bounds) == len(partition.tprs) + 1


class TestNoRealCrossing:
    """
    A profit function whose indifference equation has no real root is not a failure case.

    Complex conjugate roots mean the two operating points never swap rank on the real line, so one
    of them dominates the other over the whole support and simply wins its region. Earlier versions
    raised ``ComplexRootsError`` here and told the caller to switch integration method.
    """

    @staticmethod
    def _score_function():
        """
        Build a piecewise score function for ``profit = clv**2 * pi_0 * F_0 + contact_cost * pi_1 * F_1``.

        This comes from ``tp_benefit=clv**2`` and ``fp_cost=-contact_cost``: a negative cost makes a
        false positive profitable, so classifying everything positive dominates and the indifference
        equation has no real solution on any interior hull segment.
        """
        clv_rv = sympy.stats.Gamma('clv', 2, 1)
        contact_cost = sympy.symbols('contact_cost')
        profit = _build_profit_function(
            tp_benefit=clv_rv**2,
            tn_benefit=sympy.Integer(0),
            fp_cost=-contact_cost,
            fn_cost=sympy.Integer(0),
        )
        return _build_max_profit_score_piecewise(profit, clv_rv, [contact_cost])

    @pytest.mark.parametrize(('contact_cost', 'expected'), [(0.5, 3.25), (2.0, 4.0), (10.0, 8.0)])
    def test_dominant_vertex_gives_the_exact_answer(self, contact_cost, expected):
        """
        The whole support is one region, so the answer is that vertex's profit in closed form.

        With ``fp_cost`` negative the best operating point is (FPR=1, TPR=1) everywhere, giving
        ``pi_0 * E[clv**2] + pi_1 * contact_cost`` with ``E[clv**2] = 6`` for Gamma(2, 1).
        """
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])

        assert self._score_function()(y_true, y_score, contact_cost=contact_cost) == pytest.approx(expected)

    def test_scoring_succeeds_where_it_used_to_raise(self):
        """No error path remains: the metric returns a number for every parameter value."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])

        for contact_cost in (0.0, 0.5, 2.0, 10.0, 1000.0):
            assert np.isfinite(self._score_function()(y_true, y_score, contact_cost=contact_cost))

    def test_quad_mode_agrees(self):
        """The quadrature fallback must reach the same number as the exact path."""
        from empulse.metrics.metric.strategies.max_profit_strategy import MaxProfit

        clv_rv = sympy.stats.Gamma('clv', 2, 1)
        contact_cost = sympy.symbols('contact_cost')

        strategy = MaxProfit(integration_method='quad')
        strategy.build(
            tp_benefit=clv_rv**2,
            tn_benefit=sympy.Integer(0),
            fp_cost=-contact_cost,
            fn_cost=sympy.Integer(0),
        )

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])
        assert strategy.score(y_true, y_score, contact_cost=2.0) == pytest.approx(4.0, rel=1e-6)

    def test_real_roots_still_work(self):
        """The ordinary linear case is unaffected."""
        alpha, beta = sympy.symbols('alpha beta', positive=True)
        clv_rv = sympy.stats.Gamma('clv', alpha, beta)
        contact_cost = sympy.symbols('contact_cost')
        profit = _build_profit_function(
            tp_benefit=clv_rv,
            tn_benefit=sympy.Integer(0),
            fp_cost=contact_cost,
            fn_cost=sympy.Integer(0),
        )
        score_fn = _build_max_profit_score_piecewise(profit, clv_rv, [contact_cost])

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])

        assert np.isfinite(score_fn(y_true, y_score, alpha=2.0, beta=1.0, contact_cost=2.0))


class TestHandBuiltDensity:
    """A density written out by hand, rather than one of sympy's named distributions."""

    def test_piecewise_matches_quadrature(self):
        """``ContinuousRV`` carries a Lambda and a support set where a named distribution carries
        numeric parameters, so the "every parameter is a literal" shortcut must not try to read
        them as floats. It used to, which made the exact path raise ``TypeError: Cannot convert
        expression to float`` for every hand-built density.
        """
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_score = np.clip(0.5 + 0.5 * (y_true - 0.5) + rng.normal(0, 0.3, 200), 0, 1)

        clv, d, f = sympy.symbols('clv d f')
        x = sympy.Symbol('x')
        density = sympy.stats.ContinuousRV(x, 12 * x**2 * (1 - x), set=sympy.Interval(0, 1))
        cost_matrix = CostMatrix().add_tp_benefit(density * (clv - d - f)).add_fp_cost(d + f)

        exact = Metric(cost_matrix, MaxProfit())(y_true, y_score, clv=200.0, d=10.0, f=1.0)
        numerical = Metric(cost_matrix, MaxProfit(integration_method='quad'))(y_true, y_score, clv=200.0, d=10.0, f=1.0)
        assert exact == pytest.approx(numerical, rel=1e-6)


def _brute_force_emp(y_true, y_score, benefit_of, cost, pdf, lower, upper, n_points=400_001):
    """
    Independent reference: integrate ``max_t P(t, x) * h(x)`` on a dense grid.

    Deliberately shares nothing with the package beyond the convex hull -- no piecewise regions, no
    partial moments, no root finding -- so it can only agree with the implementation by both being
    right.
    """
    positive_class_prior = float(np.mean(y_true))
    negative_class_prior = 1.0 - positive_class_prior
    tprs, fprs = _convex_hull(y_true, y_score)

    x = np.linspace(lower, upper, n_points)
    profit = positive_class_prior * np.outer(tprs, benefit_of(x)) - negative_class_prior * np.outer(
        fprs, np.full_like(x, cost)
    )
    return float(np.trapezoid(profit.max(axis=0) * pdf(x), x))


@pytest.fixture(scope='module')
def hull_data():
    """A model that is better than chance but far from perfect, so the hull has many vertices."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_score = np.clip(0.5 + 0.5 * (y_true - 0.5) + rng.normal(0, 0.3, 300), 0, 1)
    return y_true, y_score


class TestPiecewiseAgreesWithBruteForce:
    """
    The exact path must reproduce a dense-grid integration of ``max_t P(t, x) * h(x)``.

    This is the property the piecewise machinery exists to deliver, and the one that silently failed
    for even-degree profit functions while every boundary *value* was still computed correctly.
    """

    COST = sympy.Symbol('d')

    @pytest.mark.parametrize('degree', [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize('cost', [0.5, 2.0, 5.0])
    def test_uniform(self, hull_data, degree, cost):
        y_true, y_score = hull_data
        scale = sympy.Integer(10**degree)
        clv = sympy.stats.Uniform('clv', 0, 10)
        metric = Metric(CostMatrix().add_tp_benefit(clv**degree / scale).add_fp_cost(self.COST), MaxProfit())

        expected = _brute_force_emp(
            y_true, y_score, lambda x: x**degree / float(scale), cost, lambda x: np.full_like(x, 0.1), 0.0, 10.0
        )
        assert metric(y_true, y_score, d=cost) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_beta(self, hull_data, degree):
        y_true, y_score = hull_data
        gamma = sympy.stats.Beta('gamma', 6, 14)
        metric = Metric(CostMatrix().add_tp_benefit(gamma**degree).add_fp_cost(self.COST), MaxProfit())

        expected = _brute_force_emp(
            y_true, y_score, lambda x: x**degree, 0.1, lambda x: st.beta.pdf(x, 6, 14), 0.0, 1.0
        )
        assert metric(y_true, y_score, d=0.1) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_gamma(self, hull_data, degree):
        y_true, y_score = hull_data
        clv = sympy.stats.Gamma('clv', 2, 1)
        metric = Metric(CostMatrix().add_tp_benefit(clv**degree).add_fp_cost(self.COST), MaxProfit())

        expected = _brute_force_emp(
            y_true, y_score, lambda x: x**degree, 1.0, lambda x: st.gamma.pdf(x, a=2), 1e-12, 60.0, 600_001
        )
        assert metric(y_true, y_score, d=1.0) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_normal_including_negative_support(self, hull_data, degree):
        """Even powers over a support that straddles zero are where the old ordering broke down."""
        y_true, y_score = hull_data
        clv = sympy.stats.Normal('clv', 0, 1)
        metric = Metric(CostMatrix().add_tp_benefit(clv**degree).add_fp_cost(self.COST), MaxProfit())

        expected = _brute_force_emp(y_true, y_score, lambda x: x**degree, 0.2, st.norm.pdf, -12.0, 12.0, 800_001)
        assert metric(y_true, y_score, d=0.2) == pytest.approx(expected, rel=1e-5)


class TestProfitFunctionsBeyondRadicals:
    """
    Profit shapes the symbolic root solver could not reach.

    Degree five and above has no solution in radicals, and ``exp``/``log``/``sqrt`` are not
    polynomials at all. Boundaries are found numerically now, so neither is special.
    """

    COST = sympy.Symbol('d')

    def test_quintic_builds_and_scores(self, hull_data):
        """``sympy.solve`` returns an empty list here, which used to read as 'no real roots'."""
        y_true, y_score = hull_data
        clv = sympy.stats.Uniform('clv', 0, 10)
        metric = Metric(CostMatrix().add_tp_benefit(clv**5 + clv**2 + 1).add_fp_cost(self.COST), MaxProfit())

        expected = _brute_force_emp(
            y_true, y_score, lambda x: x**5 + x**2 + 1, 3.0, lambda x: np.full_like(x, 0.1), 0.0, 10.0
        )
        assert metric(y_true, y_score, d=3.0) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        ('name', 'symbolic', 'numeric'),
        [
            ('exp', lambda clv: 1 - sympy.exp(-clv / 3), lambda x: 1 - np.exp(-x / 3)),
            ('log', lambda clv: sympy.log(1 + clv), lambda x: np.log(1 + x)),
            ('sqrt', sympy.sqrt, np.sqrt),
            ('rational', lambda clv: clv / (1 + clv), lambda x: x / (1 + x)),
        ],
    )
    def test_non_polynomial_profit(self, hull_data, name, symbolic, numeric):
        """Constructing these used to raise ``PolynomialError`` out of ``is_linear_in``."""
        y_true, y_score = hull_data
        clv = sympy.stats.Uniform('clv', 0, 10)
        metric = Metric(CostMatrix().add_tp_benefit(symbolic(clv)).add_fp_cost(self.COST), MaxProfit())

        expected = _brute_force_emp(y_true, y_score, numeric, 0.2, lambda x: np.full_like(x, 0.1), 1e-12, 10.0, 800_001)
        assert metric(y_true, y_score, d=0.2) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize(
        ('name', 'symbolic'),
        [
            ('exp', lambda clv: 1 - sympy.exp(-clv / 3)),
            ('log', lambda clv: sympy.log(1 + clv)),
            ('sqrt', sympy.sqrt),
        ],
    )
    def test_non_polynomial_agrees_with_quadrature(self, hull_data, name, symbolic):
        """The segmented and global quadrature paths must reach the same number."""
        y_true, y_score = hull_data
        clv = sympy.stats.Uniform('clv', 0, 10)
        cost_matrix = CostMatrix().add_tp_benefit(symbolic(clv)).add_fp_cost(self.COST)

        segmented = Metric(cost_matrix, MaxProfit())(y_true, y_score, d=0.2)
        global_quad = Metric(cost_matrix, MaxProfit(integration_method='quad'))(y_true, y_score, d=0.2)
        assert segmented == pytest.approx(global_quad, rel=1e-6)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_optimal_rate_beyond_linear(self, hull_data, degree):
        """
        The optimal rate is exact at any degree.

        The rate does not depend on the stochastic variable at all -- only on which vertex is
        optimal -- so its expectation is always a plain probability-of-region weighting. Linearity
        was only ever needed to keep the old region ordering valid.
        """
        y_true, y_score = hull_data
        clv = sympy.stats.Uniform('clv', 0, 10)
        metric = Metric(CostMatrix().add_tp_benefit(clv**degree).add_fp_cost(self.COST), MaxProfit())

        rate = metric.optimal_rate(y_true, y_score, d=1.0)
        assert 0.0 <= rate <= 1.0

    def test_optimal_rate_matches_region_probabilities(self, hull_data):
        """Cross-check the exact rate against the regions the score path reports."""
        y_true, y_score = hull_data
        clv = sympy.stats.Uniform('clv', 0, 10)
        metric = Metric(CostMatrix().add_tp_benefit(clv**2).add_fp_cost(self.COST), MaxProfit())

        score_fn = metric.strategy._score_function
        positive_class_prior = float(np.mean(y_true))
        tprs, fprs = _convex_hull(y_true, y_score)
        coefficients = _evaluate_coefficient_matrix(
            score_fn.coefficient_eqs,
            score_fn.coefficient_fns,
            tprs,
            fprs,
            positive_class_prior,
            1.0 - positive_class_prior,
            {'d': 1.0},
        )
        partition = compute_piecewise_bounds(
            PolynomialEnvelope(coefficients), tprs, fprs, score_fn.random_var_bounds, {}
        )
        widths = np.diff(np.clip(partition.bounds, 0.0, 10.0)) / 10.0
        rates = positive_class_prior * np.array(partition.tprs) + (1 - positive_class_prior) * np.array(partition.fprs)

        assert metric.optimal_rate(y_true, y_score, d=1.0) == pytest.approx(float((rates * widths).sum()))


# Each entry pairs the SymPy distribution with the SciPy one it should map to, plus a frozen
# instance to build an independent reference from. Parameters are chosen so the distribution has a
# finite mean, which is what the metric integrates.
NEWLY_QMC_CAPABLE = [
    ('BoundedPareto', lambda: sympy.stats.BoundedPareto('w', 2.0, 1.0, 6.0), st.truncpareto(b=2.0, c=6.0, scale=1.0)),
    ('Dagum', lambda: sympy.stats.Dagum('w', 1.5, 2.0, 1.3), st.mielke(k=3.0, s=2.0, scale=1.3)),
    (
        'ExponentialPower',
        lambda: sympy.stats.ExponentialPower('w', 0.5, 1.2, 2.5),
        st.gennorm(beta=2.5, loc=0.5, scale=1.2),
    ),
    ('Frechet', lambda: sympy.stats.Frechet('w', 2.5, 1.3, 0.2), st.invweibull(c=2.5, loc=0.2, scale=1.3)),
    ('Gompertz', lambda: sympy.stats.Gompertz('w', 1.4, 0.9), st.gompertz(c=0.9, scale=1 / 1.4)),
    ('LogLogistic', lambda: sympy.stats.LogLogistic('w', 1.8, 2.4), st.fisk(c=2.4, scale=1.8)),
    ('RaisedCosine', lambda: sympy.stats.RaisedCosine('w', 1.0, 2.0), st.cosine(loc=1.0, scale=2.0 / np.pi)),
    ('Rayleigh', lambda: sympy.stats.Rayleigh('w', 1.7), st.rayleigh(scale=1.7)),
    ('Reciprocal', lambda: sympy.stats.Reciprocal('w', 1.2, 4.5), st.loguniform(1.2, 4.5)),
    ('Weibull', lambda: sympy.stats.Weibull('w', 2.0, 1.5), st.weibull_min(c=1.5, scale=2.0)),
    ('WignerSemicircle', lambda: sympy.stats.WignerSemicircle('w', 2.0), st.semicircular(scale=2.0)),
]


class TestNewlyQmcCapableDistributions:
    """Distributions that gained a SciPy quantile function, so quasi-Monte Carlo can sample them.

    Without an entry in the registry these fell through to plain Monte Carlo, which is around
    three orders of magnitude less accurate for the same budget. Weibull was the conspicuous one:
    the exact path and the optimal-rate path both already mapped it to ``scipy.stats.weibull_min``.
    """

    COST = sympy.Symbol('d')

    @pytest.mark.parametrize(('name', 'factory', 'frozen'), NEWLY_QMC_CAPABLE)
    def test_registry_covers_it(self, name, factory, frozen):
        assert _support_all_distributions([factory()])

    @pytest.mark.parametrize(('name', 'factory', 'frozen'), NEWLY_QMC_CAPABLE)
    def test_quantile_function_matches_sympy_density(self, name, factory, frozen):
        """The mapping is the pairing of a SymPy distribution with a SciPy one *and* a parameter
        translation; the two libraries disagree on parameterisation often enough that the
        translation has to be checked against the density itself rather than against documentation.
        """
        random_variable = factory()
        distribution = sympy.stats.pspace(random_variable).distribution
        parameters = [float(argument) for argument in distribution.args]

        scipy_distribution = _sympy_dist_to_scipy[type(distribution)]
        frozen = scipy_distribution(**_sympy_dist_to_scipy_params[type(distribution)](*parameters))

        variable = sympy.Symbol('t')
        sympy_density = sympy.lambdify(variable, sympy.stats.density(random_variable).pdf(variable), 'numpy')
        points = frozen.ppf([0.1, 0.3, 0.5, 0.7, 0.9])
        assert [float(sympy_density(point)) for point in points] == pytest.approx(frozen.pdf(points), rel=1e-9)

    @pytest.mark.parametrize(('name', 'factory', 'frozen'), NEWLY_QMC_CAPABLE)
    def test_quasi_monte_carlo_matches_a_dense_grid_reference(self, name, factory, frozen):
        """End to end, against a reference that shares no code with any of the backends.

        Numerical quadrature is deliberately not the reference. SymPy reports the support of a
        shifted Frechet as ``Interval(0, oo)`` rather than ``[m, oo)``, so every backend that
        integrates over the reported support evaluates the density where it is complex and returns
        nonsense. Sampling through the quantile function is unaffected.
        """
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_score = np.clip(0.5 + 0.5 * (y_true - 0.5) + rng.normal(0, 0.3, 200), 0, 1)

        cost_matrix = CostMatrix().add_tp_benefit(factory() * 20).add_fp_cost(self.COST)
        sampled = Metric(
            cost_matrix, MaxProfit(integration_method='quasi-monte-carlo', n_mc_samples_exp=16, random_state=0)
        )(y_true, y_score, d=5.0)

        expected = _brute_force_emp(
            y_true,
            y_score,
            lambda x: 20 * x,
            5.0,
            frozen.pdf,
            float(frozen.ppf(1e-12)),
            float(frozen.ppf(1 - 1e-10)),
            2_000_001,
        )
        # A 2^16-point Sobol sequence lands within a few parts in a thousand of the exact value.
        assert sampled == pytest.approx(expected, rel=5e-3)

    def test_weibull_reaches_the_same_answer_on_every_backend(self):
        """Weibull has an exact closed form as well, so all three paths can be compared at once."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_score = np.clip(0.5 + 0.5 * (y_true - 0.5) + rng.normal(0, 0.3, 200), 0, 1)

        cost_matrix = CostMatrix().add_tp_benefit(sympy.stats.Weibull('w', 2.0, 1.5) * 20).add_fp_cost(self.COST)
        exact = Metric(cost_matrix, MaxProfit())(y_true, y_score, d=5.0)
        numerical = Metric(cost_matrix, MaxProfit(integration_method='quad'))(y_true, y_score, d=5.0)
        sampled = Metric(
            cost_matrix, MaxProfit(integration_method='quasi-monte-carlo', n_mc_samples_exp=16, random_state=0)
        )(y_true, y_score, d=5.0)

        assert exact == pytest.approx(numerical, rel=1e-6)
        assert sampled == pytest.approx(exact, rel=1e-3)

    @pytest.mark.parametrize('name', ['Cauchy', 'Levy'])
    def test_distributions_without_a_mean_are_left_to_monte_carlo(self, name):
        """Cauchy and Levy have no mean for any parameters, so the expectation does not exist.

        They are deliberately absent from the registry: sampling them through quasi-Monte Carlo
        would put a confident-looking number on an undefined quantity.
        """
        random_variable = {
            'Cauchy': lambda: sympy.stats.Cauchy('w', 0.6, 1.4),
            'Levy': lambda: sympy.stats.Levy('w', 0.3, 1.1),
        }[name]()
        assert not _support_all_distributions([random_variable])

    def test_gumbel_is_left_to_monte_carlo(self):
        """Gumbel stores a ``minimum`` flag beside its parameters.

        When set it is ``scipy.stats.gumbel_l`` rather than ``gumbel_r``, which one registry entry
        cannot express, so it stays with the backend that reads the density directly.
        """
        assert not _support_all_distributions([sympy.stats.Gumbel('w', 1.3, 0.4)])


class TestAutoPrefersSampling:
    """``auto`` now reaches for quasi-Monte Carlo whenever every distribution can be sampled."""

    def test_two_variables_use_quasi_monte_carlo(self):
        """Two stochastic variables used to go to nested quadrature, which needed about 19,000
        integrand evaluations to resolve the kinks for an answer sampling reaches to ~1e-6.
        """
        clv, d, f = sympy.symbols('clv d f')
        gamma = sympy.stats.Beta('gamma', 6, 14)
        incentive = sympy.stats.Uniform('incentive', 5, 15)
        cost_matrix = (
            CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(incentive)
        )

        strategy = Metric(cost_matrix, MaxProfit()).strategy
        assert isinstance(strategy._score_function, MaxProfitScoreQuasiMonteCarlo)

    def test_falls_back_to_quadrature_when_a_distribution_cannot_be_sampled(self):
        clv, d = sympy.symbols('clv d')
        gamma = sympy.stats.Beta('gamma', 6, 14)
        cost_matrix = CostMatrix().add_tp_benefit(gamma * clv).add_fp_cost(sympy.stats.Cauchy('c', 0.6, 1.4) + d)

        strategy = Metric(cost_matrix, MaxProfit()).strategy
        assert isinstance(strategy._score_function, MaxProfitScoreQuad)

    def test_two_variable_result_still_matches_quadrature(self):
        """The faster default must not be a less correct one."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_score = np.clip(0.5 + 0.5 * (y_true - 0.5) + rng.normal(0, 0.3, 200), 0, 1)

        clv, d, f = sympy.symbols('clv d f')
        gamma = sympy.stats.Beta('gamma', 6, 14)
        incentive = sympy.stats.Uniform('incentive', 5, 15)
        cost_matrix = (
            CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(incentive)
        )
        parameters = {'clv': 200.0, 'd': 10.0, 'f': 1.0}

        automatic = Metric(cost_matrix, MaxProfit(random_state=0))(y_true, y_score, **parameters)
        numerical = Metric(cost_matrix, MaxProfit(integration_method='quad'))(y_true, y_score, **parameters)
        assert automatic == pytest.approx(numerical, rel=1e-5)
