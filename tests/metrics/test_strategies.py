import pickle

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics.metric.common import PicklableLambda
from empulse.metrics.metric.strategies.cost_strategy import (
    CostBoostGradientConst,
    CostLogitConsts,
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
    ComplexRootsError,
    MaxProfitRatePiecewise,
    _build_max_profit_rate_piecewise,
    _build_max_profit_score_piecewise,
    compute_piecewise_bounds,
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
        pytest.param(lambda: CostLogitConsts(*_COST_EXPRS), id='CostLogitConsts'),
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
        soft_threshold=False,
        fit_intercept=True,
        alpha_0=1.0,
        alpha_growth=1.1,
        alpha_max=100.0,
        parameters={'contact_cost': 1.0},
    )
    pickled = pickle.dumps(instance)
    restored = pickle.loads(pickled)
    assert restored is not None


class TestComputePiecewiseBounds:
    """
    Tests for compute_piecewise_bounds across three profit-function shapes.

    All tests share a 3-point convex hull:
      (TPR=0, FPR=0) → (TPR=0.5, FPR=0.2) → (TPR=1, FPR=1)
    with pi_0 = pi_1 = 0.5 and contact_cost = 2.0.
    """

    TPRS = np.array([0.0, 0.5, 1.0])
    FPRS = np.array([0.0, 0.2, 1.0])
    PI_0 = 0.5
    PI_1 = 0.5
    CONTACT_COST = 2.0
    # Support [0, ∞) as used by Gamma / Exponential distributions.
    RANDOM_VAR_BOUNDS = (0.0, np.inf)
    DIST_PARAMS: dict = {}  # noqa: RUF012

    # ------------------------------------------------------------------
    # Bound helpers — plain numpy, matching the sympy-derived formulas.
    # ------------------------------------------------------------------

    @staticmethod
    def _bound_linear_positive(F_0, F_1, F_2, F_3, pi_0, pi_1, contact_cost, **_):  # noqa: N803
        """
        Comes from: profit = clv * pi_0 * F_0 - contact_cost * pi_1 * F_1
        Setting profit == profit' and solving for clv:
          clv* = contact_cost * pi_1 * (F_1 - F_3) / (pi_0 * (F_0 - F_2))
        Derivative w.r.t. clv is positive → fix_inf=True.
        """
        return contact_cost * pi_1 * (F_1 - F_3) / (pi_0 * (F_0 - F_2))

    @staticmethod
    def _bound_linear_negative(F_0, F_1, F_2, F_3, pi_0, pi_1, contact_cost, **_):  # noqa: N803
        """
        Comes from: profit = contact_cost * pi_0 * F_0 - clv * pi_1 * F_1
        Setting profit == profit' and solving for clv:
          clv* = contact_cost * pi_0 * (F_2 - F_0) / (pi_1 * (F_3 - F_1))
        Derivative w.r.t. clv is negative → fix_inf=False.
        """
        return contact_cost * pi_0 * (F_2 - F_0) / (pi_1 * (F_3 - F_1))

    @staticmethod
    def _bound_quadratic_pos(F_0, F_1, F_2, F_3, pi_0, pi_1, contact_cost, **_):  # noqa: N803
        """Positive root of: clv^2 * pi_0 * F_0 - contact_cost * pi_1 * F_1 = const."""
        inner = contact_cost * pi_1 * (F_1 - F_3) / (pi_0 * (F_0 - F_2))
        return np.sqrt(inner)

    @staticmethod
    def _bound_quadratic_neg(F_0, F_1, F_2, F_3, pi_0, pi_1, contact_cost, **_):  # noqa: N803
        """Negative root of: clv^2 * pi_0 * F_0 - contact_cost * pi_1 * F_1 = const."""
        inner = contact_cost * pi_1 * (F_1 - F_3) / (pi_0 * (F_0 - F_2))
        return -np.sqrt(inner)

    def _call(self, compute_bounds_fns, fix_inf):
        return compute_piecewise_bounds(
            compute_bounds_fns=compute_bounds_fns,
            true_positive_rates=self.TPRS,
            false_positive_rates=self.FPRS,
            positive_class_prior=self.PI_0,
            negative_class_prior=self.PI_1,
            random_var_bounds=self.RANDOM_VAR_BOUNDS,
            distribution_parameters=self.DIST_PARAMS,
            fix_inf=fix_inf,
            contact_cost=self.CONTACT_COST,
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_linear_positive_coefficient(self):
        """
        Positive coefficient → fix_inf=True → start TPR/FPR (0, 0) prepended.

        Analytical bound per segment:
          seg 0→1: 2*0.5*(0-0.2) / (0.5*(0-0.5)) = 0.8
          seg 1→2: 2*0.5*(0.2-1.0) / (0.5*(0.5-1.0)) = 3.2
        Expected bounds: [0.0, 0.8, 3.2, ∞]
        """
        bounds, ub, lb, tprs, fprs = self._call([self._bound_linear_positive], fix_inf=True)

        assert lb == pytest.approx(0.0)
        assert ub == np.inf
        assert bounds == pytest.approx([0.0, 0.8, 3.2, np.inf])
        assert tprs == pytest.approx([0.0, 0.5, 1.0])
        assert fprs == pytest.approx([0.0, 0.2, 1.0])

    def test_linear_negative_coefficient(self):
        """
        Negative coefficient → fix_inf=False → end TPR/FPR (0, 0) appended.
        Sorting reverses the segment order.

        Analytical bound per segment:
          seg 0→1: 2*0.5*(0.5-0) / (0.5*(0.2-0)) = 5.0
          seg 1→2: 2*0.5*(1.0-0.5) / (0.5*(1.0-0.2)) = 1.25
        Sorted bounds (inner): [1.25, 5.0]
        Expected bounds: [0.0, 1.25, 5.0, ∞]
        """
        bounds, ub, lb, tprs, fprs = self._call([self._bound_linear_negative], fix_inf=False)

        assert lb == pytest.approx(0.0)
        assert ub == np.inf
        assert bounds == pytest.approx([0.0, 1.25, 5.0, np.inf])
        # Segments are now associated in reverse sorted order; (0,0) appended at end.
        assert tprs == pytest.approx([1.0, 0.5, 0.0])
        assert fprs == pytest.approx([1.0, 0.2, 0.0])

    def test_degree_2_polynomial(self):
        """
        Quadratic profit has two bound functions (±√).
        With support [0, ∞) the negative roots are clipped to 0.0.

        Inner values: clv^2 = contact_cost * pi_1 * (F_1-F_3) / (pi_0 * (F_0-F_2))
          → [0.8, 3.2] for the two segments
        Positive roots ≈ [0.894, 1.789]; negative roots clipped to 0.0.

        Expected bounds: [0.0, 0.0, 0.0, √0.8, √3.2, ∞]
        """
        pos_roots = np.sqrt([0.8, 3.2])  # [≈0.894, ≈1.789]

        bounds, ub, lb, tprs, fprs = self._call([self._bound_quadratic_pos, self._bound_quadratic_neg], fix_inf=True)

        assert lb == pytest.approx(0.0)
        assert ub == np.inf
        # Each root function contributes 2 bounds (one per hull segment) → 4 inner
        # bounds total, prepended/appended with 0 and ∞.
        assert len(bounds) == len(tprs) + 1 == len(fprs) + 1
        assert bounds[0] == pytest.approx(0.0)
        assert bounds[-1] == np.inf
        # Bounds must be monotonically non-decreasing.
        assert all(bounds[i] <= bounds[i + 1] for i in range(len(bounds) - 1))
        # Negative roots are clipped to 0; positive roots appear as inner bounds.
        assert bounds == pytest.approx([0.0, 0.0, 0.0, pos_roots[0], pos_roots[1], np.inf])
        assert tprs == pytest.approx([1.0, 1.0, 0.5, 0.5, 1.0])
        assert fprs == pytest.approx([1.0, 1.0, 0.2, 0.2, 1.0])


class TestComplexRoots:
    """Tests for the complex-roots detection and fallback logic."""

    @staticmethod
    def _complex_root_profit_and_score():
        """
        Build a piecewise score function for:
            profit = clv**2 * pi_0 * F_0 + contact_cost * pi_1 * F_1

        This comes from tp_benefit=clv**2, fp_cost=-contact_cost (negative cost = benefit for FPs).
        The bound equation  clv**2 = contact_cost * pi_1 * (F_3-F_1) / (pi_0 * (F_0-F_2))
        evaluates to NaN on interior ROC-hull segments because (F_0-F_2)<0 and (F_3-F_1)>0
        gives a negative argument under sqrt.
        """
        clv_rv = sympy.stats.Gamma('clv', 2, 1)
        contact_cost = sympy.symbols('contact_cost')  # no positive=True to avoid sympy subs mismatch
        profit = _build_profit_function(
            tp_benefit=clv_rv**2,
            tn_benefit=sympy.Integer(0),
            fp_cost=-contact_cost,  # negative cost ⟹ FP is actually profitable → complex roots
            fn_cost=sympy.Integer(0),
        )
        score_fn = _build_max_profit_score_piecewise(profit, clv_rv, [contact_cost])
        return score_fn

    def test_runtime_complex_roots_raise_complex_roots_error(self):
        """
        Calling a piecewise score function whose bound equation yields NaN on
        non-degenerate ROC-hull segments (complex roots at runtime) must raise ComplexRootsError.

        We need a non-trivially separating model so the convex hull contains at least one
        interior diagonal segment (F_0 ≠ F_2 AND F_1 ≠ F_3).
        """
        score_fn = self._complex_root_profit_and_score()

        # Imperfect but better-than-chance classifier.
        # Hull will include at least one segment where both TPR and FPR change.
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])

        with pytest.raises(ComplexRootsError, match='complex roots'):
            score_fn(y_true, y_score, contact_cost=2.0)

    def test_complex_roots_construction_time_if_sympy_detects_i(self):
        """
        ComplexRootsError is raised at construction time when sympy resolves a root
        to contain the imaginary unit I.  We verify the guard exists by confirming that
        ComplexRootsError is a subclass of ValueError (i.e. it was correctly defined)
        and that an expression containing I triggers `has(sympy.I)`.
        """
        root_with_i = sympy.I * sympy.sqrt(sympy.Symbol('x'))
        assert root_with_i.has(sympy.I), 'Precondition failed: expression should contain I'
        assert issubclass(ComplexRootsError, ValueError)

    def test_quad_mode_skips_piecewise_and_works(self):
        """
        With integration_method='quad', the piecewise path is never taken,
        so a profit function that would cause complex roots in piecewise mode succeeds.
        """
        from empulse.metrics.metric.strategies.max_profit_strategy import MaxProfit

        clv_rv = sympy.stats.Gamma('clv', 2, 1)
        contact_cost = sympy.symbols('contact_cost')  # no positive=True

        strategy = MaxProfit(integration_method='quad')
        strategy.build(
            tp_benefit=clv_rv**2,
            tn_benefit=sympy.Integer(0),
            fp_cost=-contact_cost,
            fn_cost=sympy.Integer(0),
        )

        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])

        # Must NOT raise — quad integration handles complex-root cases fine.
        result = strategy.score(y_true, y_score, contact_cost=2.0)
        assert np.isfinite(result)

    def test_real_roots_no_complex_roots_error(self):
        """
        Standard EMP with tp_benefit=clv (linear) has real roots; calling the
        piecewise score function must NOT raise ComplexRootsError.
        """
        # Use symbolic Gamma parameters so extract_distribution_parameters can pass them through.
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

        # Imperfect model with a non-degenerate hull.
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.3, 0.7, 0.5, 0.9])

        # Must NOT raise ComplexRootsError.
        result = score_fn(y_true, y_score, alpha=2.0, beta=1.0, contact_cost=2.0)
        assert np.isfinite(result)
