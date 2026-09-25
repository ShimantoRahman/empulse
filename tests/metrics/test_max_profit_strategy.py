"""
The :class:`~empulse.metrics.MaxProfit` strategy: its exact piecewise integration and the training
objectives built on it.

Integration by sampling and by quadrature, and the distribution registry behind them, are in
``test_stochastic_integration.py``.
"""

import itertools

import numpy as np
import pytest
import scipy.stats as st
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import (
    CostMatrix,
    MaxProfit,
    Metric,
)
from empulse.metrics.metric.strategies.max_profit_strategy.common import _convex_hull
from empulse.metrics.metric.strategies.max_profit_strategy.envelope import PolynomialEnvelope
from empulse.metrics.metric.strategies.max_profit_strategy.max_profit_strategy import (
    _build_profit_function,
)
from empulse.metrics.metric.strategies.max_profit_strategy.piecewise import (
    _build_max_profit_score_piecewise,
    _evaluate_coefficient_matrix,
    compute_piecewise_bounds,
)

from ._helpers import brute_force_emp

# --- Exact piecewise integration ---------------------------------------------------------------------


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

    @pytest.mark.slow  # adaptive quadrature over a hand-built density
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

        expected = brute_force_emp(
            y_true, y_score, lambda x: x**degree / float(scale), cost, lambda x: np.full_like(x, 0.1), 0.0, 10.0
        )
        assert metric(y_true, y_score, d=cost) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_beta(self, hull_data, degree):
        y_true, y_score = hull_data
        gamma = sympy.stats.Beta('gamma', 6, 14)
        metric = Metric(CostMatrix().add_tp_benefit(gamma**degree).add_fp_cost(self.COST), MaxProfit())

        expected = brute_force_emp(y_true, y_score, lambda x: x**degree, 0.1, lambda x: st.beta.pdf(x, 6, 14), 0.0, 1.0)
        assert metric(y_true, y_score, d=0.1) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_gamma(self, hull_data, degree):
        y_true, y_score = hull_data
        clv = sympy.stats.Gamma('clv', 2, 1)
        metric = Metric(CostMatrix().add_tp_benefit(clv**degree).add_fp_cost(self.COST), MaxProfit())

        expected = brute_force_emp(
            y_true, y_score, lambda x: x**degree, 1.0, lambda x: st.gamma.pdf(x, a=2), 1e-12, 60.0, 600_001
        )
        assert metric(y_true, y_score, d=1.0) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize('degree', [1, 2, 3])
    def test_normal_including_negative_support(self, hull_data, degree):
        """Even powers over a support that straddles zero are where the old ordering broke down."""
        y_true, y_score = hull_data
        clv = sympy.stats.Normal('clv', 0, 1)
        metric = Metric(CostMatrix().add_tp_benefit(clv**degree).add_fp_cost(self.COST), MaxProfit())

        expected = brute_force_emp(y_true, y_score, lambda x: x**degree, 0.2, st.norm.pdf, -12.0, 12.0, 800_001)
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

        expected = brute_force_emp(
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

        expected = brute_force_emp(y_true, y_score, numeric, 0.2, lambda x: np.full_like(x, 0.1), 1e-12, 10.0, 800_001)
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


# --- Training objectives -----------------------------------------------------------------------------


def test_objective_max_profit_logit_deterministic():
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), MaxProfit())

    X, y = make_classification(n_samples=40, n_features=4, random_state=12)
    objective = metric._logit_objective(
        X,
        y,
        C=1.0,
        l1_ratio=0.0,
        fit_intercept=True,
        clv=5.0,
    )

    weights = np.zeros(X.shape[1], dtype=np.float64)
    value, gradient = objective(weights)

    y_score = 1.0 / (1.0 + np.exp(-(X @ weights)))
    expected_value = -metric(y, y_score, clv=5.0)

    assert pytest.approx(value, rel=1e-8, abs=1e-8) == expected_value
    assert gradient.shape == weights.shape
    assert np.all(np.isfinite(gradient))


def test_max_profit_logit_alpha_is_constant():
    """MaxProfit's logit objective no longer anneals alpha internally (see CHANGELOG): alpha is a
    plain constant that only ``set_alpha`` can change. Annealing on the logit path is now solely
    the job of an optimizer ``alpha_schedule`` (e.g. :class:`~empulse.optimizers.ExponentialSchedule`).
    """
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), MaxProfit(alpha=1.0))

    X, y = make_classification(n_samples=40, n_features=4, random_state=21)
    objective = metric._logit_objective(
        X,
        y,
        C=1.0,
        l1_ratio=0.0,
        fit_intercept=True,
        clv=5.0,
    )

    weights = np.zeros(X.shape[1], dtype=np.float64)
    assert objective.alpha == pytest.approx(1.0)  # type: ignore[attr-defined]
    _ = objective(weights)
    assert objective.alpha == pytest.approx(1.0)  # type: ignore[attr-defined]
    _ = objective(weights)
    assert objective.alpha == pytest.approx(1.0)  # type: ignore[attr-defined]

    objective.set_alpha(7.0)  # type: ignore[attr-defined]
    assert objective.alpha == pytest.approx(7.0)  # type: ignore[attr-defined]


def test_max_profit_alpha_validation():
    with pytest.raises(ValueError, match='alpha must be strictly positive'):
        MaxProfit(alpha=0.0)


def test_max_profit_cubic_profit_without_real_crossings(y_true_and_prediction):
    y_true, y_proba = y_true_and_prediction
    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = sympy.stats.Beta('gamma', alpha, beta)

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma**3 + gamma**2 * (clv - d - f))
        .add_fp_cost(d + f)
        .set_default(clv=100, d=10, f=1)
    )

    exact = Metric(cost_matrix, MaxProfit())(y_true, y_proba, alpha=6, beta=14)
    numerical = Metric(cost_matrix, MaxProfit(integration_method='quad'))(y_true, y_proba, alpha=6, beta=14)

    assert exact == pytest.approx(numerical, rel=1e-6)


def test_objective_boost_max_profit_deterministic_linear():
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), MaxProfit(alpha=1.0))

    _, y = make_classification(n_samples=40, random_state=12)
    y_score = np.linspace(-1.0, 1.0, y.shape[0])
    gradient, hessian = metric._gradient_boost_objective(y, y_score, clv=5.0)

    assert gradient.shape == y.shape
    assert hessian.shape == y.shape
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert np.all(hessian >= 0)


def test_objective_boost_max_profit_alpha_is_constant_across_fits():
    """Regression test: MaxProfit's boosting-side alpha used to anneal via an epoch counter that
    was reset only in `build()` (i.e. once, from `Metric.__init__`), so a second call resumed
    annealing from wherever the first call left off, making repeated calls on the same `Metric`
    non-reproducible. Alpha is now a plain constant, so repeated calls with identical inputs must
    return identical results.
    """
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), MaxProfit(alpha=2.0))

    _, y = make_classification(n_samples=40, random_state=23)
    y_score = np.linspace(-1.0, 1.0, y.shape[0])

    assert metric.strategy.alpha == pytest.approx(2.0)  # type: ignore[attr-defined]
    gradient_1, hessian_1 = metric._gradient_boost_objective(y, y_score, clv=5.0)
    assert metric.strategy.alpha == pytest.approx(2.0)  # type: ignore[attr-defined]
    gradient_2, hessian_2 = metric._gradient_boost_objective(y, y_score, clv=5.0)
    assert metric.strategy.alpha == pytest.approx(2.0)  # type: ignore[attr-defined]

    np.testing.assert_array_equal(gradient_1, gradient_2)
    np.testing.assert_array_equal(hessian_1, hessian_2)
