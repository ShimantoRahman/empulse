"""
Stochastic cost matrices: the distributions a random variable may follow, and integrating over them
by quadrature and by (quasi-)Monte Carlo sampling.

The exact piecewise path for a single variable is in ``test_max_profit_strategy.py``.
"""

import pickle
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pytest
import scipy.stats as st
import sympy
import sympy.stats
from sympy.stats import Normal

from empulse.metrics import (
    Cost,
    CostMatrix,
    MaxProfit,
    Metric,
)
from empulse.metrics.metric.strategies.max_profit_strategy.max_profit_strategy import (
    _support_all_distributions,
)
from empulse.metrics.metric.strategies.max_profit_strategy.quadrature import (
    MaxProfitScoreQuad,
    compute_integral_multiple_quad,
)
from empulse.metrics.metric.strategies.max_profit_strategy.quasi_monte_carlo import (
    MaxProfitScoreQuasiMonteCarlo,
    _scipy_distribution,
    _sympy_dist_to_scipy,
    _sympy_dist_to_scipy_params,
)

from ._helpers import brute_force_emp

# --- Distributions -----------------------------------------------------------------------------------


class TestDistributionAdapters:
    """`_distributions.ADAPTERS`/`adapter_for` replace two 11-branch isinstance chains in
    piecewise.py (score and rate) with one table -- these round-trip every registered
    distribution through `adapter_for` and confirm the score-side table agrees with it."""

    _FACTORIES: ClassVar[dict[type, Callable[[], Any]]] = {
        sympy.stats.crv_types.UniformDistribution: lambda: sympy.stats.Uniform('x', 0, 1),
        sympy.stats.crv_types.BetaDistribution: lambda: sympy.stats.Beta('x', 2, 3),
        sympy.stats.crv_types.NormalDistribution: lambda: sympy.stats.Normal('x', 0, 1),
        sympy.stats.crv_types.LogNormalDistribution: lambda: sympy.stats.LogNormal('x', 0, 1),
        sympy.stats.crv_types.GammaDistribution: lambda: sympy.stats.Gamma('x', 2, 1),
        sympy.stats.crv_types.ExponentialDistribution: lambda: sympy.stats.Exponential('x', 1),
        sympy.stats.crv_types.ChiSquaredDistribution: lambda: sympy.stats.ChiSquared('x', 3),
        sympy.stats.crv_types.WeibullDistribution: lambda: sympy.stats.Weibull('x', 1, 2),
        sympy.stats.crv_types.ParetoDistribution: lambda: sympy.stats.Pareto('x', 1, 2),
        sympy.stats.crv_types.TriangularDistribution: lambda: sympy.stats.Triangular('x', 0, 1, 0.5),
    }

    def test_every_adapter_round_trips(self):
        from empulse.metrics.metric.strategies.max_profit_strategy._distributions import ADAPTERS, adapter_for

        assert set(self._FACTORIES) == set(ADAPTERS), 'test factory table is out of sync with ADAPTERS'
        for distribution_type, factory in self._FACTORIES.items():
            random_symbol = factory()
            found = adapter_for(random_symbol)
            assert found is ADAPTERS[distribution_type], distribution_type

    def test_unregistered_distribution_returns_none(self):
        from empulse.metrics.metric.strategies.max_profit_strategy._distributions import adapter_for

        random_symbol = sympy.stats.StudentT('x', 5)  # not in ADAPTERS
        assert adapter_for(random_symbol) is None

    def test_score_classes_and_adapters_cover_the_same_distributions(self):
        from empulse.metrics.metric.strategies.max_profit_strategy._distributions import ADAPTERS
        from empulse.metrics.metric.strategies.max_profit_strategy.piecewise import _SCORE_CLASSES

        assert _SCORE_CLASSES.keys() == ADAPTERS.keys()


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

    # A 2M-point reference grid per distribution; see `slow` in CLAUDE.md.
    @pytest.mark.slow
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

        expected = brute_force_emp(
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

    @pytest.mark.slow  # nested two-dimensional quadrature
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


# Every distribution the quasi-Monte Carlo backend can sample, with parameters away from the
# defaults (a non-zero lower bound in particular), so a mapping that only happens to hold for the
# standard form still fails.
QMC_DISTRIBUTION_FACTORIES: dict[type, Callable[[], Any]] = {
    sympy.stats.crv_types.ArcsinDistribution: lambda: sympy.stats.Arcsin('w', 2.0, 5.0),
    sympy.stats.crv_types.BetaDistribution: lambda: sympy.stats.Beta('w', 2.0, 3.0),
    sympy.stats.crv_types.BetaPrimeDistribution: lambda: sympy.stats.BetaPrime('w', 2.0, 3.0),
    sympy.stats.crv_types.ChiDistribution: lambda: sympy.stats.Chi('w', 3.0),
    sympy.stats.crv_types.ChiSquaredDistribution: lambda: sympy.stats.ChiSquared('w', 3.0),
    sympy.stats.crv_types.ExGaussianDistribution: lambda: sympy.stats.ExGaussian('w', 0.5, 1.2, 0.8),
    sympy.stats.crv_types.ExponentialDistribution: lambda: sympy.stats.Exponential('w', 1.5),
    sympy.stats.crv_types.FDistributionDistribution: lambda: sympy.stats.FDistribution('w', 5.0, 7.0),
    sympy.stats.crv_types.GammaDistribution: lambda: sympy.stats.Gamma('w', 2.0, 1.5),
    sympy.stats.crv_types.GammaInverseDistribution: lambda: sympy.stats.GammaInverse('w', 3.0, 2.0),
    sympy.stats.crv_types.LaplaceDistribution: lambda: sympy.stats.Laplace('w', 0.5, 1.3),
    sympy.stats.crv_types.LogisticDistribution: lambda: sympy.stats.Logistic('w', 0.5, 1.3),
    sympy.stats.crv_types.LogNormalDistribution: lambda: sympy.stats.LogNormal('w', 0.2, 0.6),
    sympy.stats.crv_types.LomaxDistribution: lambda: sympy.stats.Lomax('w', 3.0, 2.0),
    sympy.stats.crv_types.MaxwellDistribution: lambda: sympy.stats.Maxwell('w', 1.3),
    sympy.stats.crv_types.MoyalDistribution: lambda: sympy.stats.Moyal('w', 0.5, 1.2),
    sympy.stats.crv_types.NakagamiDistribution: lambda: sympy.stats.Nakagami('w', 1.5, 2.0),
    sympy.stats.crv_types.NormalDistribution: lambda: sympy.stats.Normal('w', 0.5, 1.2),
    sympy.stats.crv_types.ParetoDistribution: lambda: sympy.stats.Pareto('w', 1.5, 3.0),
    sympy.stats.crv_types.PowerFunctionDistribution: lambda: sympy.stats.PowerFunction('w', 2.0, 1.0, 3.0),
    sympy.stats.crv_types.StudentTDistribution: lambda: sympy.stats.StudentT('w', 5.0),
    sympy.stats.crv_types.TrapezoidalDistribution: lambda: sympy.stats.Trapezoidal('w', 1.0, 2.0, 4.0, 6.0),
    sympy.stats.crv_types.TriangularDistribution: lambda: sympy.stats.Triangular('w', 1.0, 5.0, 2.0),
    sympy.stats.crv_types.UniformDistribution: lambda: sympy.stats.Uniform('w', 1.0, 4.0),
    sympy.stats.crv_types.GaussianInverseDistribution: lambda: sympy.stats.GaussianInverse('w', 1.5, 2.0),
    **{type(sympy.stats.pspace(factory()).distribution): factory for _, factory, _ in NEWLY_QMC_CAPABLE},
}


def test_qmc_distribution_table_covers_every_registered_distribution():
    assert set(QMC_DISTRIBUTION_FACTORIES) == set(_sympy_dist_to_scipy)


@pytest.mark.parametrize(
    'factory', list(QMC_DISTRIBUTION_FACTORIES.values()), ids=[t.__name__ for t in QMC_DISTRIBUTION_FACTORIES]
)
def test_qmc_scipy_distribution_matches_sympy_density(factory):
    """The frozen SciPy distribution QMC samples from must have the SymPy distribution's density.

    Arcsin and PowerFunction used to pass their upper bound as SciPy's ``scale`` (a width), which
    stretched the support: Arcsin(2, 5) was sampled on [2, 7].
    """
    random_variable = factory()
    frozen = _scipy_distribution(random_variable)

    variable = sympy.Symbol('t', real=True)
    sympy_density = sympy.lambdify(
        variable, sympy.stats.density(random_variable).pdf(variable), modules=['scipy', 'numpy']
    )
    points = frozen.ppf([0.1, 0.3, 0.5, 0.7, 0.9])
    assert [complex(sympy_density(point)).real for point in points] == pytest.approx(frozen.pdf(points), rel=1e-9)


def test_nquad_integrates_each_variable_over_its_own_range():
    """With four or more random variables the ranges must stay paired with their own variables.

    ``nquad`` passes the variables in the order of its ranges, unlike ``dblquad``/``tplquad``, so
    reversing them there integrated ``x0`` over ``x3``'s range. The integral of ``x0`` over
    [0, 1] x [0, 2] x [0, 3] x [0, 4] is 1/2 * 2 * 3 * 4 = 12.
    """
    x = sympy.symbols('x0:4')
    result = compute_integral_multiple_quad(
        profit_integrand=x[0] + 0 * sympy.Symbol('F_0') + 0 * sympy.Symbol('F_1'),
        rate_integrand=None,
        bounds=[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        true_positive_rates=[0.5],
        false_positive_rates=[0.5],
        random_variables=list(x),
        n_random=4,
    )
    assert result == pytest.approx(12.0)


class TestDistributionWithLiteralAndSymbolicParameters:
    """A distribution mixing literal and symbolic parameters, e.g. ``Beta('g', 6, b)``.

    The piecewise backends read the distribution's parameters by position from the values the
    caller passed in, so a literal parameter (never passed in) shifted every later one and raised an
    ``IndexError``. Each result must equal the same metric with every parameter symbolic.
    """

    RANKING_Y_TRUE = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    RANKING_Y_SCORE = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.65, 0.35])

    @staticmethod
    def _metrics(factory_mixed, factory_symbolic):
        clv = sympy.Symbol('clv')

        def make(random_variable):
            return Metric(CostMatrix().add_tp_benefit(random_variable * clv).add_fp_cost(10), MaxProfit())

        return make(factory_mixed()), make(factory_symbolic())

    CASES: ClassVar[list[Any]] = [
        pytest.param(
            lambda: sympy.stats.Beta('g', 6, sympy.Symbol('b')),
            lambda: sympy.stats.Beta('g', sympy.Symbol('a'), sympy.Symbol('b')),
            {'b': 14.0},
            {'a': 6.0, 'b': 14.0},
            id='Beta-literal-first',
        ),
        pytest.param(
            lambda: sympy.stats.Gamma('g', sympy.Symbol('a'), 0.5),
            lambda: sympy.stats.Gamma('g', sympy.Symbol('a'), sympy.Symbol('b')),
            {'a': 3.0},
            {'a': 3.0, 'b': 0.5},
            id='Gamma-literal-last',
        ),
        pytest.param(
            lambda: sympy.stats.Beta('g', 6, 6),
            lambda: sympy.stats.Beta('g', sympy.Symbol('a'), sympy.Symbol('b')),
            {},
            {'a': 6.0, 'b': 6.0},
            id='Beta-equal-literals',
        ),
        pytest.param(
            lambda: sympy.stats.Normal('g', sympy.Symbol('m'), 0.3),
            lambda: sympy.stats.Normal('g', sympy.Symbol('m'), sympy.Symbol('s')),
            {'m': 1.0},
            {'m': 1.0, 's': 0.3},
            id='Normal-literal-last',
        ),
    ]

    @pytest.mark.parametrize(('mixed', 'symbolic', 'mixed_params', 'symbolic_params'), CASES)
    def test_score(self, mixed, symbolic, mixed_params, symbolic_params):
        metric_mixed, metric_symbolic = self._metrics(mixed, symbolic)
        assert metric_mixed(self.RANKING_Y_TRUE, self.RANKING_Y_SCORE, clv=200.0, **mixed_params) == pytest.approx(
            metric_symbolic(self.RANKING_Y_TRUE, self.RANKING_Y_SCORE, clv=200.0, **symbolic_params)
        )

    @pytest.mark.parametrize(('mixed', 'symbolic', 'mixed_params', 'symbolic_params'), CASES)
    def test_optimal_rate(self, mixed, symbolic, mixed_params, symbolic_params):
        metric_mixed, metric_symbolic = self._metrics(mixed, symbolic)
        assert metric_mixed.optimal_rate(
            self.RANKING_Y_TRUE, self.RANKING_Y_SCORE, clv=200.0, **mixed_params
        ) == pytest.approx(
            metric_symbolic.optimal_rate(self.RANKING_Y_TRUE, self.RANKING_Y_SCORE, clv=200.0, **symbolic_params)
        )

    @pytest.mark.parametrize(('mixed', 'symbolic', 'mixed_params', 'symbolic_params'), CASES[:3])
    def test_logit_objective(self, mixed, symbolic, mixed_params, symbolic_params):
        """The gradient objective reads the same parameters (positive distributions only)."""
        metric_mixed, metric_symbolic = self._metrics(mixed, symbolic)
        features = np.column_stack([np.ones(self.RANKING_Y_TRUE.size), self.RANKING_Y_SCORE])
        weights = np.array([0.1, -0.3])
        kwargs = {'C': 1.0, 'l1_ratio': 0.0, 'fit_intercept': True, 'clv': 200.0}
        loss_mixed, grad_mixed = metric_mixed._logit_objective(
            features, self.RANKING_Y_TRUE, **kwargs, **mixed_params
        ).logit_loss_gradient(weights)
        loss_symbolic, grad_symbolic = metric_symbolic._logit_objective(
            features, self.RANKING_Y_TRUE, **kwargs, **symbolic_params
        ).logit_loss_gradient(weights)
        assert loss_mixed == pytest.approx(loss_symbolic)
        assert grad_mixed == pytest.approx(grad_symbolic)


class TestQuasiMonteCarloIsReproducible:
    """
    ``random_state`` fixes a quasi-Monte Carlo result, however often the strategy is copied.

    ``Metric`` deep-copies its strategy, and SciPy's QMC engines spawn a child from the generator
    they are given. NumPy before 2.0 dropped a generator's seed sequence when copying it, so every
    ``Metric`` built from ``MaxProfit(random_state=0)`` sampled differently, and so did every run.
    On NumPy 2 these pass either way; the ``py311-lowest`` tox environment, on NumPy 1.x, is the one
    that can fail them.
    """

    @pytest.fixture(scope='class')
    def strategy(self):
        return MaxProfit(integration_method='quasi-monte-carlo', n_mc_samples_exp=10, random_state=0)

    @pytest.fixture(scope='class')
    def cost_matrix(self):
        a, b, clv, d = sympy.symbols('a b clv d')
        benefit = sympy.stats.Uniform('v', a, b) * clv
        return CostMatrix().add_tp_benefit(benefit).add_fp_cost(d).set_default(a=0.2, b=0.7, clv=100, d=10)

    def test_metrics_built_from_one_strategy_agree(self, strategy, cost_matrix, y_true_and_prediction):
        y_true, y_score = y_true_and_prediction
        first = Metric(cost_matrix, strategy)(y_true, y_score)
        second = Metric(cost_matrix, strategy)(y_true, y_score)
        assert first == second

    def test_a_pickled_metric_agrees_with_the_original(self, strategy, cost_matrix, y_true_and_prediction):
        y_true, y_score = y_true_and_prediction
        metric = Metric(cost_matrix, strategy)
        restored = pickle.loads(pickle.dumps(Metric(cost_matrix, strategy)))
        assert restored(y_true, y_score) == metric(y_true, y_score)


# --- Every supported distribution, end to end --------------------------------------------------------


DISTRIBUTION_GRID: list[
    tuple[
        sympy.stats.crv_types.SingleContinuousDistribution | sympy.stats.drv_types.SingleDiscreteDistribution,
        tuple[float, ...],
        ...,
    ],
] = [
    (sympy.stats.crv_types.ArcsinDistribution, (0, 1)),
    (sympy.stats.crv_types.BetaDistribution, (6, 14)),
    (sympy.stats.crv_types.BetaPrimeDistribution, (6, 14)),
    (sympy.stats.crv_types.ChiDistribution, (6,)),
    (sympy.stats.crv_types.ChiSquaredDistribution, (6,)),
    (sympy.stats.crv_types.ExponentialDistribution, (6,)),
    (sympy.stats.crv_types.FDistributionDistribution, (6, 14)),
    (sympy.stats.crv_types.GammaDistribution, (5, 0.1)),
    (sympy.stats.crv_types.GammaInverseDistribution, (6, 14)),
    (sympy.stats.crv_types.LaplaceDistribution, (6, 14)),
    (sympy.stats.crv_types.LogNormalDistribution, (1e-3, 0.1)),
    (sympy.stats.crv_types.LomaxDistribution, (6, 14)),
    (sympy.stats.crv_types.ParetoDistribution, (10, 3)),
    (sympy.stats.crv_types.MaxwellDistribution, (6,)),
    (sympy.stats.crv_types.MoyalDistribution, (6, 14)),
    (sympy.stats.crv_types.NormalDistribution, (6, 14)),
    (sympy.stats.crv_types.PowerFunctionDistribution, (6, 0, 1)),
    (sympy.stats.crv_types.StudentTDistribution, (6,)),
    (sympy.stats.crv_types.TrapezoidalDistribution, (1, 2, 3, 4)),
    (sympy.stats.crv_types.TriangularDistribution, (0, 1, 0.3)),
    (sympy.stats.crv_types.UniformDistribution, (6, 14)),
    (sympy.stats.crv_types.GaussianInverseDistribution, (6, 14)),
]


def _random_churn_cost_matrix(sympy_dist, n_params, shape):
    """A churn-like cost matrix whose accept rate follows `sympy_dist`, with parameters `param_i`."""
    clv, d, f = sympy.symbols('clv d f')
    gamma = sympy.stats.crv_types.rv('gamma', sympy_dist, tuple(sympy.symbols([f'param_{i}' for i in range(n_params)])))
    cost_matrix = CostMatrix().add_fp_cost(d + f)
    if shape == 'linear':
        return cost_matrix.add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f)
    if shape == 'neg_slope':
        return cost_matrix.add_tp_benefit(clv - (gamma * clv) / 2)
    return cost_matrix.add_tp_benefit(gamma**2 * (clv - d - f))


@pytest.mark.parametrize('shape', ['linear', 'neg_slope', 'poly'])
@pytest.mark.parametrize('sympy_dist_map', DISTRIBUTION_GRID, ids=[dist[0].__name__ for dist in DISTRIBUTION_GRID])
def test_sympy_distributions_max_profit(sympy_dist_map, shape):
    """The default integration agrees with an independent method, for the score and the optimal rate."""
    sympy_dist, params = sympy_dist_map
    cost_matrix = _random_churn_cost_matrix(sympy_dist, len(params), shape)
    parameters = {'clv': 100, 'd': 10, 'f': 1, **{f'param_{i}': value for i, value in enumerate(params)}}
    y_true = [1, 0, 1, 0, 1]
    y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]

    default = Metric(cost_matrix, MaxProfit())
    qmc = Metric(cost_matrix, MaxProfit(integration_method='quasi-monte-carlo', n_mc_samples_exp=9, random_state=12))
    checks = {
        'score': (default(y_true, y_proba, **parameters), qmc(y_true, y_proba, **parameters)),
        'rate': (default.optimal_rate(y_true, y_proba, **parameters), qmc.optimal_rate(y_true, y_proba, **parameters)),
    }
    for quantity, (result, reference) in checks.items():
        assert isinstance(result, float) and not np.isnan(result), quantity
        assert isinstance(reference, float) and not np.isnan(reference), quantity
        assert pytest.approx(result, rel=1e-1) == reference, quantity


@pytest.mark.slow
@pytest.mark.parametrize('sympy_dist_map', DISTRIBUTION_GRID, ids=[dist[0].__name__ for dist in DISTRIBUTION_GRID])
def test_cost_strategy_random_equals_mean_parametrized(y_true_and_prediction, sympy_dist_map):
    """
    Parametrized: for each sympy distribution in DISTRIBUTION_GRID verify that
    using the random variable (with numeric parameters) yields the same metric
    value as replacing the random variable by its expectation (mean).
    """
    y, y_proba = y_true_and_prediction
    sympy_dist = sympy_dist_map[0]
    params = sympy_dist_map[1]
    #
    # if sympy_dist in (sympy.stats.crv_types.BetaPrimeDistribution,):
    #     pytest.xfail("Distribution has non-lambdifiable expectation")

    # symbols used in the cost expressions
    clv, d, f = sympy.symbols('clv d f')

    # create named parameter symbols (param_0, param_1, ...)
    random_symbol_params = tuple(sympy.symbols([f'param_{i}' for i in range(len(params))]))

    # prepare substitution dicts:
    #  - for calling the metric (keyword args must be strings)
    #  - for substituting into sympy expressions (symbols -> values)
    param_values_kwargs = {f'param_{i}': params[i] for i in range(len(params))}
    param_values_subs = {random_symbol_params[i]: params[i] for i in range(len(params))}

    # build random-variable based cost matrix (gamma is the rv)
    gamma = sympy.stats.crv_types.rv('gamma', sympy_dist, random_symbol_params)
    cost_matrix_rv = (
        CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(d + f)
    )
    profit_rv = Metric(cost_matrix_rv, Cost())

    # build deterministic cost matrix using a gamma symbol
    gamma_sym = sympy.symbols('gamma')
    cost_matrix_det = (
        CostMatrix().add_tp_benefit(gamma_sym * (clv - d - f)).add_tp_benefit((1 - gamma_sym) * -f).add_fp_cost(d + f)
    )
    profit_det = Metric(cost_matrix_det, Cost())

    # numeric parameters for clv/d/f
    clv_val, d_val, f_val = 100.0, 10.0, 1.0

    # compute the mean of the random variable (E[gamma]) and evaluate to float
    fixed_means = {
        sympy.stats.crv_types.BetaPrimeDistribution: lambda params: params[0] / (params[1] - 1),
        sympy.stats.crv_types.StudentTDistribution: lambda params: 0,
        sympy.stats.crv_types.FDistributionDistribution: lambda params: params[1] / (params[1] - 2),
        sympy.stats.crv_types.GammaInverseDistribution: lambda params: params[1] / (params[0] - 1),
        sympy.stats.crv_types.LogNormalDistribution: lambda params: np.exp(params[0] + params[1] ** 2 / 2),
        sympy.stats.crv_types.LomaxDistribution: lambda params: params[1] / (params[0] - 1),
        sympy.stats.crv_types.ParetoDistribution: lambda params: (params[1] * params[0]) / (params[1] - 1),
    }
    if sympy_dist in fixed_means:
        mean_value = fixed_means[sympy_dist](params)
    else:
        mean_expr = sympy.stats.E(gamma)
        mean_value = float(sympy.N(mean_expr.subs(param_values_subs)))

    # evaluate both metrics:
    val_rv = profit_rv(y, y_proba, clv=clv_val, d=d_val, f=f_val, **param_values_kwargs)
    val_det = profit_det(y, y_proba, clv=clv_val, d=d_val, f=f_val, gamma=mean_value)

    assert pytest.approx(val_rv) == val_det


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
def test_hardcoded_distribution_params_no_error(y_true_and_prediction, integration_method):
    """Distributions with literal numeric parameters must not raise ValueError asking for them as kwargs."""
    y, y_proba = y_true_and_prediction

    # Distribution parameters are fixed numbers, not sympy symbols.
    clv, d, f = sympy.symbols('clv d f')
    gamma = Normal('gamma', 0.3, 0.1)  # mu=-0.3, sigma=0.1 are plain floats, not symbols

    cost_matrix = CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_fp_cost(d + f).set_default(clv=100, d=10, f=1)

    metric = Metric(cost_matrix, MaxProfit(integration_method=integration_method, random_state=0))
    # Must not raise ValueError("Metric expected a value for -0.3…")
    result = metric(y, y_proba)
    assert isinstance(result, float)
    assert np.isfinite(result)


@pytest.mark.parametrize('integration_method', MaxProfit.INTEGRATION_METHODS)
def test_hardcoded_distribution_params_multiple_rvs(y_true_and_prediction, integration_method):
    """Multiple stochastic variables with hardcoded params must all work without user-supplied kwargs."""
    y, y_proba = y_true_and_prediction

    l_ps, l_ph = sympy.symbols('lambda_ps lambda_ph')
    s_tp_ps = Normal('s_tp_ps', -0.2, 0.1)
    s_fp_ps = Normal('s_fp_ps', -0.8, 0.1)
    s_tp_ph = sympy.symbols('s_tp_ph')
    s_fp_ph = sympy.symbols('s_fp_ph')

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(l_ps * s_tp_ps)
        .add_tp_benefit(l_ph * s_tp_ph)
        .add_fp_benefit(l_ps * s_fp_ps)
        .add_fp_benefit(l_ph * s_fp_ph)
        .set_default(lambda_ps=0.5, lambda_ph=0.5, s_tp_ph=-0.3, s_fp_ph=-0.7)
    )

    metric = Metric(cost_matrix, MaxProfit(integration_method=integration_method, random_state=0))
    result = metric(y, y_proba)
    assert isinstance(result, float)
    assert np.isfinite(result)
