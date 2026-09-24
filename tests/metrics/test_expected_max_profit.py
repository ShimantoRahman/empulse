"""Tests of the compiled expected maximum profit of a ROC convex hull, the core of the piecewise EMP."""

import math
import warnings

import numpy as np
import pytest
import scipy.stats as st
import sympy
import sympy.stats
from scipy.integrate import quad

from empulse.metrics._cy_max_profit import Distribution, expected_max_profit
from empulse.metrics.metric.strategies.max_profit_strategy.common import _convex_hull
from empulse.metrics.metric.strategies.max_profit_strategy.max_profit_strategy import _build_profit_function
from empulse.metrics.metric.strategies.max_profit_strategy.piecewise import _build_max_profit_score_piecewise

# Targeting no one (0, 0), half at random (0.5, 0.5), or everyone (1, 1). With a profit of zero at
# (0, 0) and P(x) at (1, 1), the middle vertex earns P(x) / 2 and is never strictly best, so the
# expected maximum profit is E[max(0, P(X))].
TPR = np.array([0.0, 0.5, 1.0])
FPR = np.array([0.0, 0.5, 1.0])

PHI_1 = math.exp(-0.5) / math.sqrt(2 * math.pi)  # the standard normal density at 1


def _profit_at_everyone(*coefficients: float) -> dict[str, np.ndarray]:
    """Coefficients that give profit 0 when targeting no one and sum(c_k x**k) when targeting everyone."""
    return {
        'constant': np.zeros(len(coefficients)),
        'tpr_slope': np.array(coefficients, dtype=np.float64),
        'fpr_slope': np.zeros(len(coefficients)),
    }


@pytest.mark.parametrize(
    ('distribution', 'parameters', 'support', 'profit', 'expected'),
    [
        # E[max(0, X - 1/2)] for X ~ U(0, 1): the integral of x - 1/2 from 1/2 to 1.
        (Distribution.UNIFORM, [0.0, 1.0], (0.0, 1.0), (-0.5, 1.0), 1 / 8),
        # E[max(0, X**2 - 1/4)] for X ~ U(0, 1): the integral of x**2 - 1/4 from 1/2 to 1.
        (Distribution.UNIFORM, [0.0, 1.0], (0.0, 1.0), (-0.25, 0.0, 1.0), 1 / 6),
        # E[max(0, X)] for a standard normal X.
        (Distribution.NORMAL, [0.0, 1.0], (-np.inf, np.inf), (0.0, 1.0), 1 / math.sqrt(2 * math.pi)),
        # E[max(0, X**2 - 1)] for a standard normal X: two regions, X < -1 and X > 1, with one vertex.
        (Distribution.NORMAL, [0.0, 1.0], (-np.inf, np.inf), (-1.0, 0.0, 1.0), 2 * PHI_1),
        # E[max(0, X - 1)] for X ~ Gamma(2, 1): the integral of (x - 1) x exp(-x) from 1 is 5/e - 2/e.
        (Distribution.GAMMA, [2.0, 1.0], (0.0, np.inf), (-1.0, 1.0), 3 / math.e),
        # E[max(0, X - 2)] for X ~ Pareto(1, 3): the integral of (x - 2) 3 x**-4 from 2.
        (Distribution.PARETO, [1.0, 3.0], (1.0, np.inf), (-2.0, 1.0), 1 / 8),
        # E[max(0, X - 1/2)] for X ~ Triangular(0, 1, 1/2): the integral of (x - 1/2) 4 (1 - x) from 1/2.
        (Distribution.TRIANGULAR, [0.0, 1.0, 0.5], (0.0, 1.0), (-0.5, 1.0), 1 / 12),
        # E[max(0, X**2 - 1)] for X ~ Triangular(0, 2, 1/2): the integral of (x**2 - 1) 2 (2 - x) / 3 from 1.
        (Distribution.TRIANGULAR, [0.0, 2.0, 0.5], (0.0, 2.0), (-1.0, 0.0, 1.0), 5 / 18),
        # E[max(0, X - 1/2)] for X ~ Exp(1) is P(X > 1/2), by memorylessness.
        (Distribution.EXPONENTIAL, [1.0], (0.0, np.inf), (-0.5, 1.0), math.exp(-0.5)),
        # A chi-squared with 2 degrees of freedom is Exp(1/2), so E[max(0, X - 1)] = 2 P(X > 1).
        (Distribution.CHI_SQUARED, [2.0], (0.0, np.inf), (-1.0, 1.0), 2 * math.exp(-0.5)),
        # E[max(0, X - 1)] for X ~ LogNormal(0, 1) is E[X; X > 1] - P(X > 1) = sqrt(e) Phi(1) - 1/2.
        (Distribution.LOG_NORMAL, [0.0, 1.0], (0.0, np.inf), (-1.0, 1.0), math.sqrt(math.e) * st.norm.cdf(1) - 0.5),
        # E[max(0, X - 1/2)] for X ~ Beta(2, 2): the integral of (x - 1/2) 6 x (1 - x) from 1/2.
        (Distribution.BETA, [2.0, 2.0], (0.0, 1.0), (-0.5, 1.0), 3 / 32),
        # E[max(0, X - 1)] for X ~ Weibull(1, 2) is the integral of P(X > t) = exp(-t**2) from 1.
        (Distribution.WEIBULL, [1.0, 2.0], (0.0, np.inf), (-1.0, 1.0), math.sqrt(math.pi) / 2 * math.erfc(1)),
    ],
)
def test_expected_max_profit_of_a_known_problem(distribution, parameters, support, profit, expected):
    result = expected_max_profit(
        TPR,
        FPR,
        **_profit_at_everyone(*profit),
        lower_bound=support[0],
        upper_bound=support[1],
        distribution=distribution,
        distribution_parameters=np.array(parameters, dtype=np.float64),
    )
    assert result == pytest.approx(expected, rel=1e-12)


# Each distribution with parameters of the sympy.stats constructor, its support and its scipy twin.
DISTRIBUTIONS = [
    (Distribution.UNIFORM, [0.5, 2.0], (0.5, 2.0), st.uniform(0.5, 1.5)),
    (Distribution.NORMAL, [0.3, 0.7], (-np.inf, np.inf), st.norm(0.3, 0.7)),
    (Distribution.GAMMA, [2.5, 0.4], (0.0, np.inf), st.gamma(2.5, scale=0.4)),
    (Distribution.PARETO, [0.2, 4.5], (0.2, np.inf), st.pareto(4.5, scale=0.2)),
    (Distribution.TRIANGULAR, [0.1, 2.0, 0.6], (0.1, 2.0), st.triang(0.5 / 1.9, loc=0.1, scale=1.9)),
    (Distribution.EXPONENTIAL, [1.5], (0.0, np.inf), st.expon(scale=1 / 1.5)),
    (Distribution.CHI_SQUARED, [3.0], (0.0, np.inf), st.chi2(3.0)),
    (Distribution.LOG_NORMAL, [-0.5, 0.6], (0.0, np.inf), st.lognorm(0.6, scale=math.exp(-0.5))),
    (Distribution.BETA, [3.0, 5.0], (0.0, 1.0), st.beta(3.0, 5.0)),
    (Distribution.WEIBULL, [0.8, 1.7], (0.0, np.inf), st.weibull_min(1.7, scale=0.8)),
]


@pytest.mark.parametrize(('distribution', 'parameters', 'support', 'reference'), DISTRIBUTIONS)
@pytest.mark.parametrize('n_powers', [1, 2, 3])
def test_expected_max_profit_is_the_integral_of_the_best_profit(distribution, parameters, support, reference, n_powers):
    """
    Check against the definition, the integral of the best vertex's profit against the density.

    The hull holds the best operating point only for a profit that rises with the true positive rate
    and falls with the false positive rate, so the coefficients are drawn to keep it that way over
    the whole support: positive powers of x for the first, negative for the second. The normal's
    support includes negative x, so it gets only even powers there.
    """
    rng = np.random.default_rng(n_powers)
    for _ in range(5):
        y_true = (rng.random(300) < 0.3).astype(int)
        tpr, fpr = _convex_hull(y_true, rng.random(300) + y_true * rng.random(300))
        constant = rng.normal(size=n_powers)
        tpr_slope = np.abs(rng.normal(size=n_powers))
        fpr_slope = -np.abs(rng.normal(size=n_powers))
        if distribution == Distribution.NORMAL:
            tpr_slope[1::2] = fpr_slope[1::2] = 0.0

        def best_profit(x, tpr=tpr, fpr=fpr, constant=constant, tpr_slope=tpr_slope, fpr_slope=fpr_slope):
            powers = x ** np.arange(n_powers)
            return np.max((constant + np.outer(tpr, tpr_slope) + np.outer(fpr, fpr_slope)) @ powers)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = quad(lambda x: best_profit(x) * reference.pdf(x), *support, limit=500, epsabs=1e-11)[0]
        result = expected_max_profit(
            tpr, fpr, constant, tpr_slope, fpr_slope, *support, distribution, np.array(parameters, dtype=np.float64)
        )
        assert result == pytest.approx(expected, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize(
    'random_variable',
    [
        sympy.stats.Uniform('x', 0.2, 1.4),
        sympy.stats.Normal('x', 0.4, 0.3),
        sympy.stats.Gamma('x', 2, 0.3),
        sympy.stats.Pareto('x', 0.1, 5),
        sympy.stats.Triangular('x', 0.1, 1.5, 0.4),
        sympy.stats.Exponential('x', 2),
        sympy.stats.ChiSquared('x', 2),
        sympy.stats.LogNormal('x', -1, 0.5),
        sympy.stats.Beta('x', 6, 14),
        sympy.stats.Weibull('x', 0.5, 1.5),
    ],
    ids=lambda variable: type(sympy.stats.pspace(variable).distribution).__name__,
)
@pytest.mark.parametrize('degree', [0, 1, 2])
def test_compiled_piecewise_score_matches_its_python_integration(random_variable, degree):
    """The compiled score reproduces the per-distribution Python integration it replaces."""
    benefit, cost = sympy.symbols('benefit cost')
    tp_benefit = benefit * random_variable**degree - cost
    profit_function = _build_profit_function(tp_benefit, sympy.S.Zero, cost * (1 + random_variable), sympy.S.Zero)
    score = _build_max_profit_score_piecewise(profit_function, random_variable, [benefit, cost])
    assert score._compiled is not None

    rng = np.random.default_rng(degree)
    for _ in range(20):
        y_true = (rng.random(200) < rng.uniform(0.05, 0.95)).astype(int)
        tpr, fpr = _convex_hull(y_true, np.round(rng.random(200) + y_true * rng.random(200), 2))
        parameters = {'benefit': rng.uniform(1, 100), 'cost': rng.uniform(0, 10)}
        compiled = score._score_hull(tpr, fpr, float(y_true.mean()), dict(parameters))
        compiled_path, score._compiled = score._compiled, None
        try:
            python = score._score_hull(tpr, fpr, float(y_true.mean()), dict(parameters))
        finally:
            score._compiled = compiled_path
        assert compiled == pytest.approx(python, rel=1e-12, abs=1e-12)


def test_compiled_piecewise_score_is_available():
    """The extension has a Python fallback, so a failed build would only show up as a slowdown."""
    assert expected_max_profit is not None


def _call(**overrides):
    arguments = {
        'true_positive_rates': TPR,
        'false_positive_rates': FPR,
        **_profit_at_everyone(-0.5, 1.0),
        'lower_bound': 0.0,
        'upper_bound': 1.0,
        'distribution': Distribution.UNIFORM,
        'distribution_parameters': np.array([0.0, 1.0]),
    }
    return expected_max_profit(**{**arguments, **overrides})


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'true_positive_rates': TPR[:2]}, 'as many false as true positive rates'),
        ({'true_positive_rates': TPR[:0], 'false_positive_rates': FPR[:0]}, 'at least one vertex'),
        (_profit_at_everyone(1.0, 2.0, 3.0, 4.0), 'degree at most two'),
        ({'constant': np.zeros(1)}, 'one to three coefficients'),
        ({'distribution_parameters': np.array([1.0])}, 'takes 2 parameters, got 1'),
        ({'distribution': 99}, 'Unknown distribution'),
        (
            {'distribution': Distribution.PARETO, 'distribution_parameters': np.array([1.0, 1.0])},
            r'Pareto shape parameter \(alpha=1.0\) must be strictly greater than degree k=1',
        ),
    ],
)
def test_expected_max_profit_rejects_invalid_input(overrides, message):
    with pytest.raises(ValueError, match=message):
        _call(**overrides)
