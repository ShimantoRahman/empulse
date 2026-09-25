"""
Building a :class:`~empulse.metrics.CostMatrix`: the terms, names, aliases, defaults and
constraints it accepts, and the ones it rejects when the matrix is built rather than when a metric
first computes with it.
"""

import re
import warnings

import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import (
    Cost,
    CostMatrix,
    MaxProfit,
    Metric,
    Savings,
)
from empulse.metrics.metric.cost_matrix import RESERVED_SYMBOL_NAMES

Y_TRUE = np.array([1, 0, 1, 0, 1])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.2, 0.7])


# --- Terms and symbol names --------------------------------------------------------------------------


class TestSympyReservedNamesInStringTerms:
    """A string term goes through ``sympy.sympify``, which reads some plain names as its own."""

    @pytest.mark.parametrize(
        ('term', 'culprit'),
        [
            ('E', 'E'),  # Euler's number, and a plausible name for "expense"
            ('pi', 'pi'),
            ('clv - I', 'I'),  # imaginary unit, and a plausible name for "incentive"
            ('gamma * clv', 'gamma'),  # the gamma function, not a symbol named gamma
            ('beta', 'beta'),
            ('2 * E + d', 'E'),
        ],
    )
    def test_rejects_names_sympy_claims(self, term, culprit):
        """Silently becoming a constant is the worst outcome here, so it must raise instead.

        Left unchecked, 'E' becomes 2.718..., vanishes from the metric's parameter list, and the
        metric goes on to return a plausible-looking number that no caller can influence.
        """
        with pytest.raises(ValueError, match=f"'{culprit}'"):
            CostMatrix().add_fp_cost(term)

    @pytest.mark.parametrize(
        'term',
        [
            'clv - d - f',
            'exp(clv)',  # a function *call* is fine; only bare names are rejected
            'log(clv) + sqrt(d)',
            'Min(clv, d)',
            'clv*r/(1-(1+r)**-n)',
            '2.5',
        ],
    )
    def test_accepts_ordinary_terms(self, term):
        assert CostMatrix().add_fp_cost(term).fp_cost is not None

    def test_explicit_sympy_objects_are_trusted(self):
        """Only strings are policed: an explicitly built expression means what it says."""
        matrix = CostMatrix().add_fp_cost(2 * sympy.pi)
        assert matrix.fp_cost == 2 * sympy.pi

    def test_every_add_method_validates(self):
        adders = [
            'add_tp_benefit',
            'add_tn_benefit',
            'add_fp_benefit',
            'add_fn_benefit',
            'add_tp_cost',
            'add_tn_cost',
            'add_fp_cost',
            'add_fn_cost',
        ]
        for adder in adders:
            with pytest.raises(ValueError, match="'E'"):
                getattr(CostMatrix(), adder)('E')


class TestDuplicateSymbolNames:
    """Two symbols with one name reach ``lambdify`` as two parameters of the same name."""

    def test_mixing_assumptions_raises_at_construction(self):
        """Without the check this surfaces as `SyntaxError: duplicate argument 'clv'`.

        That error points at sympy's generated source, which a caller cannot connect back to
        their own cost matrix. A string term always yields an assumption-free symbol, so mixing
        a string with an explicitly assumed Symbol of the same name is easy to do by accident.
        """
        matrix = CostMatrix().add_tp_benefit(sympy.Symbol('clv', positive=True)).add_fp_cost('clv')
        with pytest.raises(ValueError, match='share a name'):
            Metric(matrix, Cost())

    def test_error_names_both_spellings(self):
        matrix = CostMatrix().add_tp_benefit(sympy.Symbol('d', integer=True)).add_fp_cost('d')
        with pytest.raises(ValueError, match=r"sympy\.Symbol\('d', integer=True\)"):
            Metric(matrix, Cost())

    def test_consistent_assumptions_are_fine(self):
        clv = sympy.Symbol('clv', positive=True)
        matrix = CostMatrix().add_tp_benefit(clv).add_fp_cost(clv)
        assert Metric(matrix, Cost()) is not None


@pytest.mark.parametrize('reserved_name', sorted(RESERVED_SYMBOL_NAMES))
@pytest.mark.parametrize('strategy', [Cost(), Savings(), MaxProfit()], ids=['Cost', 'Savings', 'MaxProfit'])
def test_reserved_symbol_name_is_rejected_at_construction(reserved_name, strategy):
    """
    A cost-matrix symbol sharing a name reserved for internal use (e.g. ``y``, ``s``, ``F_0``)
    must be rejected at `Metric` construction time, for every strategy - not silently fused with
    the internal symbol of the same name, and not left to surface as a cryptic internal error
    only once the metric is actually called.
    """
    # Built from an explicit sympy.Symbol rather than the bare string: sympy.sympify() special-cases
    # some single-letter names (e.g. 'N' parses to the sympy.N() function, not a Symbol), which is a
    # sympy quirk unrelated to what's under test here.
    cost_matrix = CostMatrix().add_tp_benefit(sympy.Symbol(reserved_name)).add_fp_cost('b')

    with pytest.raises(ValueError, match=re.escape(f"'{reserved_name}'")):
        Metric(cost_matrix, strategy)


@pytest.mark.parametrize('reserved_name', sorted(RESERVED_SYMBOL_NAMES))
def test_reserved_alias_name_is_rejected_at_construction(reserved_name):
    """An alias sharing a name reserved for internal use must be rejected too, not just a bare symbol."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({reserved_name: 'a'})

    with pytest.raises(ValueError, match=re.escape(f"'{reserved_name}'")):
        Metric(cost_matrix, Cost())


def test_non_reserved_symbol_names_are_unaffected():
    """Sanity check: ordinary symbol names must not be rejected by the reserved-name guard."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b')
    metric = Metric(cost_matrix, Cost())

    assert metric(Y_TRUE, Y_SCORE, a=1.0, b=1.0) == pytest.approx(0.18)


# --- Aliases and defaults ----------------------------------------------------------------------------


def test_matrix_alias_wrong_types():
    clv = sympy.symbols('clv')
    cost_matrix = CostMatrix().add_tp_benefit(clv)
    with pytest.raises(ValueError, match=r'Either a dictionary or both an alias and a symbol should be provided'):
        cost_matrix.alias(None)  # type: ignore


def test_alias_target_typo_is_rejected_at_construction():
    """An alias whose target does not match any cost-matrix symbol must be rejected at construction."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'aa'})  # 'aa' is a typo for 'a'

    with pytest.raises(ValueError, match="'my_cost' -> 'aa'"):
        Metric(cost_matrix, Cost())


def test_set_default_before_alias_is_rejected_at_construction():
    """
    `set_default()` called with an alias name before that alias is registered stores the raw,
    untranslated key (see the ordering note in `CostMatrix.set_default`'s docstring) - this must
    be rejected at `Metric` construction time rather than silently dropped.
    """
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').set_default(my_cost=5.0).alias({'my_cost': 'a'})

    with pytest.raises(ValueError, match=r"default\(s\) for \['my_cost'\]"):
        Metric(cost_matrix, Cost())


def test_set_default_after_alias_is_accepted():
    """Sanity check: the documented correct ordering (alias() before set_default()) must still work."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'}).set_default(my_cost=5.0)
    metric = Metric(cost_matrix, Cost())

    # The default should behave exactly like passing a=5.0 explicitly.
    expected = Metric(CostMatrix().add_fp_cost('a').add_fn_cost('b'), Cost())(Y_TRUE, Y_SCORE, a=5.0, b=1.0)
    assert metric(Y_TRUE, Y_SCORE, b=1.0) == pytest.approx(expected)


def test_default_for_a_real_symbol_is_accepted():
    """Sanity check: a default set directly on a real symbol name (no alias involved) must work."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').set_default(a=1.0)
    metric = Metric(cost_matrix, Cost())

    assert metric(Y_TRUE, Y_SCORE, b=1.0) == pytest.approx(0.18)


# --- Warnings about the matrix itself ----------------------------------------------------------------


def test_fp_benefit_matches_cost_matrix_fp_benefit():
    """Metric.fp_benefit must delegate to CostMatrix.fp_benefit like its five siblings, not
    reimplement the negation inline."""
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())

    assert metric.fp_benefit == cost_matrix.fp_benefit
    assert metric.fp_benefit == -metric.fp_cost


def test_empty_cost_matrix_warns_at_construction():
    with pytest.warns(UserWarning, match='no cost or benefit terms'):
        Metric(CostMatrix(), Cost())


def test_cost_matrix_with_cancelling_terms_warns_at_construction():
    """A cost matrix whose terms algebraically cancel to exactly zero is just as 'empty'."""
    a = sympy.Symbol('a')
    cost_matrix = CostMatrix().add_fp_cost(a).add_fp_cost(-a)

    with pytest.warns(UserWarning, match='no cost or benefit terms'):
        Metric(cost_matrix, Cost())


def test_non_empty_cost_matrix_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Metric(CostMatrix().add_fp_cost('a').add_fn_cost('b'), Cost())


def test_constant_nonzero_cost_matrix_does_not_warn():
    """A deliberate constant (non-instance-dependent, non-symbolic) cost is not 'empty'."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Metric(CostMatrix().add_fp_cost(5), Cost())


# --- CostMatrix.constrain() --------------------------------------------------------------------------


def _bounded_metric(lower=None, upper=None, **kwargs):
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').constrain('a', lower, upper, **kwargs)
    return Metric(cost_matrix, Cost())


@pytest.mark.parametrize('value', [-1.0, 1.5])
def test_value_outside_declared_bounds_raises(value):
    metric = _bounded_metric(0, 1)
    with pytest.raises(ValueError, match=re.escape(f'a should lay between 0 and 1, got a value of {value}')):
        metric(Y_TRUE, Y_SCORE, a=value, b=1.0)


@pytest.mark.parametrize('value', [0.0, 0.5, 1.0])
def test_value_inside_declared_bounds_is_accepted(value):
    """The bounds are inclusive, so both endpoints must pass."""
    assert isinstance(_bounded_metric(0, 1)(Y_TRUE, Y_SCORE, a=value, b=1.0), float)


def test_a_lower_bound_alone_is_allowed():
    metric = _bounded_metric(lower=0)
    assert isinstance(metric(Y_TRUE, Y_SCORE, a=10.0, b=1.0), float)
    with pytest.raises(ValueError, match='a should be at least 0'):
        metric(Y_TRUE, Y_SCORE, a=-0.5, b=1.0)


def test_an_upper_bound_alone_is_allowed():
    metric = _bounded_metric(upper=1)
    with pytest.raises(ValueError, match='a should be at most 1'):
        metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0)


def test_instance_dependent_values_are_checked_elementwise():
    """A single out-of-range entry in a per-sample vector must be caught."""
    values = np.full(len(Y_TRUE), 0.5)
    values[2] = 3.0
    with pytest.raises(ValueError, match='a should lay between 0 and 1'):
        _bounded_metric(0, 1)(Y_TRUE, Y_SCORE, a=values, b=1.0)


def test_the_error_names_the_alias_the_caller_used():
    """As with the array-length check, the message must use the caller's vocabulary."""
    cost_matrix = (
        CostMatrix().add_fp_cost('a').add_fn_cost('b').alias('acceptance_rate', sympy.Symbol('a')).constrain('a', 0, 1)
    )
    metric = Metric(cost_matrix, Cost())
    with pytest.raises(ValueError, match='acceptance_rate should lay between 0 and 1'):
        metric(Y_TRUE, Y_SCORE, acceptance_rate=2.0, b=1.0)


def test_constrain_resolves_an_alias_to_its_symbol():
    """Constraining by alias must bind to the underlying symbol, so passing either name is checked."""
    cost_matrix = (
        CostMatrix()
        .add_fp_cost('a')
        .add_fn_cost('b')
        .alias('acceptance_rate', sympy.Symbol('a'))
        .constrain('acceptance_rate', 0, 1)
    )
    metric = Metric(cost_matrix, Cost())
    with pytest.raises(ValueError, match='should lay between 0 and 1'):
        metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0)


def test_two_constraints_on_one_symbol_compose_to_the_tighter_pair():
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').constrain('a', 0, 10).constrain('a', 2, 5)
    metric = Metric(cost_matrix, Cost())
    assert isinstance(metric(Y_TRUE, Y_SCORE, a=3.0, b=1.0), float)
    with pytest.raises(ValueError, match='a should lay between 2 and 5'):
        metric(Y_TRUE, Y_SCORE, a=7.0, b=1.0)


def test_a_predicate_can_span_several_parameters():
    """The cross-parameter rule the package deliberately does not impose by default."""
    cost_matrix = (
        CostMatrix()
        .add_fp_cost('a')
        .add_fn_cost('b')
        .constrain(lambda params: params['a'] > params['b'], message='a must exceed b')
    )
    metric = Metric(cost_matrix, Cost())
    assert isinstance(metric(Y_TRUE, Y_SCORE, a=5.0, b=1.0), float)
    with pytest.raises(ValueError, match='a must exceed b'):
        metric(Y_TRUE, Y_SCORE, a=1.0, b=5.0)


def test_a_predicate_sees_defaults_that_were_not_passed():
    cost_matrix = (
        CostMatrix()
        .add_fp_cost('a')
        .add_fn_cost('b')
        .set_default(b=5.0)
        .constrain(lambda params: params['a'] > params['b'], message='a must exceed b')
    )
    with pytest.raises(ValueError, match='a must exceed b'):
        Metric(cost_matrix, Cost())(Y_TRUE, Y_SCORE, a=1.0)


def test_constrain_rejects_a_symbol_with_no_bounds():
    with pytest.raises(ValueError, match="Constraining 'a' requires a lower bound, an upper bound, or both"):
        CostMatrix().add_fp_cost('a').constrain('a')


def test_constrain_rejects_inverted_bounds():
    with pytest.raises(ValueError, match='Lower bound 1 is greater than upper bound 0'):
        CostMatrix().add_fp_cost('a').constrain('a', 1, 0)


def test_constrain_requires_a_message_for_a_predicate():
    with pytest.raises(ValueError, match='A message is required when constraining with a callable'):
        CostMatrix().add_fp_cost('a').constrain(lambda params: True)


def test_constrain_rejects_a_non_symbol_target():
    with pytest.raises(TypeError, match=re.escape('The target must be a sympy.Symbol')):
        CostMatrix().add_fp_cost('a').constrain(5, 0, 1)


def test_constraints_survive_the_metrics_deep_copy():
    """Metric deep-copies its cost matrix, so mutating the original must not change the metric."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').constrain('a', 0, 1)
    metric = Metric(cost_matrix, Cost())
    cost_matrix.constrain('b', 0, 1)  # added after the Metric was built
    assert isinstance(metric(Y_TRUE, Y_SCORE, a=0.5, b=99.0), float)
    with pytest.raises(ValueError, match='a should lay between 0 and 1'):
        metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0)


def test_distribution_shape_parameters_are_checked_without_being_declared():
    """sympy already knows a Beta shape must be positive; the package just has to ask it."""
    gamma = sympy.stats.Beta('gamma', sympy.Symbol('alpha'), sympy.Symbol('beta'))
    metric = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost('b'), MaxProfit())
    assert isinstance(metric(Y_TRUE, Y_SCORE, alpha=6.0, beta=14.0, b=1.0), float)
    with pytest.raises(ValueError, match='Shape parameter Alpha must be positive'):
        metric(Y_TRUE, Y_SCORE, alpha=-1.0, beta=14.0, b=1.0)
