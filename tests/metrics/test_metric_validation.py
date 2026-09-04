"""
Misuse / defensive-guard tests for the Metric core.

These guard against ways Metric can be misused that don't raise an error but instead silently compute the wrong number.
"""

import re

import numpy as np
import pytest
import sympy

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, Savings
from empulse.metrics.metric.common import RESERVED_SYMBOL_NAMES

Y_TRUE = np.array([1, 0, 1, 0, 1])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.2, 0.7])


# --- reusing one MetricStrategy instance across two Metrics -------------------------


def test_metric_owns_an_independent_strategy_copy():
    """`Metric` must not share mutable strategy state with the object passed to it."""
    strategy = Cost()
    metric = Metric(CostMatrix().add_fp_cost('a'), strategy)

    assert metric.strategy is not strategy


def test_reusing_one_strategy_instance_keeps_metrics_independent():
    """
    Building a second Metric from the same strategy instance must not corrupt the first.

    Before the fix, `Metric.__init__` built the strategy object in place, so the second
    `Metric`'s `build()` call silently overwrote the first metric's compiled formula.
    """
    strategy = Cost()
    metric_fp = Metric(CostMatrix().add_fp_cost('a'), strategy)
    # Building a second Metric from the *same* strategy instance, with an overlapping symbol
    # name, used to silently rewrite metric_fp's formula.
    metric_fn = Metric(CostMatrix().add_fn_cost('a'), strategy)

    expected_fp = Metric(CostMatrix().add_fp_cost('a'), Cost())(Y_TRUE, Y_SCORE, a=100.0)
    expected_fn = Metric(CostMatrix().add_fn_cost('a'), Cost())(Y_TRUE, Y_SCORE, a=100.0)

    assert metric_fp(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(expected_fp)
    assert metric_fn(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(expected_fn)
    assert expected_fp != expected_fn  # sanity check the two formulas actually differ


def test_reusing_one_strategy_instance_with_disjoint_symbols_still_works():
    """Same scenario as above but with non-overlapping symbol names, for good measure."""
    strategy = Cost()
    metric_a = Metric(CostMatrix().add_fp_cost('a'), strategy)
    metric_b = Metric(CostMatrix().add_fn_cost('b'), strategy)

    assert metric_a(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(
        Metric(CostMatrix().add_fp_cost('a'), Cost())(Y_TRUE, Y_SCORE, a=100.0)
    )
    assert metric_b(Y_TRUE, Y_SCORE, b=100.0) == pytest.approx(
        Metric(CostMatrix().add_fn_cost('b'), Cost())(Y_TRUE, Y_SCORE, b=100.0)
    )


# --- mutating a CostMatrix after building a Metric ----------------------------------


def test_metric_owns_an_independent_cost_matrix_copy():
    """`Metric` must not share the mutable `CostMatrix` object passed to it."""
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())

    assert metric.cost_matrix is not cost_matrix


def test_mutating_source_cost_matrix_does_not_affect_the_metric():
    """
    A `Metric` is a snapshot: later mutation of the `CostMatrix` it was built from must not
    change either its computed score or its advertised cost expressions.
    """
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())

    score_before = metric(Y_TRUE, Y_SCORE, a=100.0)
    fn_cost_before = metric.fn_cost

    cost_matrix.add_fn_cost('b')  # mutate the source matrix after the metric was built

    assert metric.fn_cost == fn_cost_before
    assert metric.fn_cost == sympy.Integer(0)
    assert metric(Y_TRUE, Y_SCORE, a=100.0) == pytest.approx(score_before)


def test_mutating_source_cost_matrix_new_parameter_is_unknown_to_the_metric():
    """
    A parameter added to the source matrix after Metric construction must not be silently
    accepted - it should warn as an unknown parameter, since the metric never learned about it.
    """
    cost_matrix = CostMatrix().add_fp_cost('a')
    metric = Metric(cost_matrix, Cost())
    cost_matrix.add_fn_cost('b')

    with pytest.warns(UserWarning, match=r"Unknown parameters passed to metric: \['b'\]"):
        metric(Y_TRUE, Y_SCORE, a=100.0, b=100.0)


# --- user symbols colliding with names reserved for internal use --------------------


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


# --- passing both an alias and its underlying symbol ---------------------------------


@pytest.mark.parametrize('kwarg_order', ['symbol_first', 'alias_first'])
def test_alias_and_underlying_symbol_together_raises(kwarg_order):
    """
    Passing both a symbol and one of its aliases must raise, regardless of which one the caller
    happened to type first - not silently let whichever kwarg is seen last win, which would make
    the result depend on kwarg order.
    """
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())
    kwargs = (
        {'a': 1.0, 'my_cost': 999.0, 'b': 1.0}
        if kwarg_order == 'symbol_first'
        else {
            'my_cost': 999.0,
            'a': 1.0,
            'b': 1.0,
        }
    )

    with pytest.raises(ValueError, match="conflicting values for symbol 'a'"):
        metric(Y_TRUE, Y_SCORE, **kwargs)


def test_two_aliases_for_the_same_symbol_together_raises():
    """Passing two different aliases that both target the same symbol must raise too."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a', 'other_name': 'a'})
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match="conflicting values for symbol 'a'"):
        metric(Y_TRUE, Y_SCORE, my_cost=1.0, other_name=2.0, b=1.0)


def test_alias_or_symbol_passed_alone_is_unaffected():
    """Sanity check: passing only the symbol, or only the alias, must still work as before."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())

    assert metric(Y_TRUE, Y_SCORE, a=1.0, b=1.0) == pytest.approx(0.18)
    assert metric(Y_TRUE, Y_SCORE, my_cost=1.0, b=1.0) == pytest.approx(0.18)


# --- alias()/set_default() referring to a symbol that does not exist -----------------


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
