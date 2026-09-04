"""
Misuse / defensive-guard tests for the Metric core.

These guard against ways Metric can be misused that don't raise an error but instead silently compute the wrong number.
"""

import re
import warnings

import numpy as np
import pytest
import sympy

from empulse.metrics import Cost, CostMatrix, MaxProfit, Metric, Savings, cost_loss, expected_cost_loss
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


# --- Metric performs no validation of y_true / y_score (regression vs. cost_loss) ----


COST_MATRIX = CostMatrix().add_fp_cost('a').add_fn_cost('b')


@pytest.mark.parametrize(
    'y_true',
    [
        np.array([1, 2, 1, 2, 1]),
        np.array([0, 1, 0, 1, 2]),
    ],
)
def test_non_binary_y_true_raises(y_true):
    """A non-binary y_true must be rejected, matching cost_loss()'s existing behaviour."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='should be binary'):
        metric(y_true, Y_SCORE, a=1.0, b=1.0)


def test_single_class_y_true_is_accepted():
    """
    A y_true with only one distinct value must be accepted, not rejected.

    Unlike cost_loss()/lift_score(), Metric deliberately does not require variance in y_true: a
    single-class fold or sample-weighted subset is a legitimate input here (e.g. cross-validation,
    or the `check_classifiers_one_label_sample_weights` sklearn compliance check), and rejecting it
    would be needless friction rather than catching a real mistake.
    """
    metric = Metric(COST_MATRIX, Cost())

    score = metric(np.array([1, 1, 1, 1, 1]), Y_SCORE, a=1.0, b=1.0)

    assert isinstance(score, float)


def test_nan_in_y_true_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='NaN'):
        metric(np.array([1.0, 0.0, np.nan, 0.0, 1.0]), Y_SCORE, a=1.0, b=1.0)


def test_nan_in_y_score_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='NaN'):
        metric(Y_TRUE, np.array([0.9, 0.1, np.nan, 0.2, 0.7]), a=1.0, b=1.0)


def test_inf_in_y_score_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='Inf'):
        metric(Y_TRUE, np.array([0.9, 0.1, np.inf, 0.2, 0.7]), a=1.0, b=1.0)


def test_expected_cost_loss_and_cost_loss_raise_the_same_error_on_non_binary_labels():
    """expected_cost_loss (a Metric) and cost_loss (the legacy function) must reject bad input alike."""
    y_bad = np.array([1, 2, 1, 2, 1])

    with pytest.raises(ValueError, match='should be binary') as legacy_exc_info:
        cost_loss(y_bad, Y_SCORE, fp_cost=1.0, fn_cost=1.0)
    with pytest.raises(ValueError, match='should be binary') as metric_exc_info:
        expected_cost_loss(y_bad, Y_SCORE, fp_cost=1.0, fn_cost=1.0)

    assert str(legacy_exc_info.value) == str(metric_exc_info.value)


def test_optimal_threshold_still_accepts_empty_y_true_and_y_score():
    """The documented empty-placeholder path (predict-time threshold, no labels available) must still work."""
    metric = Metric(COST_MATRIX, Cost())

    threshold = metric.optimal_threshold(np.array([]), np.array([]), a=1.0, b=1.0)

    assert threshold == pytest.approx(0.5)


def test_optimal_rate_still_accepts_empty_y_true_with_real_y_score():
    """The documented empty-y_true path (rate needed, but no labels available yet) must still work."""
    metric = Metric(COST_MATRIX, Cost())

    rate = metric.optimal_rate(np.array([]), Y_SCORE, a=1.0, b=1.0)

    assert 0.0 <= rate <= 1.0


def test_optimal_threshold_validates_non_empty_y_true():
    """When y_true is actually provided (non-empty), it must still be validated like everywhere else."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='should be binary'):
        metric.optimal_threshold(np.array([1, 2, 1, 2, 1]), Y_SCORE, a=1.0, b=1.0)


def test_call_accepts_a_row_vector_y_score():
    """A shape (1, n_samples) y_score (as catboost's internal metric callback passes) must still work."""
    metric = Metric(COST_MATRIX, Cost())

    score = metric(Y_TRUE, Y_SCORE.reshape(1, -1), a=1.0, b=1.0)

    assert score == pytest.approx(metric(Y_TRUE, Y_SCORE, a=1.0, b=1.0))


# --- wrong-length instance-dependent parameters ---------------------------------------


def test_wrong_length_array_parameter_raises():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match=r"Parameter 'a' has length 3, but expected length 5"):
        metric(Y_TRUE, Y_SCORE, a=np.array([1.0, 2.0, 3.0]), b=1.0)


def test_wrong_length_array_parameter_error_names_the_alias_the_caller_used():
    """The error should name what the caller actually typed, not the resolved symbol name."""
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b').alias({'my_cost': 'a'})
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match=r"Parameter 'my_cost' has length 2"):
        metric(Y_TRUE, Y_SCORE, my_cost=np.array([1.0, 2.0]), b=1.0)


def test_size_one_array_parameter_is_accepted_as_scalar_broadcast():
    """A length-1 array must be accepted as a scalar-like broadcast, not rejected as 'wrong length'."""
    metric = Metric(COST_MATRIX, Cost())

    broadcast = metric(Y_TRUE, Y_SCORE, a=np.array([2.0]), b=1.0)
    scalar = metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0)

    assert broadcast == pytest.approx(scalar)


def test_correct_length_array_parameter_is_accepted():
    metric = Metric(COST_MATRIX, Cost())

    score = metric(Y_TRUE, Y_SCORE, a=np.full(Y_TRUE.shape, 2.0), b=1.0)

    assert score == pytest.approx(metric(Y_TRUE, Y_SCORE, a=2.0, b=1.0))


def test_optimal_threshold_also_validates_parameter_length():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match=r"Parameter 'a' has length 2, but expected length 5"):
        metric.optimal_threshold(Y_TRUE, Y_SCORE, a=np.array([1.0, 2.0]), b=1.0)


def test_optimal_threshold_predict_time_empty_arrays_skip_length_check():
    """The documented empty-placeholder path (both y_true and y_score empty) must not spuriously
    reject scalar-reduced loss params - there's no sample count to check a length against."""
    metric = Metric(COST_MATRIX, Cost())

    threshold = metric.optimal_threshold(np.array([]), np.array([]), a=1.0, b=1.0)

    assert threshold == pytest.approx(0.5)


# --- optimal_threshold/optimal_rate out of [0, 1] or degenerate cost matrices --------


def test_degenerate_cost_matrix_raises_instead_of_returning_a_huge_number():
    """A cost matrix whose denominator evaluates to 0 must raise, not silently substitute eps."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='degenerate'):
        metric.optimal_threshold(Y_TRUE, Y_SCORE, a=1.0, b=-1.0)


def test_degenerate_cost_matrix_raises_for_optimal_rate_too():
    metric = Metric(COST_MATRIX, Cost())

    with pytest.raises(ValueError, match='degenerate'):
        metric.optimal_rate(Y_TRUE, Y_SCORE, a=1.0, b=-1.0)


def test_out_of_range_threshold_is_clipped_and_warns():
    """A threshold outside [0, 1] must be clipped into range and warn, not returned raw."""
    metric = Metric(COST_MATRIX, Cost())

    with pytest.warns(UserWarning, match=r'fell outside \[0, 1\]'):
        threshold = metric.optimal_threshold(Y_TRUE, Y_SCORE, a=1.0, b=-3.0)

    assert 0.0 <= threshold <= 1.0


def test_in_range_threshold_does_not_warn():
    metric = Metric(COST_MATRIX, Cost())

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        threshold = metric.optimal_threshold(Y_TRUE, Y_SCORE, a=1.0, b=1.0)

    assert 0.0 <= threshold <= 1.0


def test_symbol_used_only_in_the_threshold_numerator_is_still_required():
    """
    A symbol that cancels out of the denominator (fp_cost + tn_benefit + fn_cost + tp_benefit) but
    survives in the numerator (fp_cost + tn_benefit) must still be caught as missing by
    _check_parameters - not left to surface as a cryptic internal TypeError.
    """
    a = sympy.Symbol('a')
    # numerator = fp_cost + tn_benefit = a; denominator = a + 0 + (1 - a) + 0 = 1 (a cancels out).
    cost_matrix = CostMatrix().add_fp_cost(a).add_fn_cost(1 - a)
    metric = Metric(cost_matrix, Cost())

    with pytest.raises(ValueError, match='expected a value for a'):
        metric.optimal_threshold(Y_TRUE, Y_SCORE)
