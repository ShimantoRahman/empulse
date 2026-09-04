"""
Misuse / defensive-guard tests for the Metric core.

These guard against ways Metric can be misused that don't raise an error but instead silently compute the wrong number.
"""

import numpy as np
import pytest
import sympy

from empulse.metrics import Cost, CostMatrix, Metric

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
