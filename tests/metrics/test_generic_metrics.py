"""
Equivalence tests for the thin-wrapper generic metrics in ``empulse.metrics``.

``max_profit_score``, ``expected_cost_loss``, ``expected_log_cost_loss``, and
``expected_savings_score`` used to be hand-written native math functions living in
``empulse/metrics/max_profit.py`` and ``empulse/metrics/savings.py``. They are now built from
:class:`~empulse.metrics.Metric` instances (see ``empulse/metrics/metric/prebuilt_metrics.py``).
The native math still exists, unchanged, in ``tests/metrics/reference/`` purely as ground truth
-- these tests check that the new instances numerically reproduce it (subject to the documented,
intentional behavior differences noted below).

Notes
-----
* ``max_profit_score`` now takes ``tp_cost``/``tn_cost`` (all-cost naming, consistent with the
  rest of the package) instead of the reference's ``tp_benefit``/``tn_benefit``:
  ``tp_cost = -tp_benefit`` and ``tn_cost = -tn_benefit``. ``fp_cost``/``fn_cost`` are unchanged.
* ``MaxProfit`` (pre-existing, unrelated to this refactor) aggregates any instance-dependent
  (array-like) parameter to its mean before computing the maximum profit, since the profit
  surface is optimized over the convex hull of the ROC curve rather than per-sample. The
  reference ``max_profit``, by contrast, multiplies the raw arrays directly against the
  hull-reduced TPR/FPR arrays, which only broadcasts correctly when a cost happens to be a
  scalar. So the instance-dependent case below compares the new instance against the reference
  called with the *mean* of the arrays, matching the new instance's own documented aggregation
  behavior, rather than against the reference called with the raw arrays.
* ``expected_cost_loss``/``expected_log_cost_loss`` are built on the ``Cost``/``LogCost``
  strategies, which always return the *mean* cost across samples. The reference functions default
  to returning the *sum* (``normalize=False``), so comparisons below pass ``normalize=True`` to
  the reference call.
* ``expected_savings_score``'s ``baseline='prior'`` behavior was *fixed* as part of this refactor:
  the old native implementation (preserved verbatim in the reference module) hard-thresholds the
  constant prior probability through ``cost_loss``'s auto-thresholding logic, which degenerates
  the "expected cost at the prior probability" baseline into a hard all-zero/all-one decision in
  most cases, and it also only ever considered ``prior_pos = mean(y_true)``, never the cheaper
  ``prior_neg = 1 - prior_pos`` alternative. The new ``Savings``-strategy-based instance correctly
  evaluates the continuous expected cost at both ``prior_pos`` and ``prior_neg`` and takes
  whichever is cheaper. So ``baseline='prior'`` is *not* compared against the (buggy) reference
  here; instead it is checked against an independent hand-computed formula using
  ``expected_cost_loss`` at the constant prior probabilities. All other baselines
  (``'zero_one'``, ``'zero'``, ``'one'``, and array-like) are unaffected by the fix and are
  compared directly against the reference.
"""

import numpy as np
import pytest

from empulse.metrics import expected_cost_loss, expected_log_cost_loss, expected_savings_score, max_profit_score

from .reference import max_profit as ref_max_profit
from .reference import savings as ref_savings


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.integers(0, 2, n).astype(float)
    y_score = rng.random(n)
    return y_true, y_score


# --- max_profit_score --------------------------------------------------------------------------


def test_max_profit_score_matches_reference_class_dependent(dataset):
    y_true, y_score = dataset
    tp_benefit, tn_benefit, fp_cost, fn_cost = 4.0, 1.0, 2.0, 3.0

    result = max_profit_score(
        y_true, y_score, tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    expected = ref_max_profit.max_profit_score(
        y_true, y_score, tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    assert result == pytest.approx(expected)


def test_max_profit_score_matches_reference_instance_dependent(dataset):
    """Instance-dependent costs are aggregated to their mean before computing max profit."""
    y_true, y_score = dataset
    rng = np.random.default_rng(11)
    tp_benefit = rng.uniform(1, 5, size=y_true.shape)
    fn_cost = rng.uniform(0.5, 2, size=y_true.shape)
    tn_benefit, fp_cost = 0.5, 1.5

    result = max_profit_score(
        y_true, y_score, tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    expected = ref_max_profit.max_profit_score(
        y_true,
        y_score,
        tp_benefit=float(np.mean(tp_benefit)),
        tn_benefit=tn_benefit,
        fp_cost=fp_cost,
        fn_cost=float(np.mean(fn_cost)),
    )
    assert result == pytest.approx(expected)


# --- expected_cost_loss / expected_log_cost_loss ------------------------------------------------


def test_expected_cost_loss_matches_reference(dataset):
    y_true, y_score = dataset
    params = {'tp_cost': 1.0, 'tn_cost': 3.0, 'fp_cost': 2.0, 'fn_cost': 4.0}

    result = expected_cost_loss(y_true, y_score, **params)
    expected = ref_savings.expected_cost_loss(y_true, y_score, **params, normalize=True)
    assert result == pytest.approx(expected)


def test_expected_cost_loss_matches_reference_instance_dependent(dataset):
    y_true, y_score = dataset
    rng = np.random.default_rng(13)
    params = {
        'tp_cost': rng.uniform(0, 2, size=y_true.shape),
        'fn_cost': rng.uniform(0, 2, size=y_true.shape),
        'fp_cost': 1.5,
        'tn_cost': 0.5,
    }

    result = expected_cost_loss(y_true, y_score, **params)
    expected = ref_savings.expected_cost_loss(y_true, y_score, **params, normalize=True)
    assert result == pytest.approx(expected)


def test_expected_log_cost_loss_matches_reference(dataset):
    y_true, y_score = dataset
    params = {'tp_cost': 0.3, 'tn_cost': 0.6, 'fp_cost': 1.2, 'fn_cost': 2.1}

    result = expected_log_cost_loss(y_true, y_score, **params)
    expected = ref_savings.expected_log_cost_loss(y_true, y_score, **params, normalize=True)
    assert result == pytest.approx(expected)


# --- expected_savings_score ----------------------------------------------------------------------


@pytest.mark.parametrize('baseline', ['zero_one', 'zero', 'one'])
def test_expected_savings_score_matches_reference(dataset, baseline):
    y_true, y_score = dataset
    params = {'tp_cost': 1.0, 'tn_cost': 3.0, 'fp_cost': 2.0, 'fn_cost': 4.0}

    result = expected_savings_score(y_true, y_score, baseline=baseline, **params)
    expected = ref_savings.expected_savings_score(y_true, y_score, baseline=baseline, **params)
    assert result == pytest.approx(expected)


def test_expected_savings_score_matches_reference_array_baseline(dataset):
    y_true, y_score = dataset
    params = {'tp_cost': 1.0, 'tn_cost': 3.0, 'fp_cost': 2.0, 'fn_cost': 4.0}
    baseline = np.full(y_true.shape, 0.5)

    result = expected_savings_score(y_true, y_score, baseline=baseline, **params)
    expected = ref_savings.expected_savings_score(y_true, y_score, baseline=baseline, **params)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    'y_true, tp_cost, fp_cost, tn_cost, fn_cost',
    [
        (np.array([1, 0, 0, 1]), 1.0, 2.0, 3.0, 4.0),
        (np.array([1, 0, 0, 1]), 0.0, 1.0, 0.0, 1.0),
        (np.array([0, 1, 1, 0]), 0.0, np.array([4, 1, 2, 2]), 0.0, np.array([1, 3, 3, 1])),
    ],
)
def test_expected_savings_score_prior_baseline_matches_hand_computed_formula(
    y_true, tp_cost, fp_cost, tn_cost, fn_cost
):
    """
    ``baseline='prior'`` was fixed as part of this refactor (see module docstring): the correct
    semantics are "the expected cost at the prior probability, whichever of ``prior_pos``/
    ``prior_neg`` is cheaper" -- computed here directly with ``expected_cost_loss`` rather than
    the (differently, and more subtly, buggy) native reference.
    """
    rng = np.random.default_rng(17)
    y_score = rng.random(len(y_true))

    prior_pos = float(np.mean(y_true))
    prior_neg = 1 - prior_pos

    def cost_at(constant_score):
        s = np.full(y_true.shape, constant_score)
        return expected_cost_loss(y_true, s, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)

    cost_base = min(cost_at(prior_pos), cost_at(prior_neg))
    if cost_base == 0.0:
        cost_base = float(np.finfo(float).eps)

    cost = expected_cost_loss(y_true, y_score, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)
    expected = 1 - cost / cost_base

    result = expected_savings_score(
        y_true, y_score, baseline='prior', tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost
    )
    assert result == pytest.approx(expected)
