"""
Reference (native/legacy) implementations of the generic expected-cost/savings metrics.

These are the hand-written math implementations that used to live in ``empulse.metrics.savings``
before ``expected_cost_loss``, ``expected_log_cost_loss``, and ``expected_savings_score`` were
refactored to build on :class:`~empulse.metrics.Metric` instances instead, and before
``make_objective_aec``/``AECObjective``/``AECMetric`` were removed (their job is now done by
passing any :class:`~empulse.metrics.Metric` as ``loss=`` to a cost-sensitive boosting model
directly). They are kept here, unchanged, purely as ground truth to numerically verify the new
prebuilt metrics against, and to keep exercising the original edge-case and input-validation test
coverage in ``test_savings.py``.

``cost_loss`` and ``savings_score`` (the hard-label/auto-thresholding variants) were **not**
refactored -- they remain in the live package unchanged, since there is no
:class:`~empulse.metrics.Metric`/:class:`~empulse.metrics.MetricStrategy` equivalent for their
auto-thresholding behavior. ``_validate_input``/``_compute_expected_cost`` also remain live (still
used by ``cost_loss``/``savings_score``), so they are imported from the package here rather than
duplicated.
"""

from typing import Any, Literal

import numpy as np

from empulse._types import FloatArrayLike, FloatNDArray
from empulse.metrics.savings import _compute_expected_cost, _validate_input, cost_loss

# --- Expected cost / expected log cost --------------------------------------------------------


def _compute_log_expected_cost(
    y_true: FloatNDArray,
    y_pred: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> FloatNDArray:
    epsilon: np.floating[Any] = np.finfo(y_pred.dtype).eps  # type: ignore[arg-type]
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    inverse_y_pred = 1 - y_pred
    log_y_pred = np.log(y_pred)
    log_inv_y_pred: FloatNDArray = np.log(inverse_y_pred)
    return y_true * (log_y_pred * tp_cost + log_inv_y_pred * fn_cost) + (1 - y_true) * (
        log_y_pred * fp_cost + log_inv_y_pred * tn_cost
    )


def expected_cost_loss(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    tp_cost: float | FloatArrayLike = 0.0,
    fp_cost: float | FloatArrayLike = 0.0,
    tn_cost: float | FloatArrayLike = 0.0,
    fn_cost: float | FloatArrayLike = 0.0,
    normalize: bool = False,
    check_input: bool = True,
) -> float:
    """Expected cost of a classifier."""
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    cost = _compute_expected_cost(y_true, y_proba, tp_cost, tn_cost, fn_cost, fp_cost)

    if normalize:
        return float(np.mean(cost))
    return float(np.sum(cost))


def expected_log_cost_loss(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    tp_cost: FloatArrayLike | float = 0.0,
    tn_cost: FloatArrayLike | float = 0.0,
    fn_cost: FloatArrayLike | float = 0.0,
    fp_cost: FloatArrayLike | float = 0.0,
    normalize: bool = False,
    check_input: bool = True,
) -> float:
    """Expected log cost of a classifier."""
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )
    cost = _compute_log_expected_cost(y_true, y_proba, tp_cost, tn_cost, fn_cost, fp_cost)
    if normalize:
        return float(np.mean(cost))
    return float(np.sum(cost))


# --- Expected savings --------------------------------------------------------------------------


def expected_savings_score(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    baseline: Literal['zero_one', 'one', 'zero', 'prior'] | FloatArrayLike = 'zero_one',
    tp_cost: float | FloatArrayLike = 0.0,
    fp_cost: float | FloatArrayLike = 0.0,
    tn_cost: float | FloatArrayLike = 0.0,
    fn_cost: float | FloatArrayLike = 0.0,
    check_input: bool = True,
) -> float:
    """Expected savings of a classifier compared to a baseline."""
    y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost = _validate_input(
        y_true, y_proba, tp_cost, fp_cost, tn_cost, fn_cost, check_input
    )

    if not isinstance(baseline, str):
        baseline = np.asarray(baseline)
        cost_base = expected_cost_loss(
            y_true,
            baseline,
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )
    elif baseline == 'zero_one':
        # Calculate the cost of naive prediction
        cost_base = min(
            cost_loss(
                y_true,
                np.zeros_like(y_true),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
            cost_loss(
                y_true,
                np.ones_like(y_true),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
        )
    elif baseline == 'one':
        cost_base = cost_loss(
            y_true,
            np.ones_like(y_true),
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )
    elif baseline == 'zero':
        cost_base = cost_loss(
            y_true,
            np.zeros_like(y_true),
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            check_input=False,
        )
    elif baseline == 'prior':
        prior_pos = np.mean(y_true)
        prior_neg = 1 - prior_pos
        cost_base = min(
            cost_loss(
                y_true,
                np.full_like(y_true, prior_pos),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
            cost_loss(
                y_true,
                np.full_like(y_true, prior_neg),
                tp_cost=tp_cost,
                fp_cost=fp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                check_input=False,
            ),
        )
    else:
        raise ValueError("Invalid baseline. Must be 'zero_one', 'zero', 'one', 'prior', or an array-like.")

    # avoid division by zero
    if cost_base == 0.0:
        cost_base = float(np.finfo(float).eps)

    cost = expected_cost_loss(
        y_true, y_proba, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost, check_input=False
    )
    return 1.0 - cost / cost_base


# --- make_objective_aec / AECObjective / AECMetric ----------------------------------------------
