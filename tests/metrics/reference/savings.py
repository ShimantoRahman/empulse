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

from collections.abc import Callable, Sequence
from functools import partial, update_wrapper
from typing import Any, Literal, overload

import numpy as np
from scipy.special import expit

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


@overload
def make_objective_aec(
    model: Literal['catboost'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> tuple['AECObjective', 'AECMetric']: ...


@overload
def make_objective_aec(
    model: Literal['xgboost', 'lightgbm'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]: ...


@overload
def make_objective_aec(
    model: Literal['cslogit'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> Callable[[FloatNDArray, FloatNDArray, FloatNDArray], tuple[float, FloatNDArray]]: ...


def make_objective_aec(
    model: Literal['xgboost', 'lightgbm', 'catboost', 'cslogit'],
    *,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> (
    Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]
    | Callable[[FloatNDArray, FloatNDArray, FloatNDArray], tuple[float, FloatNDArray]]
    | tuple['AECObjective', 'AECMetric']
):
    """Create an objective function for the Average Expected Cost (AEC) measure."""
    if model == 'xgboost':
        objective: Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]] = partial(
            _objective_boost, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost
        )
        update_wrapper(objective, _objective_boost)
    elif model == 'lightgbm':

        def objective(y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
            """Create an objective function for the AEC measure."""
            return _objective_boost(y_true, y_score, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)

    elif model == 'catboost':
        return (
            AECObjective(tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost),
            AECMetric(tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost),
        )
    elif model == 'cslogit':
        objective = partial(_objective_cslogit, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)  # type: ignore[assignment]
        update_wrapper(objective, _objective_cslogit)
    else:
        raise ValueError(
            f"Expected model to be one of 'xgboost', 'lightgbm', 'catboost' or 'cslogit', got {model} instead."
        )

    return objective


def _objective_cslogit(
    features: FloatNDArray,
    weights: FloatNDArray,
    y_true: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> tuple[float, FloatNDArray]:
    y_pred = expit(np.dot(weights, features.T))

    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=1)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=1)

    average_expected_cost = expected_cost_loss(
        y_true,
        y_pred,
        tp_cost=tp_cost,
        tn_cost=tn_cost,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        normalize=True,
        check_input=False,
    )
    gradient = np.mean(
        features * y_pred * (1 - y_pred) * (y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)), axis=0
    )
    return average_expected_cost, gradient


def _objective_boost(
    y_true: FloatNDArray,
    y_score: FloatNDArray,
    tp_cost: FloatNDArray | float = 0.0,
    tn_cost: FloatNDArray | float = 0.0,
    fn_cost: FloatNDArray | float = 0.0,
    fp_cost: FloatNDArray | float = 0.0,
) -> tuple[FloatNDArray, FloatNDArray]:
    """Create an objective function for the AEC measure."""
    y_proba = expit(y_score)
    cost = y_true * (tp_cost - fn_cost) + (1 - y_true) * (fp_cost - tn_cost)
    gradient = y_proba * (1 - y_proba) * cost
    hessian = np.abs((1 - 2 * y_proba) * gradient)
    return gradient, hessian


class AECObjective:
    """AEC objective for catboost."""

    def __init__(
        self,
        tp_cost: FloatNDArray | float = 0.0,
        tn_cost: FloatNDArray | float = 0.0,
        fn_cost: FloatNDArray | float = 0.0,
        fp_cost: FloatNDArray | float = 0.0,
    ):
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def calc_ders_range(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: FloatNDArray
    ) -> list[tuple[float, float]]:
        """Compute first and second derivative of the loss function with respect to the predicted value."""
        weights = weights.astype(int)
        # Use weights as a proxy to index the costs
        tp_cost = self.tp_cost[weights] if isinstance(self.tp_cost, np.ndarray) else self.tp_cost
        tn_cost = self.tn_cost[weights] if isinstance(self.tn_cost, np.ndarray) else self.tn_cost
        fn_cost = self.fn_cost[weights] if isinstance(self.fn_cost, np.ndarray) else self.fn_cost
        fp_cost = self.fp_cost[weights] if isinstance(self.fp_cost, np.ndarray) else self.fp_cost

        y_proba = expit(predictions)
        cost = targets * (tp_cost - fn_cost) + (1 - targets) * (fp_cost - tn_cost)
        gradient = y_proba * (1 - y_proba) * cost
        hessian = np.abs((1 - 2 * y_proba) * gradient)
        # convert from two arrays to one list of tuples
        return list(zip(-gradient, -hessian, strict=False))


class AECMetric:
    """AEC metric for catboost."""

    def __init__(
        self,
        tp_cost: FloatNDArray | float = 0.0,
        tn_cost: FloatNDArray | float = 0.0,
        fn_cost: FloatNDArray | float = 0.0,
        fp_cost: FloatNDArray | float = 0.0,
    ) -> None:
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def is_max_optimal(self) -> bool:
        """Return whether great values of metric are better."""
        return False

    def evaluate(
        self, predictions: Sequence[float], targets: Sequence[float], weights: FloatNDArray
    ) -> tuple[float, float]:
        """Evaluate metric value."""
        weights = weights.astype(int)
        # Use weights as a proxy to index the costs
        tp_cost = self.tp_cost[weights] if isinstance(self.tp_cost, np.ndarray) else self.tp_cost
        tn_cost = self.tn_cost[weights] if isinstance(self.tn_cost, np.ndarray) else self.tn_cost
        fn_cost = self.fn_cost[weights] if isinstance(self.fn_cost, np.ndarray) else self.fn_cost
        fp_cost = self.fp_cost[weights] if isinstance(self.fp_cost, np.ndarray) else self.fp_cost

        y_proba = expit(predictions)
        return expected_cost_loss(
            targets,
            y_proba,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            normalize=True,
            check_input=False,
        ), 1

    def get_final_error(self, error: float, weight: float) -> float:
        """Return final value of metric based on error and weight."""
        return error
