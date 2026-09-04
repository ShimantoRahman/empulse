"""
Reference (native/legacy) implementations of the customer churn metrics.

These are the hand-written math implementations that used to live in
``empulse.metrics.churn`` before it was refactored to build ``empc_score``, ``mpc_score``,
``empb_score``, ``auepc_score``, and ``expected_cost_loss_churn`` from
:class:`~empulse.metrics.Metric` instances instead. They are kept here, unchanged, purely as
ground truth to numerically verify the new prebuilt metrics against (see
``test_churn_reference_equivalence.py``), and to keep exercising the original edge-case and
input-validation test coverage in ``test_churn.py``.
"""

import numbers
import warnings
from collections.abc import Callable, Sequence
from functools import lru_cache, partial, update_wrapper
from typing import Literal, TypeVar, overload

import numpy as np
from numpy.typing import NBitBase
from scipy import stats as st
from scipy.special import expit

from empulse._types import FloatArrayLike, FloatNDArray
from empulse.metrics._cy_convex_hull import convex_hull
from empulse.metrics._validation import (
    _check_fraction,
    _check_gt_one,
    _check_positive,
    _check_shape,
    _check_y_pred,
    _check_y_true,
)

from ._common import _compute_prior_class_probabilities, _compute_profits, _compute_tpr_fpr_diffs

T = TypeVar('T', bound=NBitBase)


# --- Input validation ----------------------------------------------------------------------


def _validate_input(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, clv: float | FloatArrayLike, d: float, f: float
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    if not isinstance(clv, numbers.Real):
        clv = np.asarray(clv)
        mean_clv = np.mean(clv)
        _check_positive(float(mean_clv), 'clv')
    else:
        _check_positive(clv, 'clv')
    _check_positive(d, 'incentive_cost')
    _check_positive(f, 'contact_cost')
    if isinstance(clv, numbers.Real) and clv <= d:
        raise ValueError(f'clv should be greater than d, got a value of {clv} for clv and for {d} instead.')
    if isinstance(clv, np.ndarray) and np.mean(clv) <= d:
        raise ValueError(f'mean clv should be greater than d, got a value of {clv} for mean clv and for {d} instead.')

    return y_true, y_pred, clv


def _validate_input_emp(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    alpha: float,
    beta: float,
    clv: float | FloatArrayLike,
    d: float,
    f: float,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    _check_gt_one(alpha, 'alpha')
    _check_gt_one(beta, 'beta')
    return _validate_input(y_true, y_pred, clv, d, f)


def _validate_input_mp(
    y_true: FloatArrayLike, y_pred: FloatArrayLike, gamma: float, clv: float | FloatArrayLike, d: float, f: float
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    _check_fraction(gamma, 'gamma')
    return _validate_input(y_true, y_pred, clv, d, f)


def _validate_input_mpc(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    clv: FloatArrayLike,
    accept_rate: float,
    incentive_fraction: float,
    contact_cost: float,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray | float]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    clv = np.asarray(clv)
    _check_fraction(accept_rate, 'accept_rate')
    _check_fraction(incentive_fraction, 'incentive_fraction')
    _check_positive(contact_cost, 'contact_cost')

    return y_true, y_pred, clv


def _validate_input_empb(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    clv: FloatArrayLike,
    alpha: float,
    beta: float,
    incentive_fraction: float,
    contact_cost: float,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    clv = np.asarray(clv)
    _check_positive(alpha, 'alpha')
    _check_positive(beta, 'beta')
    _check_fraction(incentive_fraction, 'incentive_fraction')
    _check_positive(contact_cost, 'contact_cost')

    return y_true, y_pred, clv


def _validate_input_cost_loss_churn(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    clv: FloatArrayLike,
    accept_rate: float,
    incentive_fraction: float | FloatArrayLike,
    contact_cost: float,
) -> tuple[
    FloatNDArray,
    FloatNDArray,
    FloatNDArray,
    FloatNDArray | float,
]:
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    clv = np.asarray(clv)
    _check_fraction(accept_rate, 'accept_rate')
    if isinstance(incentive_fraction, float | int):
        _check_fraction(incentive_fraction, 'incentive_fraction')
    else:
        incentive_fraction = np.asarray(incentive_fraction)
    _check_positive(contact_cost, 'contact_cost')

    return y_true, y_pred, clv, incentive_fraction


# --- Deterministic: mpc / mpc_score ---------------------------------------------------------


def mpc_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    accept_rate: float = 0.3,
    clv: float = 200,
    incentive_cost: float = 10,
    contact_cost: float = 1,
    check_input: bool = True,
) -> float:
    """Maximum Profit Measure for Customer Churn (MPC), only returning the MPC score."""
    return mpc(
        y_true,
        y_score,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        accept_rate=accept_rate,
        check_input=check_input,
    )[0]


def mpc(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    accept_rate: float = 0.3,
    clv: FloatArrayLike | float = 200,
    incentive_cost: float = 10,
    contact_cost: float = 1,
    check_input: bool = True,
) -> tuple[float, float]:
    """Maximum Profit Measure for Customer Churn (MPC)."""
    profits, customer_thresholds = _compute_profit_churn(
        y_true,
        y_score,
        clv,
        incentive_cost,
        contact_cost,
        accept_rate,
        check_input,
    )
    max_profit_index = np.argmax(profits)

    return profits[max_profit_index], customer_thresholds[max_profit_index]


def _compute_profit_churn(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    clv: FloatArrayLike | float = 200,
    d: float = 10,
    f: float = 1,
    gamma: float = 0.3,
    check_input: bool = True,
) -> tuple[FloatNDArray, FloatNDArray]:
    if check_input:
        y_true, y_score, clv = _validate_input_mp(y_true, y_score, gamma, clv, d, f)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)
    if isinstance(clv, np.ndarray):
        clv = np.mean(clv)
    cost_benefits = _compute_cost_benefits(gamma, clv, d, f)
    return _compute_profits(y_true, y_score, cost_benefits)


@lru_cache(maxsize=1)
def _compute_cost_benefits(gamma: float, clv: float, d: float, f: float) -> FloatNDArray:
    delta = d / clv
    phi = f / clv

    true_positive_benefit = clv * (gamma * (1 - delta) - phi)
    false_positive_cost = -1 * clv * (delta + phi)
    return np.array([true_positive_benefit, false_positive_cost])


# --- Stochastic: empc / empc_score, empb / empb_score, auepc_score --------------------------


def empc_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    alpha: float = 6,
    beta: float = 14,
    clv: float | FloatArrayLike = 200,
    incentive_cost: float = 10,
    contact_cost: float = 1,
    check_input: bool = True,
) -> float:
    """Expected Maximum Profit Measure for Customer Churn (EMPC), only returning the EMPC score."""
    return empc(
        y_true,
        y_score,
        alpha=alpha,
        beta=beta,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        check_input=check_input,
    )[0]


def empc(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    alpha: float = 6,
    beta: float = 14,
    clv: float | FloatArrayLike = 200,
    incentive_cost: float = 10,
    contact_cost: float = 1,
    check_input: bool = True,
) -> tuple[float, float]:
    """Expected Maximum Profit Measure for Customer Churn (EMPC)."""
    if check_input:
        y_true, y_score, clv = _validate_input_emp(y_true, y_score, alpha, beta, clv, incentive_cost, contact_cost)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)

    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    if isinstance(clv, np.ndarray):
        clv = float(np.mean(clv))

    delta = incentive_cost / clv
    phi = contact_cost / clv
    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = convex_hull(y_true, y_score)
    tpr_diff, fpr_diff = _compute_tpr_fpr_diffs(true_positive_rates, false_positive_rates)

    tpr_coef = phi * positive_class_prob
    fpr_coef = (delta + phi) * negative_class_prob

    gamma_bounds = _compute_gamma_bounds(tpr_coef, fpr_coef, delta, tpr_diff, fpr_diff, positive_class_prob)
    gamma_cdf_diff = np.diff(st.beta.cdf(gamma_bounds, a=alpha, b=beta))
    gamma_cdf_1_diff = np.diff(st.beta.cdf(gamma_bounds, a=alpha + 1, b=beta))

    cutoff = len(true_positive_rates) - len(gamma_cdf_diff)
    if cutoff > 0:
        true_positive_rates = true_positive_rates[:-cutoff]
        false_positive_rates = false_positive_rates[:-cutoff]
    mean_gamma = st.beta.mean(a=alpha, b=beta)
    temp_1 = mean_gamma * (clv * (1 - delta) * positive_class_prob * true_positive_rates)
    temp_2 = clv * (tpr_coef * true_positive_rates + fpr_coef * false_positive_rates)
    empc = (temp_1 * gamma_cdf_1_diff - temp_2 * gamma_cdf_diff).sum()

    customer_threshold = (
        gamma_cdf_diff * (positive_class_prob * true_positive_rates + negative_class_prob * false_positive_rates)
    ).sum()

    return empc, customer_threshold


def _compute_gamma_bounds(
    tpr_coef: float,
    fpr_coef: float,
    delta: float,
    tpr_diff: FloatNDArray,
    fpr_diff: FloatNDArray,
    positive_class_prob: float,
) -> FloatNDArray:
    """Compute the gamma bounds of the integral."""
    numerator = fpr_coef * fpr_diff + tpr_coef * tpr_diff
    denominator = positive_class_prob * (1 - delta) * tpr_diff
    # ignore division by zero warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        gamma_bounds = numerator / denominator
    gamma_bounds = np.append([0], gamma_bounds)
    return np.append(gamma_bounds[gamma_bounds < 1], [1])


def empb_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    clv: FloatArrayLike,
    alpha: float = 6,
    beta: float = 14,
    incentive_fraction: float = 0.05,
    contact_cost: float = 15,
    check_input: bool = True,
) -> float:
    """Expected Maximum Profit Measure for B2B Customer Churn (EMPB), only returning the EMPB score."""
    return empb(
        y_true,
        y_score,
        alpha=alpha,
        beta=beta,
        clv=clv,
        contact_cost=contact_cost,
        incentive_fraction=incentive_fraction,
        check_input=check_input,
    )[0]


def empb(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    clv: FloatArrayLike,
    alpha: float = 6,
    beta: float = 14,
    incentive_fraction: float = 0.05,
    contact_cost: float = 15,
    check_input: bool = True,
) -> tuple[float, float]:
    """Expected Maximum Profit Measure for B2B Customer Churn (EMPB)."""
    if check_input:
        y_true, y_score, clv = _validate_input_empb(y_true, y_score, clv, alpha, beta, incentive_fraction, contact_cost)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)
    gamma = alpha / (alpha + beta)

    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true[sorted_indices]
    sorted_clv = clv[sorted_indices]

    # Calculate cumulative sums for benefits and costs.
    # The contact cost is incurred whenever a churner is contacted, regardless of whether they
    # accept the incentive offer, so it is not scaled by the acceptance rate (gamma); only the
    # retention benefit net of the incentive cost is contingent on acceptance.
    cumulative_benefits = np.cumsum((gamma * (1 - incentive_fraction) * sorted_clv - contact_cost) * sorted_y_true)
    cumulative_costs = np.cumsum((-contact_cost - incentive_fraction * sorted_clv) * (1 - sorted_y_true))
    cumulative_profits = cumulative_benefits + cumulative_costs

    # Add a zero at the beginning to indicate not contacting anyone
    cumulative_profits = np.insert(cumulative_profits, 0, 0)

    # Find the maximum profit and corresponding threshold
    max_profit_index = np.argmax(cumulative_profits)
    max_profit = cumulative_profits[max_profit_index]
    threshold = max_profit_index / len(y_score)

    return float(max_profit), float(threshold)


def auepc_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    clv: FloatArrayLike,
    alpha: float = 6,
    beta: float = 14,
    incentive_fraction: float = 0.05,
    contact_cost: float = 15,
    normalize: bool = True,
    check_input: bool = True,
) -> float:
    """Area Under the Expected Profit Curve (AUEPC)."""
    if check_input:
        y_true, y_score, clv = _validate_input_empb(y_true, y_score, clv, alpha, beta, incentive_fraction, contact_cost)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)
    if clv.ndim > 1:
        clv = clv[:, 0]

    accept_rate = alpha / (alpha + beta)

    # Calculate the expected profit vector for the perfect model
    perfect_pred_indices = np.argsort(np.where(y_true == 1, 1, -1) * clv)[::-1]
    perfect_targets = y_true[perfect_pred_indices]
    perfect_clv_targets = clv[perfect_pred_indices]

    # The contact cost is incurred whenever a churner is contacted, regardless of whether they
    # accept the incentive offer, so it is not scaled by the acceptance rate; only the retention
    # benefit net of the incentive cost is contingent on acceptance.
    perfect_benefits = np.cumsum(
        (accept_rate * (1 - incentive_fraction) * perfect_clv_targets - contact_cost) * perfect_targets
    )
    perfect_costs = np.cumsum((-contact_cost - incentive_fraction * perfect_clv_targets) * (1 - perfect_targets))
    perfect_profits = perfect_benefits + perfect_costs

    # Calculate the expected profit vector for the perfect model
    sorted_indices = y_score.argsort()[::-1]
    targets = y_true[sorted_indices]
    clv_targets = clv[sorted_indices]

    benefits = np.cumsum((accept_rate * (1 - incentive_fraction) * clv_targets - contact_cost) * targets)
    costs = np.cumsum((-contact_cost - incentive_fraction * clv_targets) * (1 - targets))
    profits = benefits + costs

    # Stop at the point where perfect profits become negative
    stop_index: int = np.argmax(perfect_profits < 0) if np.any(perfect_profits < 0) else len(perfect_profits)  # type: ignore[assignment]

    # Calculate the AUEPC
    score = float(np.trapezoid(profits[:stop_index] / perfect_profits[:stop_index], dx=1 / len(profits)))  # type: ignore[attr-defined]
    if normalize:
        score /= (stop_index - 1) / len(profits)
    return score


# --- Cost: make_objective_churn, AEC classes, expected_cost_loss_churn ----------------------


@overload
def make_objective_churn(
    model: Literal['catboost'],
    *,
    accept_rate: float = 0.3,
    clv: float | FloatNDArray = 200,
    incentive_fraction: float | FloatNDArray = 0.05,
    contact_cost: float = 15,
) -> tuple['AECObjectiveChurn', 'AECMetricChurn']: ...


@overload
def make_objective_churn(
    model: Literal['xgboost', 'lightgbm'],
    *,
    accept_rate: float = 0.3,
    clv: float | FloatNDArray = 200,
    incentive_fraction: float | FloatNDArray = 0.05,
    contact_cost: float = 15,
) -> Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]: ...


def make_objective_churn(
    model: Literal['xgboost', 'lightgbm', 'catboost'],
    *,
    accept_rate: float = 0.3,
    clv: float | FloatNDArray = 200,
    incentive_fraction: float | FloatNDArray = 0.05,
    contact_cost: float = 15,
) -> (
    Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]
    | tuple['AECObjectiveChurn', 'AECMetricChurn']
):
    """Create an objective function for the Expected Cost measure for customer churn."""
    if model == 'xgboost':
        objective: Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]] = partial(
            _objective,
            accept_rate=accept_rate,
            clv=clv,
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost,
        )
        update_wrapper(objective, _objective)
    elif model == 'lightgbm':

        def objective(y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
            """Create an objective function for the churn AEC measure."""
            return _objective(
                y_true,
                y_score,
                accept_rate=accept_rate,
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
            )

    elif model == 'catboost':
        return (
            AECObjectiveChurn(
                accept_rate=accept_rate,
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
            ),
            AECMetricChurn(
                accept_rate=accept_rate,
                clv=clv,
                incentive_fraction=incentive_fraction,
                contact_cost=contact_cost,
            ),
        )
    else:
        raise ValueError(f"Expected model to be 'xgboost', 'lightgbm' or 'catboost', got {model} instead.")
    return objective


def _objective(
    y_true: FloatNDArray,
    y_score: FloatNDArray,
    accept_rate: float = 0.3,
    clv: float | FloatNDArray = 200,
    incentive_fraction: float | FloatNDArray = 0.05,
    contact_cost: float = 1,
) -> tuple[FloatNDArray, FloatNDArray]:
    """Objective function for XGBoost to maximize the profit of a churn model."""
    y_proba = expit(y_score)

    incentive_cost = incentive_fraction * clv
    profits = (
        contact_cost + incentive_cost + y_true * (accept_rate * incentive_cost - incentive_cost - clv * accept_rate)
    )
    gradient = y_proba * (1 - y_proba) * profits
    hessian = np.abs((1 - 2 * y_proba) * gradient)
    return gradient, hessian


class AECObjectiveChurn:
    """AEC churn objective for catboost."""

    def __init__(
        self,
        accept_rate: float = 0.3,
        clv: float | FloatNDArray = 200,
        incentive_fraction: float | FloatNDArray = 0.05,
        contact_cost: float = 1,
    ):
        self.accept_rate = accept_rate
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost

    def calc_ders_range(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: FloatNDArray
    ) -> list[tuple[float, float]]:
        """Compute first and second derivative of the loss function wrt the predicted value for each object."""
        # Use weights as a proxy to index the costs
        weights = weights.astype(int)
        clv = self.clv[weights] if isinstance(self.clv, np.ndarray) else self.clv

        y_proba = expit(predictions)
        incentive_cost = self.incentive_fraction * clv
        profits = (
            self.contact_cost
            + incentive_cost
            + targets * (self.accept_rate * incentive_cost - incentive_cost - clv * self.accept_rate)
        )
        gradient = y_proba * (1 - y_proba) * profits
        hessian = np.abs((1 - 2 * y_proba) * gradient)
        return list(zip(-gradient, -hessian, strict=False))


class AECMetricChurn:
    """AEC churn metric for catboost."""

    def __init__(
        self,
        accept_rate: float = 0.3,
        clv: float | FloatNDArray = 200,
        incentive_fraction: float | FloatNDArray = 0.05,
        contact_cost: float = 1,
    ):
        self.accept_rate = accept_rate
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost

    def is_max_optimal(self) -> bool:
        """Return whether great values of metric are better."""
        return False

    def evaluate(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: FloatNDArray
    ) -> tuple[float, float]:
        """Evaluate metric value."""
        # Use weights as a proxy to index the costs
        weights = weights.astype(int)
        clv = self.clv[weights] if isinstance(self.clv, np.ndarray) else self.clv

        y_proba = expit(predictions)
        return expected_cost_loss_churn(
            targets,
            y_proba,
            accept_rate=self.accept_rate,
            clv=clv,
            incentive_fraction=self.incentive_fraction,
            contact_cost=self.contact_cost,
            normalize=True,
            check_input=False,
        ), 1

    def get_final_error(self, error: float, weight: float) -> float:
        """Return final value of metric based on error and weight."""
        return error


def expected_cost_loss_churn(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    accept_rate: float = 0.3,
    clv: float | FloatArrayLike = 200,
    incentive_fraction: float | FloatArrayLike = 0.05,
    contact_cost: float = 1,
    normalize: bool = False,
    check_input: bool = True,
) -> float:
    """Expected cost of a classifier for customer churn."""
    if check_input:
        y_true, y_proba, clv, incentive_fraction = _validate_input_cost_loss_churn(
            y_true,
            y_proba,
            clv=clv,
            accept_rate=accept_rate,
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost,
        )
    else:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        clv = np.asarray(clv)
        incentive_fraction = np.asarray(incentive_fraction)

    incentive_cost = incentive_fraction * clv
    profits = y_proba * (
        contact_cost + incentive_cost + y_true * (accept_rate * incentive_cost - incentive_cost - clv * accept_rate)
    )
    if normalize:
        return float(np.mean(profits))
    return float(np.sum(profits))
