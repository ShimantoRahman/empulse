"""
Reference (native/legacy) implementations of the customer acquisition metrics.

These are the hand-written math implementations that used to live in
``empulse.metrics.acquisition`` before it was refactored to build ``mpa_score``, ``empa_score``,
and ``expected_cost_loss_acquisition`` from :class:`~empulse.metrics.Metric` instances instead.
They are kept here, unchanged, purely as ground truth to numerically verify the new prebuilt
metrics against, and to keep exercising the original edge-case and input-validation test
coverage in ``test_acquisition.py``.
"""

import warnings
from collections.abc import Callable, Sequence
from functools import lru_cache, partial, update_wrapper
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
import scipy.stats as st
from scipy.special import expit

from empulse._types import FloatArrayLike, FloatNDArray
from empulse.metrics._cy_convex_hull import convex_hull
from empulse.metrics._validation import _check_fraction, _check_positive, _check_shape, _check_y_pred, _check_y_true

from ._common import _compute_prior_class_probabilities, _compute_profits, _compute_tpr_fpr_diffs

if TYPE_CHECKING:  # pragma: no cover
    try:
        from lightgbm import Dataset
        from xgboost import DMatrix

        Matrix = TypeVar('Matrix', bound=FloatNDArray | DMatrix | Dataset)
    except ImportError:
        Matrix = TypeVar('Matrix', bound=FloatNDArray)  # type: ignore[misc]
else:
    Matrix = TypeVar('Matrix', bound=FloatNDArray)


# --- Input validation ----------------------------------------------------------------------


def _validate_input(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    contact_cost: float,
    sales_cost: float,
    direct_selling: float,
    commission: float,
) -> tuple[FloatNDArray, FloatNDArray]:
    """Validate input for all acquisition parameters."""
    y_true = _check_y_true(y_true)
    y_pred = _check_y_pred(y_pred)
    _check_shape(y_true, y_pred)
    _check_positive(contact_cost, 'contact_cost')
    _check_positive(sales_cost, 'sales_cost')
    _check_fraction(direct_selling, 'direct_selling')
    _check_fraction(commission, 'commission')

    return y_true, y_pred


def _validate_input_stochastic(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    alpha: float,
    beta: float,
    contact_cost: float,
    sales_cost: float,
    direct_selling: float,
    commission: float,
) -> tuple[FloatNDArray, FloatNDArray]:
    """Validate input for all stochastic acquisition parameters."""
    _check_positive(alpha, 'alpha')
    _check_positive(beta, 'beta')
    return _validate_input(
        y_true,
        y_pred,
        contact_cost,
        sales_cost,
        direct_selling,
        commission,
    )


def _validate_input_deterministic(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    contribution: float,
    contact_cost: float,
    sales_cost: float,
    direct_selling: float,
    commission: float,
) -> tuple[FloatNDArray, FloatNDArray]:
    """Validate input for all deterministic acquisition parameters."""
    _check_positive(contribution, 'contribution')
    return _validate_input(
        y_true,
        y_pred,
        contact_cost,
        sales_cost,
        direct_selling,
        commission,
    )


# --- Deterministic: mpa / mpa_score ---------------------------------------------------------


def mpa_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    contribution: float = 8_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    check_input: bool = True,
) -> float:
    """Maximum Profit measure for customer Acquisition (MPA), only returning the MPA score."""
    return mpa(
        y_true,
        y_score,
        contribution=contribution,
        contact_cost=contact_cost,
        sales_cost=sales_cost,
        direct_selling=direct_selling,
        commission=commission,
        check_input=check_input,
    )[0]


def mpa(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    contribution: float = 8_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    check_input: bool = True,
) -> tuple[float, float]:
    """Maximum Profit measure for customer Acquisition (MPA)."""
    profits, customer_thresholds = _compute_profit_acquisition(
        y_true,
        y_score,
        contribution,
        contact_cost,
        sales_cost,
        direct_selling,
        commission,
        check_input,
    )
    max_profit_index = np.argmax(profits)
    return profits[max_profit_index], customer_thresholds[max_profit_index]


def _compute_profit_acquisition(
    y_true: FloatArrayLike,
    y_pred: FloatArrayLike,
    contribution: float = 8_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    check_input: bool = True,
) -> tuple[FloatNDArray, FloatNDArray]:
    if check_input:
        y_true, y_pred = _validate_input_deterministic(
            y_true, y_pred, contribution, contact_cost, sales_cost, direct_selling, commission
        )
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
    cost_benefits = _compute_cost_benefits(contribution, contact_cost, sales_cost, direct_selling, commission)
    return _compute_profits(y_true, y_pred, cost_benefits)


@lru_cache(maxsize=1)
def _compute_cost_benefits(
    contribution: float,
    contact_cost: float,
    sales_cost: float,
    direct_selling: float,
    commission: float,
) -> FloatNDArray:
    true_positive_benefit = direct_selling * (contribution - contact_cost - sales_cost) + (1 - direct_selling) * (
        (1 - commission) * contribution - contact_cost
    )
    false_positive_cost = -contact_cost
    return np.array([true_positive_benefit, false_positive_cost])


# --- Stochastic: empa / empa_score -----------------------------------------------------------


def empa_score(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    alpha: float = 12,
    beta: float = 0.0015,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    check_input: bool = True,
) -> float:
    """Expected Maximum Profit measure for customer Acquisition (EMPA), only returning the EMPA score."""
    return empa(
        y_true,
        y_score,
        alpha=alpha,
        beta=beta,
        contact_cost=contact_cost,
        sales_cost=sales_cost,
        direct_selling=direct_selling,
        commission=commission,
        check_input=check_input,
    )[0]


def empa(
    y_true: FloatArrayLike,
    y_score: FloatArrayLike,
    *,
    alpha: float = 12,
    beta: float = 0.0015,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    check_input: bool = True,
) -> tuple[float, float]:
    """Expected Maximum Profit measure for customer Acquisition (EMPA)."""
    if check_input:
        y_true, y_score = _validate_input_stochastic(
            y_true, y_score, alpha, beta, contact_cost, sales_cost, direct_selling, commission
        )
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = convex_hull(y_true, y_score)
    true_positive_rates = np.expand_dims(true_positive_rates, axis=1)
    false_positive_rates = np.expand_dims(false_positive_rates, axis=1)
    tpr_diff, fpr_diff = _compute_tpr_fpr_diffs(true_positive_rates, false_positive_rates)

    fpr_coef = contact_cost * negative_class_prob
    tpr_coef = (-direct_selling * sales_cost - contact_cost) * positive_class_prob
    denominator = (direct_selling + (1 - direct_selling) * (1 - commission)) * positive_class_prob

    bounds = _compute_integration_bounds(tpr_coef, fpr_coef, denominator, tpr_diff, fpr_diff)
    cdf_diff = np.diff(st.gamma.cdf(bounds, a=alpha, loc=0, scale=1 / beta), axis=0)
    cdf_1_diff = np.diff(st.gamma.cdf(bounds, a=alpha + 1, loc=0, scale=1 / beta), axis=0)

    cdf_coef = tpr_coef * true_positive_rates - fpr_coef * false_positive_rates
    cdf_1_coef = denominator * true_positive_rates

    expected_profit = (alpha / beta) * (cdf_1_coef * cdf_1_diff).sum(axis=0) + (cdf_coef * cdf_diff).sum(axis=0)

    threshold = (
        cdf_diff * (positive_class_prob * true_positive_rates + negative_class_prob * false_positive_rates)
    ).sum()

    return expected_profit.sum(), threshold


def _compute_integration_bounds(
    tpr_coef: float,
    fpr_coef: float,
    denominator: float,
    tpr_diff: FloatNDArray,
    fpr_diff: FloatNDArray,
) -> FloatNDArray:
    """Compute the integration bounds for the contribution of a new customer."""
    # ignore division by zero warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        clv_bounds = (fpr_coef * fpr_diff - tpr_coef * tpr_diff) / (denominator * tpr_diff)
    # add zero and infinity to bounds
    if clv_bounds.ndim == 2:
        return np.concatenate([
            np.zeros((1, clv_bounds.shape[1])),
            clv_bounds,
            np.full(shape=(1, clv_bounds.shape[1]), fill_value=np.inf),
        ])
    elif clv_bounds.ndim == 1:
        integration_bounds: FloatNDArray = np.concatenate([[0], clv_bounds, [np.inf]]).reshape(-1, 1)
        return integration_bounds
    else:
        raise ValueError(f'Invalid number of dimensions: {clv_bounds.ndim}')


# --- Cost: make_objective_acquisition, AEC classes, expected_cost_loss_acquisition ----------


@overload
def make_objective_acquisition(
    model: Literal['catboost'],
    *,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
) -> tuple['AECObjectiveAcquisition', 'AECMetricAcquisition']: ...


@overload
def make_objective_acquisition(
    model: Literal['xgboost', 'lightgbm'],
    *,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
) -> Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]: ...


def make_objective_acquisition(
    model: Literal['xgboost', 'lightgbm', 'catboost'],
    *,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
) -> (
    tuple['AECObjectiveAcquisition', 'AECMetricAcquisition']
    | Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]]
):
    """Create an objective function for the Expected Cost measure for customer acquisition."""
    if model == 'xgboost':
        objective: Callable[[FloatNDArray, FloatNDArray], tuple[FloatNDArray, FloatNDArray]] = partial(
            _objective,
            contribution=contribution,
            contact_cost=contact_cost,
            sales_cost=sales_cost,
            direct_selling=direct_selling,
            commission=commission,
        )
        update_wrapper(objective, _objective)
    elif model == 'lightgbm':

        def objective(y_true: FloatNDArray, y_score: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
            """Create an objective function for the churn AEC measure."""
            return _objective(
                y_true,
                y_score,
                contribution=contribution,
                contact_cost=contact_cost,
                sales_cost=sales_cost,
                direct_selling=direct_selling,
                commission=commission,
            )

    elif model == 'catboost':
        return (
            AECObjectiveAcquisition(
                contribution=contribution,
                contact_cost=contact_cost,
                sales_cost=sales_cost,
                direct_selling=direct_selling,
                commission=commission,
            ),
            AECMetricAcquisition(
                contribution=contribution,
                contact_cost=contact_cost,
                sales_cost=sales_cost,
                direct_selling=direct_selling,
                commission=commission,
            ),
        )
    else:
        raise ValueError(f"Expected model to be 'xgboost' or 'lightgbm', got {model} instead.")
    return objective


def _objective(
    y_true: FloatNDArray,
    y_score: FloatNDArray,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
) -> tuple[FloatNDArray, FloatNDArray]:
    """Create an objective function for `XGBoostClassifier` for customer acquisition."""
    y_proba = expit(y_score)
    cost = (
        y_true
        * (
            direct_selling * (contact_cost + sales_cost - contribution)
            + (1 - direct_selling) * (contact_cost - (1 - commission) * contribution)
        )
        + (1 - y_true) * contact_cost
    )
    gradient = y_proba * (1 - y_proba) * cost
    hessian = np.abs((1 - 2 * y_proba) * gradient)
    return gradient, hessian


class AECObjectiveAcquisition:
    """AEC acquisition objective for catboost."""

    def __init__(
        self,
        contribution: float = 7_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
    ):
        self.contribution = contribution
        self.sales_cost = sales_cost
        self.contact_cost = contact_cost
        self.direct_selling = direct_selling
        self.commission = commission

    def calc_ders_range(
        self, predictions: Sequence[float], targets: FloatNDArray, weights: Sequence[float]
    ) -> list[tuple[float, float]]:
        """Compute first and second derivative of the loss function wrt the predicted value for each object."""
        y_proba = expit(predictions)
        cost = (
            targets
            * (
                self.direct_selling * (self.contact_cost + self.sales_cost - self.contribution)
                + (1 - self.direct_selling) * (self.contact_cost - (1 - self.commission) * self.contribution)
            )
            + (1 - targets) * self.contact_cost
        )
        gradient = y_proba * (1 - y_proba) * cost
        hessian = np.abs((1 - 2 * y_proba) * gradient)
        return list(zip(-gradient, -hessian, strict=False))


class AECMetricAcquisition:
    """AEC acquisition metric for catboost."""

    def __init__(
        self,
        contribution: float = 7_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
    ):
        self.contribution = contribution
        self.sales_cost = sales_cost
        self.contact_cost = contact_cost
        self.direct_selling = direct_selling
        self.commission = commission

    def is_max_optimal(self) -> bool:
        """Return whether great values of metric are better."""
        return False

    def evaluate(
        self, predictions: Sequence[float], targets: Sequence[float], weights: Sequence[float]
    ) -> tuple[float, float]:
        """Evaluate metric value."""
        y_proba = expit(predictions)
        return expected_cost_loss_acquisition(
            targets,
            y_proba,
            contribution=self.contribution,
            contact_cost=self.contact_cost,
            sales_cost=self.sales_cost,
            direct_selling=self.direct_selling,
            commission=self.commission,
            normalize=True,
            check_input=False,
        ), 1

    def get_final_error(self, error: float, weight: float) -> float:
        """Return final value of metric based on error and weight."""
        return error


def expected_cost_loss_acquisition(
    y_true: FloatArrayLike,
    y_proba: FloatArrayLike,
    *,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    normalize: bool = False,
    check_input: bool = True,
) -> float:
    """Expected cost of a classifier for customer acquisition."""
    if check_input:
        y_true, y_proba = _validate_input_deterministic(
            y_true, y_proba, contribution, contact_cost, sales_cost, direct_selling, commission
        )
    else:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

    costs = (
        y_true
        * y_proba
        * (
            direct_selling * (sales_cost + contact_cost - contribution)
            + (1 - direct_selling) * (contact_cost - (1 - commission) * contribution)
        )
        + (1 - y_true) * y_proba * contact_cost
    )
    if normalize:
        return float(np.mean(costs))
    return float(np.sum(costs))
