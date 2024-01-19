from functools import partial
from typing import Callable, Union

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike


def create_objective_churn(
        accept_rate: float = 0.3,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: Union[float, ArrayLike] = 10,
        contact_cost: float = 1,
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    """
    Create a custom objective function for XGBoost to maximize the profit of a churn model.

    Parameters
    ----------
    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (0 < gamma < 1).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If clv is a float: constant customer lifetime value per retained customer (clv > d).
        If clv is an array: individualized customer lifetime value of each customer when retained (mean(clv) > d).

    incentive_cost : float, default=10
        Constant cost of retention offer (d > 0).

    contact_cost : float, default=1
        Constant cost of contact (f > 0).

    Returns
    -------
    objective : Callable
        A custom objective function for XGBoost.

    """
    return partial(
        _objective,
        accept_rate=accept_rate,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
    )


def _objective(
        y_pred: np.ndarray,
        dtrain: Union[xgb.DMatrix, np.ndarray],
        accept_rate: float = 0.3,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: Union[float, ArrayLike] = 10,
        contact_cost: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Objective function for XGBoost to maximize the profit of a churn model.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    dtrain : xgb.DMatrix
        Training data.

    Returns
    -------
    gradient, hessian : tuple[np.ndarray, np.ndarray]
        Gradient and hessian of the objective function.
    """

    if isinstance(dtrain, np.ndarray):
        y_true = dtrain
    elif isinstance(dtrain, xgb.DMatrix):
        y_true = dtrain.get_label()
    else:
        raise TypeError(f"Expected dtrain to be of type np.ndarray or xgb.DMatrix, got {type(dtrain)} instead.")
    y_pred = 1 / (1 + np.exp(-y_pred))

    delta = incentive_cost / clv
    profits = contact_cost + (delta * clv) + y_true * (clv * (accept_rate * delta - delta - accept_rate))
    gradient = y_pred * (1 - y_pred) * profits
    hessian = abs((1 - 2 * y_pred) * gradient)
    return gradient, hessian


def mpc_cost_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        accept_rate: float = 0.3,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: Union[float, ArrayLike] = 10,
        contact_cost: float = 1,
) -> float:
    """
    Parameters
    ----------

    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, must be probability estimates.

    clv : float or 1D array-like, shape=(n_samples), default=200
        If clv is a float: constant customer lifetime value per retained customer (clv > d).
        If clv is an array: individualized customer lifetime value of each customer when retained (mean(clv) > d).

    incentive_cost : float, default=10
        Constant cost of retention offer (d > 0).

    contact_cost : float, default=1
        Constant cost of contact (f > 0).

    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (0 < gamma < 1).

    Returns
    -------
    empc_cost : float
        Instance-specific cost function according to the EMPC measure.

    Notes
    -----
    The instance-specific cost function for customer churn is defined as [1]_:

    .. math:: C(s_i) = y_i[s_i(f-\\gamma (1-\\delta )CLV_i] + (1-y_i)[s_i(\\delta CLV_i + f)]

    The measure requires that the churn class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    delta = incentive_cost / clv
    profits = y_pred * (contact_cost + (delta * clv) + y_true * (clv * (accept_rate * delta - delta - accept_rate)))
    return float(np.mean(profits))
