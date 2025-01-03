from functools import partial, update_wrapper
from typing import Callable, Union

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike

from empulse.metrics.churn._validation import _validate_input_mpc


def make_objective_churn(
        accept_rate: float = 0.3,
        clv: Union[float, ArrayLike] = 200,
        incentive_fraction: Union[float, ArrayLike] = 0.05,
        contact_cost: float = 15,
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    """
    Create an objective function for the :class:`xgboost:xgboost.XGBClassifier` for customer churn.

    The objective function presumes a situation where identified churners are
    contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer.
    For detailed information, consult the paper [1]_.

    .. seealso::
        :class:`~empulse.models.B2BoostClassifier` : Uses the instance-specific cost function as objective function.

    Parameters
    ----------
    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (``0 < accept_rate < 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
        If ``array``: individualized customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).

    contact_cost : float, default=1
        Constant cost of contact (``contact_cost > 0``).

    Returns
    -------
    objective : Callable
        A custom objective function for :class:`xgboost:xgboost.XGBClassifier`.

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
    objective = partial(
        _objective,
        accept_rate=accept_rate,
        clv=clv,
        incentive_fraction=incentive_fraction,
        contact_cost=contact_cost,
    )
    update_wrapper(objective, _objective)
    return objective


def _objective(
        y_pred: np.ndarray,
        dtrain: Union[xgb.DMatrix, np.ndarray],
        accept_rate: float = 0.3,
        clv: Union[float, ArrayLike] = 200,
        incentive_fraction: Union[float, ArrayLike] = 0.05,
        contact_cost: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Objective function for XGBoost to maximize the profit of a churn model.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    dtrain : xgb.DMatrix or np.ndarray
        Training data.

    Returns
    -------
    gradient  : np.ndarray
        Gradient of the objective function.

    hessian : np.ndarray
        Hessian of the objective function.
    """

    if isinstance(dtrain, np.ndarray):
        y_true = dtrain
    elif isinstance(dtrain, xgb.DMatrix):
        y_true = dtrain.get_label()
    else:
        raise TypeError(f"Expected dtrain to be of type np.ndarray or xgb.DMatrix, got {type(dtrain)} instead.")
    y_pred = 1 / (1 + np.exp(-y_pred))

    incentive_cost = incentive_fraction * clv
    profits = contact_cost + incentive_cost + y_true * (
            accept_rate * incentive_cost - incentive_cost - clv * accept_rate
    )
    gradient = y_pred * (1 - y_pred) * profits
    hessian = np.abs((1 - 2 * y_pred) * gradient)
    return gradient, hessian


def expected_cost_loss_churn(
        y_true: ArrayLike,
        y_proba: ArrayLike,
        *,
        accept_rate: float = 0.3,
        clv: Union[float, ArrayLike] = 200,
        incentive_fraction: Union[float, ArrayLike] = 0.05,
        contact_cost: float = 1,
        normalize: bool = False,
        check_input: bool = True
) -> float:
    """
    Expected cost of a classifier for customer churn.

    The cost function presumes a situation where identified churners are
    contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer.
    For detailed information, consult the paper [1]_.

    .. seealso::
        :class:`~empulse.models.B2BoostClassifier` : Uses the instance-specific cost function as objective function.

    Parameters
    ----------

    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_proba : 1D array-like, shape=(n_samples,)
        Target probabilities, should lie between 0 and 1.

    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (``0 < accept_rate < 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (``clv` > incentive_cost``).
        If ``array``: individualized customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).

    contact_cost : float, default=1
        Constant cost of contact (``contact_cost > 0``).

    normalize : bool, default=False
        Normalize the cost by the number of samples.
        If ``True``, return the average expected cost for customer churn.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

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

    Examples
    --------
    .. code-block::  python

        import xgboost as xgb
        from empulse.metrics import make_objective_churn

        objective = make_objective_churn()
        clf = xgb.XGBClassifier(objective=objective, n_estimators=100, max_depth=3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.

    """
    if check_input:
        y_true, y_proba, clv = _validate_input_mpc(
            y_true,
            y_proba,
            clv=clv,
            accept_rate=accept_rate,
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost
        )
    else:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        clv = np.asarray(clv)

    incentive_cost = incentive_fraction * clv
    profits = y_proba * (contact_cost + incentive_cost + y_true * (
            accept_rate * incentive_cost - incentive_cost - clv * accept_rate
    ))
    if normalize:
        return profits.mean()
    return profits.sum()
