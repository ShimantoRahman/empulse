from functools import partial, update_wrapper
from typing import Callable, Union

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike

from empulse.metrics.acquisition._validation import _validate_input_deterministic


def make_objective_acquisition(
        contribution: float = 7_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
) -> Callable[[np.ndarray, Union[xgb.DMatrix, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """
    Create an objective function for the :class:`xgboost:xgboost.XGBClassifier` customer acquisition

    The objective function presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    Parameters
    ----------
    contribution : float, default=7000
        Average contribution of a new customer (``contribution ≥ 0``).

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (``sales_cost ≥ 0``).

    contact_cost : float, default=50
        Average contact cost of targeted leads (``contact_cost ≥ 0``).

    direct_selling : float, default=1
        Fraction of leads sold to directly (``0 ≤ direct_selling ≤ 1``).
        ``direct_selling = 0`` for indirect channel.
        ``direct_selling = 1`` for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermediaries (``0 ≤ commission ≤ 1``).

        .. note::
            The commission is only relevant when there is an indirect channel (``direct_selling < 1``).

    Returns
    -------
    objective : Callable
        A custom objective function for XGBoost.


    Examples
    --------
    .. code-block::  python

        import xgboost as xgb
        from empulse.metrics import make_objective_acquisition

        objective = make_objective_acquisition()
        clf = xgb.XGBClassifier(objective=objective, n_estimators=100, max_depth=3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """

    objective = partial(
        _objective,
        contribution=contribution,
        contact_cost=contact_cost,
        sales_cost=sales_cost,
        direct_selling=direct_selling,
        commission=commission,
    )
    update_wrapper(objective, _objective)
    return objective


def _objective(
        y_pred: np.ndarray,
        dtrain: Union[xgb.DMatrix, np.ndarray],
        contribution: float = 7_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an objective function for `XGBoostClassifier` for customer acquisition

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
    cost = y_true * (direct_selling * (contact_cost + sales_cost - contribution) + (1 - direct_selling) * (
            contact_cost - (1 - commission) * contribution
    )) + (1 - y_true) * contact_cost
    gradient = y_pred * (1 - y_pred) * cost
    hessian = np.abs((1 - 2 * y_pred) * gradient)
    return gradient, hessian


def mpa_cost_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        contribution: float = 7_000,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
        check_input: bool = True,
) -> float:
    """
    Profit-driven cost function for customer acquisition

    The cost function presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    Parameters
    ----------

    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, must be probability estimates.

    contribution : float, default=7000
        Average contribution of a new customer (``contribution ≥ 0``).

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (``sales_cost ≥ 0``).

    contact_cost : float, default=50
        Average contact cost of targeted leads (``contact_cost ≥ 0``).

    direct_selling : float, default=1
        Fraction of leads sold to directly (``0 ≤ direct_selling ≤ 1``).
        ``direct_selling = 0`` for indirect channel.
        ``direct_selling = 1`` for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermediaries (``0 ≤ commission ≤ 1``).

        .. note::
            The commission is only relevant when there is an indirect channel (``direct_selling < 1``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empa_cost : float
        Instance-specific cost function according to the EMPA measure.


    """
    if check_input:
        y_true, y_pred = _validate_input_deterministic(
            y_true,
            y_pred,
            contribution,
            contact_cost,
            sales_cost,
            direct_selling,
            commission
        )
    else:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

    costs = y_true * y_pred * (direct_selling * (sales_cost + contact_cost - contribution) + (1 - direct_selling) * (
            contact_cost - (1 - commission) * contribution
    )) + (1 - y_true) * y_pred * contact_cost
    return float(np.mean(costs))
