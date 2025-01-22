from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Callable, Literal, TypeVar, overload

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.special import expit

if TYPE_CHECKING:
    try:
        from xgboost import DMatrix
    except ImportError:
        try:
            from lightgbm import Dataset
        except ImportError:
            Matrix = TypeVar('Matrix', bound=np.ndarray)
        else:
            Matrix = TypeVar('Matrix', bound=(np.ndarray, Dataset))
    else:
        try:
            from lightgbm import Dataset
        except ImportError:
            Matrix = TypeVar('Matrix', bound=(np.ndarray, DMatrix))
        else:
            Matrix = TypeVar('Matrix', bound=(np.ndarray, DMatrix, Dataset))
else:
    Matrix = TypeVar('Matrix', bound=np.ndarray)


from empulse.metrics.acquisition._validation import _validate_input_deterministic


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
) -> Callable[[np.ndarray, Matrix], tuple[np.ndarray, np.ndarray]]: ...


def make_objective_acquisition(
    model: Literal['xgboost', 'lightgbm', 'catboost'],
    *,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
) -> tuple['AECObjectiveAcquisition', 'AECMetricAcquisition'] | Callable[[ndarray, Matrix], tuple[ndarray, ndarray]]:
    """
    Create an objective function for the Expected Cost measure for customer acquisition.

    The objective function presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    Read more in the :ref:`User Guide <cost_functions>`.

    Parameters
    ----------
    model : {'xgboost', 'lightgbm', 'catboost'}
        The model for which the objective function is created.

        - 'xgboost' : :class:`xgboost:xgboost.XGBClassifier`
        - 'lightgbm' : :class:`lightgbm:lightgbm.LGBMClassifier`
        - 'catboost' : :class:`catboost.CatBoostClassifier`

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

        from xgboost import XGBClassifier
        from empulse.metrics import make_objective_acquisition

        objective = make_objective_acquisition(model='xgboost')
        clf = XGBClassifier(objective=objective, n_estimators=100, max_depth=3)

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    if model == 'xgboost':
        objective = partial(
            _objective,
            contribution=contribution,
            contact_cost=contact_cost,
            sales_cost=sales_cost,
            direct_selling=direct_selling,
            commission=commission,
        )
        update_wrapper(objective, _objective)
    elif model == 'lightgbm':

        def objective(y_pred: np.ndarray, train_data: Matrix) -> tuple[np.ndarray, np.ndarray]:
            """
            Create an objective function for the churn AEC measure.

            Parameters
            ----------
            y_pred : np.ndarray
                Predicted values.
            train_data : xgb.DMatrix or np.ndarray
                Training data.

            Returns
            -------
            gradient  : np.ndarray
                Gradient of the objective function.

            hessian : np.ndarray
                Hessian of the objective function.
            """
            return _objective(
                y_pred,
                train_data,
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
    y_pred: np.ndarray,
    dtrain: Matrix,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an objective function for `XGBoostClassifier` for customer acquisition.

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
    elif hasattr(dtrain, 'get_label'):
        y_true = dtrain.get_label()
    else:
        raise TypeError(f'Expected dtrain to be of type np.ndarray or xgb.DMatrix, got {type(dtrain)} instead.')

    y_pred = 1 / (1 + np.exp(-y_pred))
    cost = (
        y_true
        * (
            direct_selling * (contact_cost + sales_cost - contribution)
            + (1 - direct_selling) * (contact_cost - (1 - commission) * contribution)
        )
        + (1 - y_true) * contact_cost
    )
    gradient = y_pred * (1 - y_pred) * cost
    hessian = np.abs((1 - 2 * y_pred) * gradient)
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

    def calc_ders_range(self, predictions, targets, weights):
        """
        Compute first and second derivative of the loss function with respect to the predicted value for each object.

        Parameters
        ----------
        predictions : indexed container of floats
            Current predictions for each object.

        targets : indexed container of floats
            Target values you provided with the dataset.

        weights : float, optional (default=None)
            Instance weight.

        Returns
        -------
            der1 : list-like object of float
            der2 : list-like object of float

        """
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
        return list(zip(-gradient, -hessian))


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

    def is_max_optimal(self):
        """Return whether great values of metric are better."""
        return False

    def evaluate(self, predictions, targets, weights):
        """
        Evaluate metric value.

        Parameters
        ----------
        predictions : list of indexed containers (containers with only __len__ and __getitem__ defined) of float
            Vectors of approx labels.

        targets : one dimensional indexed container of float
            Vectors of true labels.

        weights : one dimensional indexed container of float, optional (default=None)
            Weight for each instance.

        Returns
        -------
            weighted error : float
            total weight : float

        """
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

    def get_final_error(self, error, weight):
        """
        Return final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        metric value : float

        """
        return error


def expected_cost_loss_acquisition(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    contribution: float = 7_000,
    contact_cost: float = 50,
    sales_cost: float = 500,
    direct_selling: float = 1,
    commission: float = 0.1,
    normalize: bool = False,
    check_input: bool = True,
) -> float:
    """
    Expected cost of a classifier for customer acquisition.

    The cost function presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The company gains a contribution from a successful acquisition.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_proba : 1D array-like, shape=(n_samples,)
        Target probabilities, should lie between 0 and 1.

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

    normalize : bool, default=True
        Normalize the cost function by the number of samples.
        If ``True``, return the average expected cost for customer acquisition.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empa_cost : float
        Instance-specific cost function according to the EMPA measure.


    """  # noqa: D401
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
        return costs.mean()
    return costs.sum()
