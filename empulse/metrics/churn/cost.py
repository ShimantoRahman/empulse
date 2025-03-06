from collections.abc import Callable, Sequence
from functools import partial, update_wrapper
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

if TYPE_CHECKING:  # pragma: no cover
    try:
        from lightgbm import Dataset
        from xgboost import DMatrix

        Matrix = TypeVar('Matrix', bound=NDArray | DMatrix | Dataset)
    except ImportError:
        Matrix = TypeVar('Matrix', bound=NDArray)  # type: ignore[misc]
else:
    Matrix = TypeVar('Matrix', bound=NDArray)

from empulse.metrics.churn._validation import _validate_input_cost_loss_churn


@overload
def make_objective_churn(
    model: Literal['catboost'],
    *,
    accept_rate: float = 0.3,
    clv: float | NDArray = 200,
    incentive_fraction: float | NDArray = 0.05,
    contact_cost: float = 15,
) -> tuple['AECObjectiveChurn', 'AECMetricChurn']: ...


@overload
def make_objective_churn(
    model: Literal['xgboost', 'lightgbm'],
    *,
    accept_rate: float = 0.3,
    clv: float | NDArray = 200,
    incentive_fraction: float | NDArray = 0.05,
    contact_cost: float = 15,
) -> Callable[[np.ndarray, Matrix], tuple[np.ndarray, np.ndarray]]: ...


def make_objective_churn(
    model: Literal['xgboost', 'lightgbm', 'catboost'],
    *,
    accept_rate: float = 0.3,
    clv: float | NDArray = 200,
    incentive_fraction: float | NDArray = 0.05,
    contact_cost: float = 15,
) -> Callable[[np.ndarray, Matrix], tuple[np.ndarray, np.ndarray]] | tuple['AECObjectiveChurn', 'AECMetricChurn']:
    """
    Create an objective function for the Expected Cost measure for customer churn.

    The objective function presumes a situation where identified churners are
    contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer.
    For detailed information, consult the paper [1]_.

    Read more in the :ref:`User Guide <cost_functions>`.

    .. seealso::
        :class:`~empulse.models.B2BoostClassifier` : Uses the instance-specific cost function as objective function.

    Parameters
    ----------
    model : {'xgboost', 'lightgbm', 'catboost'}
        The model for which the objective function is created.

        - 'xgboost' : :class:`xgboost:xgboost.XGBClassifier`
        - 'lightgbm' : :class:`lightgbm:lightgbm.LGBMClassifier`
        - 'catboost' : :class:`catboost.CatBoostClassifier`

    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (``0 < accept_rate < 1``).

    clv : float or 1D array, shape=(n_samples), default=200
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

    Examples
    --------

    .. code-block::  python

        from xgboost import XGBClassifier
        from empulse.metrics import make_objective_churn

        objective = make_objective_churn(model='xgboost')
        clf = XGBClassifier(objective=objective, n_estimators=100, max_depth=3)

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
    if model == 'xgboost':
        objective: Callable[[np.ndarray, Matrix], tuple[np.ndarray, np.ndarray]] = partial(
            _objective,
            accept_rate=accept_rate,
            clv=clv,
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost,
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
    y_pred: np.ndarray,
    dtrain: Matrix,
    accept_rate: float = 0.3,
    clv: float | NDArray = 200,
    incentive_fraction: float | NDArray = 0.05,
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
    elif hasattr(dtrain, 'get_label'):
        y_true = dtrain.get_label()  # type: ignore[assignment]
    else:
        raise TypeError(f'Expected dtrain to be of type np.ndarray or xgb.DMatrix, got {type(dtrain)} instead.')
    y_pred = 1 / (1 + np.exp(-y_pred))

    incentive_cost = incentive_fraction * clv
    profits = (
        contact_cost + incentive_cost + y_true * (accept_rate * incentive_cost - incentive_cost - clv * accept_rate)
    )
    gradient = y_pred * (1 - y_pred) * profits
    hessian = np.abs((1 - 2 * y_pred) * gradient)
    return gradient, hessian


class AECObjectiveChurn:
    """AEC churn objective for catboost."""

    def __init__(
        self,
        accept_rate: float = 0.3,
        clv: float | NDArray = 200,
        incentive_fraction: float | NDArray = 0.05,
        contact_cost: float = 1,
    ):
        self.accept_rate = accept_rate
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost

    def calc_ders_range(
        self, predictions: Sequence[float], targets: NDArray, weights: NDArray
    ) -> list[tuple[float, float]]:
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
        clv: float | NDArray = 200,
        incentive_fraction: float | NDArray = 0.05,
        contact_cost: float = 1,
    ):
        self.accept_rate = accept_rate
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost

    def is_max_optimal(self) -> bool:
        """Return whether great values of metric are better."""
        return False

    def evaluate(self, predictions: Sequence[float], targets: NDArray, weights: NDArray) -> tuple[float, float]:
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


def expected_cost_loss_churn(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    accept_rate: float = 0.3,
    clv: float | ArrayLike = 200,
    incentive_fraction: float | ArrayLike = 0.05,
    contact_cost: float = 1,
    normalize: bool = False,
    check_input: bool = True,
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

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.

    """  # noqa: D401
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
        return profits.mean()
    return profits.sum()
