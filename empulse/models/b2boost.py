from typing import Any, Optional, Union

from numpy.typing import ArrayLike
from xgboost import XGBClassifier

from ..metrics import create_objective_churn


class B2BoostClassifier(XGBClassifier):
    """
    `XGBoostClassifier` [1]_ wrapper with instance-specific cost function for customer churn [2]_.

    Parameters
    ----------
    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (0 < `accept_rate` < 1).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (`clv` > `incentive_cost`).
        If ``array``: individualized customer lifetime value of each customer when retained
        (mean(`clv`) > `incentive_cost`).

    incentive_cost : float, default=10
        Constant cost of retention offer (`incentive_cost` > 0).

    contact_cost : float, default=1
        Constant cost of contact (`contact_cost` > 0).

    kwargs : dict
        Other parameters passed to `XGBClassifier` init.

    Notes
    -----
    The instance-specific cost function for customer churn is defined as [2]_:

    .. math:: C(s_i) = y_i[s_i(f-\\gamma (1-\\delta )CLV_i] + (1-y_i)[s_i(\\delta CLV_i + f)]

    The measure requires that the churn class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    .. seealso::
        :func:`~empulse.metrics.create_objective_churn` : Creates the instance-specific cost function
        for customer churn.

    References
    ----------
    .. [1] https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
    .. [2] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    def __init__(
            self,
            *,
            clv: Union[float, ArrayLike] = 200,
            d: float = 10,
            f: float = 1,
            gamma: float = 0.3,
            use_label_encoder: Optional[bool] = None,
            **kwargs: Any,
    ) -> None:
        objective = create_objective_churn(clv=clv, incentive_cost=d, contact_cost=f, accept_rate=gamma)
        super().__init__(objective=objective, use_label_encoder=use_label_encoder, **kwargs)
