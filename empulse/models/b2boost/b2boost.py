from typing import Any, Optional, Union

from numpy.typing import ArrayLike
from xgboost import XGBClassifier

from ...metrics.churn.cost import create_objective_churn


class B2BoostClassifier(XGBClassifier):
    """
    XGBoostClassifier wrapper with value-driven cost function.

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

    kwargs : dict
        Other parameters passed to XGBClassifier.

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
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
