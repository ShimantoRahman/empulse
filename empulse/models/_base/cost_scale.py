from typing import Any

import numpy as np

from ..._types import IntNDArray
from ...metrics import BaseMetric, objective_scale_from_costs


def decision_cost_scale(loss: BaseMetric, y: IntNDArray, **loss_params: Any) -> float:
    """
    Return the average cost of deciding a training sample wrongly instead of rightly.

    For a positive that is ``|fn_cost - tp_cost|``, for a negative ``|fp_cost - tn_cost|``. Dividing a
    training objective by it makes hyperparameters with absolute units (a booster's minimum child
    weight, a parsimony penalty, a convergence tolerance) mean the same whatever currency the costs
    are in, and adding a constant to the costs of one class leaves it unchanged. Stochastic costs
    are replaced by their mean. Falls back to 1 when no decision costs anything, and is 1 for a
    loss whose value is a pure number rather than an amount of money, such as AUEPC.
    """
    if loss._is_unitless:
        return 1.0
    fp_cost, fn_cost, tp_cost, tn_cost = loss._evaluate_costs(replace_stochastic=True, **loss_params)
    return objective_scale_from_costs(y, -np.asarray(tp_cost), -np.asarray(tn_cost), fp_cost, fn_cost)
