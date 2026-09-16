"""Shared cost-criterion setup for CSTreeClassifier and CSForestClassifier."""

import numpy as np

from ...._types import FloatNDArray
from .cost_impurity import CostImpurity, EntropyCostImpurity, GiniCostImpurity


def _as_array_cost(cost: FloatNDArray | float) -> FloatNDArray:
    """Return *cost* as a flat float64 array for `set_array_costs()`, or empty if it's a scalar."""
    if isinstance(cost, np.ndarray):
        array: FloatNDArray = cost.reshape(-1).astype(np.float64)
        return array
    return np.array([], dtype=np.float64)


def build_cost_criterion(
    criterion: str | CostImpurity,
    tp_cost: FloatNDArray | float,
    tn_cost: FloatNDArray | float,
    fn_cost: FloatNDArray | float,
    fp_cost: FloatNDArray | float,
    n_samples: int,
) -> CostImpurity:
    """
    Validate cost shapes, offset negative costs, and return a configured cost-sensitive criterion.

    Parameters
    ----------
    criterion : {'cost', 'gini', 'entropy', 'log_loss'} or CostImpurity
        Which impurity weighting to use, or a pre-built (unconfigured) custom
        :class:`CostImpurity` instance. A custom instance passed in is never reused directly: a
        fresh instance of the same type is constructed instead, since this function always
        overwrites all cost state - reusing (or deep-copying) the caller's instance would either
        mutate an ``__init__`` parameter or crash on its uninitialized C-level buffers.
    tp_cost, tn_cost, fn_cost, fp_cost : float or ndarray of shape (n_samples,)
        The (already resolved) costs/benefits for each outcome.
    n_samples : int
        Number of training samples, used to validate array-valued cost shapes.

    Returns
    -------
    CostImpurity
        A criterion instance with ``set_costs()``/``set_array_costs()`` already applied.
    """
    for name, cost in zip(
        ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=True
    ):
        if isinstance(cost, np.ndarray) and cost.shape[0] != n_samples:
            raise ValueError(f'{name} has shape {cost.shape}, but should have shape ({n_samples},)')

    min_cost = float('inf')
    for cost in (tp_cost, tn_cost, fn_cost, fp_cost):
        min_cost = min(min_cost, float(np.min(cost)) if isinstance(cost, np.ndarray) else float(cost))

    # Apply offset if minimum is negative so that node_impurity >= 0 (required by sklearn)
    cost_offset = -min_cost if min_cost < 0 else 0.0
    if cost_offset > 0:
        tp_cost = tp_cost.copy() + cost_offset if isinstance(tp_cost, np.ndarray) else tp_cost + cost_offset
        tn_cost = tn_cost.copy() + cost_offset if isinstance(tn_cost, np.ndarray) else tn_cost + cost_offset
        fn_cost = fn_cost.copy() + cost_offset if isinstance(fn_cost, np.ndarray) else fn_cost + cost_offset
        fp_cost = fp_cost.copy() + cost_offset if isinstance(fp_cost, np.ndarray) else fp_cost + cost_offset

    if criterion == 'cost':
        criterion_: CostImpurity = CostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    elif criterion == 'gini':
        criterion_ = GiniCostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    elif criterion in {'entropy', 'log_loss'}:
        criterion_ = EntropyCostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    elif isinstance(criterion, CostImpurity):
        criterion_ = type(criterion)(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    else:
        raise ValueError(f'Unknown criterion: {criterion}')

    criterion_.set_costs(
        tp_cost=tp_cost if not isinstance(tp_cost, np.ndarray) else 0.0,
        tn_cost=tn_cost if not isinstance(tn_cost, np.ndarray) else 0.0,
        fp_cost=fp_cost if not isinstance(fp_cost, np.ndarray) else 0.0,
        fn_cost=fn_cost if not isinstance(fn_cost, np.ndarray) else 0.0,
    )
    criterion_.set_array_costs(
        tp_cost=_as_array_cost(tp_cost),
        tn_cost=_as_array_cost(tn_cost),
        fp_cost=_as_array_cost(fp_cost),
        fn_cost=_as_array_cost(fn_cost),
        n_samples=n_samples,
    )

    return criterion_
