"""Turn the four outcome costs into the per-sample cost records the tree builder reads."""

import numpy as np

from ...._types import FloatNDArray, IntNDArray

CRITERIA = {'cost': 0, 'gini': 1, 'entropy': 2, 'log_loss': 2}


def criterion_kind(criterion: str) -> int:
    """Return the builder's code for a ``criterion`` name."""
    try:
        return CRITERIA[criterion]
    except KeyError:
        raise ValueError(f'Unknown criterion: {criterion}') from None


def cost_records(
    y: IntNDArray,
    tp_cost: FloatNDArray | float,
    tn_cost: FloatNDArray | float,
    fn_cost: FloatNDArray | float,
    fp_cost: FloatNDArray | float,
) -> FloatNDArray:
    """
    Return one record ``[a, b, y, 0]`` per sample, for ``Splitter``.

    ``a`` is what predicting the sample positive costs (``tp_cost`` for a positive, ``fp_cost`` for a
    negative), ``b`` what predicting it negative costs (``fn_cost`` or ``tn_cost``). Array costs must
    have one entry per sample.

    When any cost is negative (a benefit), all four are shifted up by the same amount so the smallest
    is zero. That changes no decision, since every candidate prediction gains the same constant, but
    keeps the impurities non-negative, which the gini and entropy weightings and the zero-impurity
    stopping rule assume.
    """
    y = np.asarray(y).reshape(-1)
    n_samples = y.shape[0]
    costs = {'tp_cost': tp_cost, 'tn_cost': tn_cost, 'fn_cost': fn_cost, 'fp_cost': fp_cost}
    for name, cost in costs.items():
        if isinstance(cost, np.ndarray) and cost.shape[0] != n_samples:
            raise ValueError(f'{name} has shape {cost.shape}, but should have shape ({n_samples},)')

    min_cost = min(float(np.min(cost)) if isinstance(cost, np.ndarray) else float(cost) for cost in costs.values())
    offset = -min_cost if min_cost < 0 else 0.0
    if offset > 0:
        costs = {name: cost + offset for name, cost in costs.items()}

    def per_sample(cost: FloatNDArray | float) -> FloatNDArray:
        return np.broadcast_to(np.asarray(cost, dtype=np.float64).reshape(-1), (n_samples,))

    is_positive = y == 1
    records = np.zeros((n_samples, 4), dtype=np.float64)
    records[:, 0] = np.where(is_positive, per_sample(costs['tp_cost']), per_sample(costs['fp_cost']))
    records[:, 1] = np.where(is_positive, per_sample(costs['fn_cost']), per_sample(costs['tn_cost']))
    records[:, 2] = is_positive
    return records
