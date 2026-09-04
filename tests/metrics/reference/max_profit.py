"""
Reference (native/legacy) implementation of the generic Maximum Profit measure.

This is the hand-written math implementation that used to live in
``empulse.metrics.max_profit`` before it was refactored to build ``max_profit_score`` from a
:class:`~empulse.metrics.Metric` instance instead. It is kept here, unchanged, purely as ground
truth to numerically verify the new prebuilt metric against.

Note that this reference version parameterizes true positive/negative outcomes by *benefit*
(``tp_benefit``/``tn_benefit``), whereas the new ``max_profit_score`` parameterizes them by *cost*
(``tp_cost``/``tn_cost``), i.e. ``tp_cost = -tp_benefit`` and ``tn_cost = -tn_benefit``.
``fp_cost``/``fn_cost`` are unchanged between the two.
"""

import numpy as np
from numpy.typing import ArrayLike

from empulse.metrics._cy_convex_hull import convex_hull


def max_profit_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    fn_cost: float = 0.0,
    fp_cost: float = 0.0,
) -> float:
    """Maximum Profit measure (MP), only returning the MP score."""
    return max_profit(y_true, y_score, tp_benefit=tp_benefit, tn_benefit=tn_benefit, fn_cost=fn_cost, fp_cost=fp_cost)[
        0
    ]


def max_profit(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    fn_cost: float = 0.0,
    fp_cost: float = 0.0,
) -> tuple[float, float]:
    """Maximum Profit measure (MP)."""
    y_true = np.asarray(y_true, dtype=np.int32, order='C')
    y_score = np.asarray(y_score, dtype=np.float64, order='C')

    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0

    f0, f1 = convex_hull(y_true, y_score)

    profits = (tp_benefit + fn_cost) * pi0 * f0 - (tn_benefit + fp_cost) * pi1 * f1 + tn_benefit * pi1 - fn_cost * pi0
    best_index = np.argmax(profits)
    maximum_profit = float(profits[best_index])
    customer_threshold = float(f0[best_index] * pi0 + f1[best_index] * pi1)
    return maximum_profit, customer_threshold
