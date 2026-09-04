from .metric.prebuilt_metrics import make_generic_max_profit_metric

max_profit_score = make_generic_max_profit_metric()
max_profit_score.__name__ = 'max_profit_score'
max_profit_score.__doc__ = r"""
Maximum Profit Measure (MP).

A generic :class:`~empulse.metrics.Metric` built from the :class:`~empulse.metrics.MaxProfit`
strategy on a plain cost matrix, accepting class- or instance-dependent ``tp_cost``,
``tn_cost``, ``fp_cost``, and ``fn_cost`` parameters (all costs; a benefit is a negative cost).

.. note::
   This replaces the previous native ``max_profit``/``max_profit_score`` functions, which took
   ``tp_benefit``/``tn_benefit`` (instead of ``tp_cost``/``tn_cost``). ``tp_cost = -tp_benefit``
   and ``tn_cost = -tn_benefit``; ``fp_cost``/``fn_cost`` are unchanged.

The MP is defined as [1]_:

.. math::

    \text{MP} = b_0 \pi_0 F_0(T) + b_1 \pi_1 (1 - F_1(T)) - c_0 \pi_0 (1 - F_0(T)) - c_1 F_1(T)

where :math:`T` is the threshold at which the maximum profit is achieved.

.. rubric:: Methods

``__call__(y_true, y_score, *, tp_cost=0.0, tn_cost=0.0, fp_cost=0.0, fn_cost=0.0)``
    Compute the maximum profit that can be achieved by a classifier at its optimal decision threshold.

``optimal_threshold(y_true, y_score, *, tp_cost=0.0, tn_cost=0.0, fp_cost=0.0, fn_cost=0.0)``
    Compute the classification threshold that maximizes the profit.

``optimal_rate(y_true, y_score, *, tp_cost=0.0, tn_cost=0.0, fp_cost=0.0, fn_cost=0.0)``
    Compute the predicted positive rate at which the maximum profit is achieved.

Examples
--------
Reimplement MPC:

.. code-block:: python

    from empulse.metrics import max_profit_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    clv = 200
    d = 10
    f = 1
    gamma = 0.3
    tp_cost = -(clv * (gamma * (1 - (d / clv)) - (f / clv)))
    fp_cost = d + f

    max_profit_score(y_true, y_score, tp_cost=tp_cost, fp_cost=fp_cost)

References
----------
.. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
    A Novel Profit Maximizing Metric for Measuring Classification
    Performance of Customer Churn Prediction Models. IEEE Transactions on
    Knowledge and Data Engineering, 25(5), 961-973.
"""
