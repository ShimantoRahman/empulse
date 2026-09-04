from ..metric.prebuilt_metrics import make_acquisition_max_profit_metric

mpa_score = make_acquisition_max_profit_metric(stochastic=False)
mpa_score.__name__ = 'mpa_score'
mpa_score.__doc__ = r"""
Maximum Profit measure for customer Acquisition (MPA).

MPA presumes a situation where leads are targeted either directly or indirectly. Directly
targeted leads are contacted and handled by the internal sales team. Indirectly targeted
leads are contacted and then referred to intermediaries, which receive a commission. The
company gains a contribution from a successful acquisition.

See :func:`~empulse.metrics.empa_score` for a stochastic version of this metric.

.. rubric:: Methods

``__call__(y_true, y_score, *, contribution=8000, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)``
    Compute the maximum profit that can be achieved by a classifier at its optimal decision threshold.

``optimal_threshold(y_true, y_score, *, contribution=8000, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)``
    Compute the classification threshold that maximizes the profit.

``optimal_rate(y_true, y_score, *, contribution=8000, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)``
    Compute the predicted positive rate (fraction of leads that should be targeted) at which
    the maximum profit is achieved.

Examples
--------
.. code-block:: python

    from empulse.metrics import mpa_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    mpa_score(y_true, y_score, direct_selling=1)
"""
