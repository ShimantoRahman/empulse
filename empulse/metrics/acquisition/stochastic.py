from ..metric.prebuilt_metrics import make_acquisition_max_profit_metric

empa_score = make_acquisition_max_profit_metric(stochastic=True)
empa_score.__doc__ = r"""
Expected Maximum Profit measure for customer Acquisition (EMPA).

EMPA presumes a situation where leads are targeted either directly or indirectly. Directly
targeted leads are contacted and handled by the internal sales team. Indirectly targeted
leads are contacted and then referred to intermediaries, which receive a commission. The
contribution of a successful acquisition is modeled as a Gamma distribution.

.. note::
    Unlike the retired native ``empa``/``empa_score`` functions, ``beta`` here is the *scale*
    of the Gamma distribution (mean = ``alpha * beta``), not its *rate* (mean =
    ``alpha / beta``). This is a consequence of how :class:`~empulse.metrics.MaxProfit`'s
    stochastic-integration engine parameterizes distributions. The default value has been
    adjusted (``1 / 0.0015``) so that calling with no arguments reproduces the same result as
    before.

See :func:`~empulse.metrics.mpa_score` for a deterministic version of this metric.

.. rubric:: Methods

``__call__(y_true, y_score, *, alpha=12, beta=1/0.0015, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)``
    Compute the expected maximum profit that can be achieved by a classifier at its optimal
    decision threshold.

``optimal_threshold(y_true, y_score, *, alpha=12, beta=1/0.0015, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)``
    Compute the classification threshold that maximizes the expected profit.

``optimal_rate(y_true, y_score, *, alpha=12, beta=1/0.0015, contact_cost=50, sales_cost=500, direct_selling=1, commission=0.1)``
    Compute the predicted positive rate (fraction of leads that should be targeted) at which
    the maximum expected profit is achieved.

Examples
--------
.. code-block:: python

    from empulse.metrics import empa_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    empa_score(y_true, y_score, direct_selling=1)
"""
