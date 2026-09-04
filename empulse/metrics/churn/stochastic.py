from ..metric.prebuilt_metrics import (
    make_churn_auepc_metric,
    make_churn_empirical_max_profit_metric,
    make_churn_max_profit_metric,
)

empc_score = make_churn_max_profit_metric(stochastic=True)
empc_score.__doc__ = r"""
Expected Maximum Profit Measure for Customer Churn (EMPC).

EMPC presumes a situation where identified churners are contacted and offered an incentive to
remain customers. Only a fraction of churners accepts the incentive offer, this fraction is
described by a :math:`Beta(\alpha, \beta)` distribution. As opposed to
:func:`~empulse.metrics.empb_score`, the incentive cost is a fixed value, rather than a
fraction of the customer lifetime value. For detailed information, consult the paper [1]_.

See :func:`~empulse.metrics.mpc_score` for a deterministic version of this metric.

The EMPC is defined as [1]_:

.. math::

    \int_\gamma CLV (\gamma (1 - \delta) - \phi) \pi_0 F_0(T) - \
    CLV (\delta + \phi) \pi_1 F_1(T) d\gamma

The EMPC requires that the churn class is encoded as 0, and it is NOT interchangeable.
However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

.. rubric:: Methods

``__call__(y_true, y_score, *, alpha=6, beta=14, clv=200, incentive_cost=10, contact_cost=1)``
    Compute the expected maximum profit that can be achieved by a classifier at its optimal
    decision threshold.

``optimal_threshold(y_true, y_score, *, alpha=6, beta=14, clv=200, incentive_cost=10, contact_cost=1)``
    Compute the classification threshold that maximizes the expected profit.

``optimal_rate(y_true, y_score, *, alpha=6, beta=14, clv=200, incentive_cost=10, contact_cost=1)``
    Compute the predicted positive rate (fraction of the customer base that should be
    targeted) at which the maximum expected profit is achieved.

Examples
--------
.. code-block:: python

    from empulse.metrics import empc_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    empc_score(y_true, y_score, alpha=6, beta=14, clv=200, incentive_cost=10, contact_cost=1)

References
----------
.. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
    A Novel Profit Maximizing Metric for Measuring Classification
    Performance of Customer Churn Prediction Models. IEEE Transactions on
    Knowledge and Data Engineering, 25(5), 961-973.
"""

empb_score = make_churn_empirical_max_profit_metric()
empb_score.__doc__ = r"""
Expected Maximum Profit Measure for B2B Customer Churn (EMPB).

EMPB presumes a situation where identified churners are contacted and offered an incentive to
remain customers. Only a fraction of churners accepts the incentive offer, this fraction is
described by a :math:`Beta(\alpha, \beta)` distribution. As opposed to
:func:`~empulse.metrics.empc_score`, the incentive cost is a fraction of the customer lifetime
value, rather than a fixed value. For detailed information, consult the paper [1]_.

``clv`` is required and should be a 1D array-like with one value per sample. The contact cost
is incurred whenever a churner is contacted, regardless of whether they accept the incentive
offer; only the retention benefit net of the incentive cost is contingent on acceptance.
See :func:`~empulse.metrics.auepc_score` for the area under the expected profit curve.

.. rubric:: Methods

``__call__(y_true, y_score, *, clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)``
    Compute the expected maximum profit that can be achieved by a classifier at its optimal
    decision threshold.

``optimal_threshold(y_true, y_score, *, clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)``
    Compute the classification threshold that maximizes the expected profit.

``optimal_rate(y_true, y_score, *, clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)``
    Compute the predicted positive rate (fraction of the customer base that should be
    targeted) at which the maximum expected profit is achieved.

Examples
--------
.. code-block:: python

    from empulse.metrics import empb_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    clv = [100, 200, 150, 300, 120, 250, 180, 220]
    empb_score(y_true, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)

References
----------
.. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
    B2Boost: Instance-dependent profit-driven modelling of B2B churn.
    Annals of Operations Research, 1-27.
"""

auepc_score = make_churn_auepc_metric()
auepc_score.__doc__ = r"""
Area Under the Expected Profit Curve (AUEPC).

Calculate the area under the ratio of the expected profit of the model and the perfect model.
The expected profit is based on the EMPB's definition of profit. AUEPC presumes a situation
where identified churners are contacted and offered an incentive to remain customers. Only a
fraction of churners accepts the incentive offer, this fraction is described by a
:math:`Beta(\alpha, \beta)` distribution. For detailed information, consult the paper [1]_.

``clv`` is required and should be a 1D array-like with one value per sample.
See :func:`~empulse.metrics.empb_score` to instead return the maximum profit.

.. rubric:: Methods

``__call__(y_true, y_score, *, clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)``
    Compute the area under the expected profit curve.

``optimal_threshold(y_true, y_score, *, clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)``
    Compute the classification threshold that maximizes the underlying expected profit
    (see :func:`~empulse.metrics.empb_score`).

``optimal_rate(y_true, y_score, *, clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)``
    Compute the predicted positive rate (fraction of the customer base that should be
    targeted) at which the maximum underlying expected profit is achieved.

Examples
--------
.. code-block:: python

    from empulse.metrics import auepc_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    clv = [100, 200, 150, 300, 120, 250, 180, 220]
    auepc_score(y_true, y_score, clv=clv, alpha=6, beta=14, incentive_fraction=0.05, contact_cost=15)

References
----------
.. [1] Rahman, S., Janssens, B., Bogaert, M. (2025).
    Profit-Driven Pre-Processing in B2B Customer Churn Modeling using Fairness Techniques.
    Journal of Business Research.
"""
