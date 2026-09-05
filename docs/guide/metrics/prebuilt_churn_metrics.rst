.. _prebuilt_churn_metrics:

======================
Customer Churn Metrics
======================

Customer churn is the use case value-driven classification was originally developed for, and
Empulse ships a family of ready-made metrics for it. They all describe the same business
situation: you predict which customers are about to leave, contact the ones you predict will
churn, and offer them a retention incentive.

.. note::
    These names are **prebuilt** :class:`~empulse.metrics.Metric` instances, not functions.
    Earlier versions of Empulse exposed hand-written ``empc``, ``mpc`` and ``empb`` functions
    which returned a ``(score, threshold)`` tuple; those have been removed. The score is now
    obtained by calling the metric, and the threshold by its
    :meth:`~empulse.metrics.Metric.optimal_rate` or
    :meth:`~empulse.metrics.Metric.optimal_threshold` method.

The Cost-Benefit Matrix
=======================

Contacting a customer costs :math:`f` whether or not the offer is accepted. A contacted customer
accepts the retention offer with probability :math:`\gamma`, in which case you keep their customer
lifetime value :math:`CLV` but pay the incentive :math:`d`. Customers you do not contact cost you
nothing extra, so the two "predicted negative" outcomes are zero.

.. list-table::

    * -
      - Actual churner :math:`y_i = 1`
      - Actual non-churner :math:`y_i = 0`
    * - Predicted churner :math:`\hat{y}_i = 1`
      - ``tp_benefit`` :math:`= \gamma (CLV - d - f) - (1 - \gamma) f`
      - ``fp_cost`` :math:`= d + f`
    * - Predicted non-churner :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= 0`
      - ``tn_benefit`` :math:`= 0`

The metrics differ in how they treat :math:`\gamma`, and in whether the incentive is a fixed
amount or a fraction of the customer's value.

.. list-table::
    :widths: 25 20 25 30
    :header-rows: 1

    * - Metric
      - Strategy
      - Acceptance rate :math:`\gamma`
      - Incentive
    * - :func:`~empulse.metrics.mpc_score`
      - :class:`~empulse.metrics.MaxProfit`
      - Fixed (``accept_rate``)
      - Fixed amount (``incentive_cost``)
    * - :func:`~empulse.metrics.empc_score`
      - :class:`~empulse.metrics.MaxProfit`
      - ``Beta(alpha, beta)``
      - Fixed amount (``incentive_cost``)
    * - :func:`~empulse.metrics.empb_score`
      - :class:`~empulse.metrics.EmpiricalMaxProfit`
      - ``Beta(alpha, beta)``
      - Fraction of CLV (``incentive_fraction``)
    * - :func:`~empulse.metrics.auepc_score`
      - :class:`~empulse.metrics.AUEPC`
      - ``Beta(alpha, beta)``
      - Fraction of CLV (``incentive_fraction``)
    * - :func:`~empulse.metrics.expected_cost_loss_churn`
      - :class:`~empulse.metrics.Cost`
      - Fixed (``accept_rate``)
      - Fraction of CLV (``incentive_fraction``)

Maximum Profit for Customer Churn (MPC)
=======================================

:func:`~empulse.metrics.mpc_score` treats the acceptance rate as a single known number [1]_. It
reports the profit per customer at the profit-maximising cut-off.

Defaults: ``accept_rate=0.3``, ``clv=200``, ``incentive_cost=10``, ``contact_cost=1``.

.. code-block:: python

    import numpy as np
    from empulse.metrics import mpc_score

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    profit = mpc_score(y_true, y_score)
    profit_custom = mpc_score(y_true, y_score, accept_rate=0.5, clv=500, incentive_cost=25)

Expected Maximum Profit for Customer Churn (EMPC)
=================================================

In practice you rarely know the acceptance rate exactly. :func:`~empulse.metrics.empc_score`
models :math:`\gamma` as a ``Beta(alpha, beta)`` random variable and integrates over it, giving the
*expected* maximum profit [2]_.

Defaults: ``alpha=6``, ``beta=14`` (a mean acceptance rate of :math:`6/(6+14) = 0.3`),
``clv=200``, ``incentive_cost=10``, ``contact_cost=1``.

.. code-block:: python

    from empulse.metrics import empc_score

    expected_profit = empc_score(y_true, y_score)

``clv`` may be instance-dependent, which is the usual reason to reach for this metric: customers
are not equally valuable, and targeting should follow value rather than churn probability alone.

.. code-block:: python

    clv = np.array([100, 250, 80, 900, 120, 400, 60, 300])

    expected_profit_per_customer = empc_score(y_true, y_score, clv=clv)

How many customers should you target?
-------------------------------------

This is the question the maximum-profit family exists to answer.
:meth:`~empulse.metrics.Metric.optimal_rate` returns the fraction of the customer base to contact,
and :func:`~empulse.metrics.classification_threshold` converts that fraction into a probability
cut-off you can hand to a classifier.

.. code-block:: python

    from empulse.metrics import classification_threshold

    target_fraction = empc_score.optimal_rate(y_true, y_score, clv=clv)
    threshold = classification_threshold(y_true, y_score, customer_threshold=target_fraction)

.. note::
    The removed ``empc``/``mpc`` functions returned this fraction alongside the score as a tuple.
    Use ``.optimal_rate(...)`` instead, and note it accepts the same parameters as the metric.

Expected Maximum Profit with a fractional incentive (EMPB)
==========================================================

:func:`~empulse.metrics.empb_score` expresses the retention offer as a *fraction* of the
customer's value rather than a flat amount, which fits discount-based campaigns better. It uses
the :class:`~empulse.metrics.EmpiricalMaxProfit` strategy, computing the profit from the empirical
convex hull of the ROC curve instead of a closed-form integral.

Defaults: ``alpha=6``, ``beta=14``, ``incentive_fraction=0.05``, ``contact_cost=15``.
``clv`` has no default and must be supplied.

.. code-block:: python

    from empulse.metrics import empb_score

    profit_b = empb_score(y_true, y_score, clv=clv)
    profit_b_custom = empb_score(y_true, y_score, clv=clv, incentive_fraction=0.10, contact_cost=5)

Area Under the Expected Profit Curve (AUEPC)
============================================

:func:`~empulse.metrics.auepc_score` uses the same cost matrix as
:func:`~empulse.metrics.empb_score`, but summarises the whole profit curve rather than only its
peak. Where EMPB asks "how much can I make at the best cut-off?", AUEPC asks "how good is this
ranking across all cut-offs?" — making it the more robust choice when the operating point is not
yet fixed.

.. code-block:: python

    from empulse.metrics import auepc_score

    ranking_quality = auepc_score(y_true, y_score, clv=clv)

Expected Cost Loss for Churn
============================

The metrics above maximise profit and expect *ranking scores*. When you instead want to
**minimise cost** and already have calibrated probabilities,
:func:`~empulse.metrics.expected_cost_loss_churn` applies the same business model through the
:class:`~empulse.metrics.Cost` strategy. Lower is better, and it always returns the mean cost per
customer.

Defaults: ``accept_rate=0.3``, ``clv=200``, ``incentive_fraction=0.05``, ``contact_cost=1``.

.. code-block:: python

    from empulse.metrics import expected_cost_loss_churn

    y_proba = np.array([0.05, 0.95, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65])

    mean_cost = expected_cost_loss_churn(y_true, y_proba, clv=clv)

Using a churn metric to train a model
=====================================

Because these are :class:`~empulse.metrics.Metric` instances, they can be passed straight to a
model as its ``loss``, so the model optimises the business objective during training rather than
only being scored on it afterwards.

.. code-block:: python

    from empulse.metrics import expected_cost_loss_churn
    from empulse.models import CSBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    clv_train = np.random.default_rng(42).uniform(100, 500, size=y.shape[0])

    model = CSBoostClassifier(loss=expected_cost_loss_churn)
    model.fit(X, y, clv=clv_train)

.. note::
    Not every strategy is supported by every model. See :ref:`metric_class_in_model` for the
    compatibility table — in particular, the :class:`~empulse.metrics.MaxProfit` strategy used by
    ``empc_score``/``mpc_score`` is only supported by a subset of models.

See also
========

- :ref:`choosing_metric` — how to pick between cost, savings and profit metrics.
- :ref:`user_defined_value_metric` — build your own metric when none of these fit.
- :ref:`prebuilt_acquisition_metrics` and :ref:`prebuilt_credit_scoring_metrics` — the equivalent
  families for other use cases.

References
==========

.. [1] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
       New insights into churn prediction in the telecommunication sector:
       A profit driven data mining approach.
       *European Journal of Operational Research*, 218(1), 211-229.

.. [2] Verbraken, T., Verbeke, W., & Baesens, B. (2013).
       A novel profit maximizing metric for measuring classification performance of customer
       churn prediction models. *IEEE Transactions on Knowledge and Data Engineering*,
       25(5), 961-973.
