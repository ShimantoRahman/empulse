.. _prebuilt_credit_scoring_metrics:

======================
Credit Scoring Metrics
======================

In credit scoring the classifier decides which loan applications to reject. The two errors are
very unlike each other: rejecting a good applicant forfeits the interest you would have earned,
while accepting a bad one loses part of the principal. Empulse ships two ready-made metrics for
this trade-off, following the profit-based framework of Verbraken et al. [1]_.

.. note::
    These names are **prebuilt** :class:`~empulse.metrics.Metric` and
    :class:`~empulse.metrics.MixtureMetric` instances, not functions. Earlier versions of Empulse
    exposed hand-written ``empcs`` and ``mpcs`` functions returning a ``(score, threshold)``
    tuple; those have been removed. Call the metric for the score, and use
    :meth:`~empulse.metrics.Metric.optimal_rate` for the fraction of applicants to reject.

The Cost-Benefit Matrix
=======================

Here the positive class is a **defaulter**, so a "predicted positive" means the application is
rejected. Both quantities are expressed as fractions of the loan amount, which makes the metric
scale-free:

- :math:`\lambda` (``loan_lost_rate``) — the fraction of the principal lost when a loan defaults.
  Correctly rejecting a defaulter saves you this.
- :math:`ROI` (``roi``) — the return you would have earned on a good loan. Wrongly rejecting a good
  applicant costs you this.

.. list-table::

    * -
      - Actual defaulter :math:`y_i = 1`
      - Actual non-defaulter :math:`y_i = 0`
    * - Rejected :math:`\hat{y}_i = 1`
      - ``tp_benefit`` :math:`= \lambda`
      - ``fp_cost`` :math:`= ROI`
    * - Accepted :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= 0`
      - ``tn_benefit`` :math:`= 0`

Maximum Profit for Credit Scoring (MPCS)
========================================

:func:`~empulse.metrics.mpcs_score` treats the loss given default as a single known fraction.

Defaults: ``loan_lost_rate=0.275``, ``roi=0.2644``.

.. code-block:: python

    import numpy as np
    from empulse.metrics import mpcs_score

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    profit = mpcs_score(y_true, y_score)
    profit_custom = mpcs_score(y_true, y_score, loan_lost_rate=0.4, roi=0.30)

Expected Maximum Profit for Credit Scoring (EMPCS)
==================================================

The loss given default varies from loan to loan, and lenders typically observe it as a mixture:
some defaults are recovered in full, some are total write-offs, and the rest fall in between.
:func:`~empulse.metrics.empcs_score` captures exactly that shape. It is a
:class:`~empulse.metrics.MixtureMetric` combining three components:

1. full recovery (:math:`\lambda = 0`), weighted by ``success_rate``
2. total loss (:math:`\lambda = 1`), weighted by ``default_rate``
3. a partial loss in between, taking the remaining weight

Defaults: ``success_rate=0.55``, ``default_rate=0.1``, ``roi=0.2644``.

.. code-block:: python

    from empulse.metrics import empcs_score

    expected_profit = empcs_score(y_true, y_score)
    expected_profit_custom = empcs_score(y_true, y_score, success_rate=0.4, default_rate=0.2)

How many applicants should you reject?
--------------------------------------

.. code-block:: python

    from empulse.metrics import classification_threshold

    reject_fraction = empcs_score.optimal_rate(y_true, y_score)
    threshold = classification_threshold(y_true, y_score, customer_threshold=reject_fraction)

Instance-dependent alternatives
===============================

Both metrics above are *class-dependent*: every loan shares the same ``roi`` and loss rate. Real
portfolios are not like that — a loan's expected loss depends on its own principal and term.

For instance-dependent credit scoring, build a metric from the cost matrix that
:func:`~empulse.datasets.fetch_give_me_some_credit` and
:func:`~empulse.datasets.load_credit_scoring_pakdd` already ship, which expresses the false-negative
cost as a per-applicant credit line times the loss given default.

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_give_me_some_credit
    from empulse.metrics import Metric, Savings

    dataset = fetch_give_me_some_credit(backend=pd)

    # replace with your own model's predicted probabilities
    y_proba = np.random.default_rng(0).uniform(size=len(dataset.target))

    savings = Metric(dataset.cost_matrix, Savings())
    score = savings(dataset.target, y_proba, **dataset.instance_costs)

See :ref:`credit_scoring_pakdd` and :ref:`give_me_some_credit` for the full cost matrices, and
:ref:`user_defined_value_metric` for writing your own from scratch.

See also
========

- :ref:`choosing_metric` — how to pick between cost, savings and profit metrics.
- :ref:`metric_class_in_model` — which models can train on which strategies.
- :ref:`prebuilt_churn_metrics` and :ref:`prebuilt_acquisition_metrics` — the equivalent families
  for other use cases.

References
==========

.. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
       Development and application of consumer credit scoring models using profit-based
       classification measures. *European Journal of Operational Research*, 238(2), 505-513.
