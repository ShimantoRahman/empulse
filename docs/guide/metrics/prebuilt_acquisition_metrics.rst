.. _prebuilt_acquisition_metrics:

============================
Customer Acquisition Metrics
============================

Customer acquisition inverts the churn problem: instead of keeping existing customers, you spend
money contacting *leads* in the hope of converting them. Empulse ships ready-made metrics for this
use case, following the profit-based framework of Verbraken et al. [1]_.

.. note::
    These names are **prebuilt** :class:`~empulse.metrics.Metric` instances, not functions.
    Earlier versions of Empulse exposed hand-written ``empa`` and ``mpa`` functions returning a
    ``(score, threshold)`` tuple; those have been removed. Call the metric for the score, and use
    :meth:`~empulse.metrics.Metric.optimal_rate` for the fraction of leads to target.

The Cost-Benefit Matrix
=======================

Every contacted lead costs :math:`c` (``contact_cost``), converted or not. Leads are handled in two
ways, and ``direct_selling`` is the fraction handled directly:

- **Directly**, by your own sales team: you gain the contribution :math:`R` of a converted lead but
  pay the sales cost :math:`s` (``sales_cost``).
- **Indirectly**, through an intermediary: you avoid the sales cost but pay a commission
  :math:`\kappa` (``commission``) on the contribution.

Leads you do not contact cost nothing, so both "predicted negative" outcomes are zero.

.. list-table::

    * -
      - Actual converter :math:`y_i = 1`
      - Actual non-converter :math:`y_i = 0`
    * - Predicted converter :math:`\hat{y}_i = 1`
      - ``tp_benefit`` :math:`= \sigma (R - c - s) + (1 - \sigma)\,(R (1 - \kappa) - c)`
      - ``fp_cost`` :math:`= c`
    * - Predicted non-converter :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= 0`
      - ``tn_benefit`` :math:`= 0`

with :math:`\sigma` the ``direct_selling`` fraction (``1`` = fully direct, ``0`` = fully indirect).

.. list-table::
    :widths: 32 22 46
    :header-rows: 1

    * - Metric
      - Strategy
      - Contribution of a conversion
    * - :func:`~empulse.metrics.mpa_score`
      - :class:`~empulse.metrics.MaxProfit`
      - Fixed (``contribution``)
    * - :func:`~empulse.metrics.empa_score`
      - :class:`~empulse.metrics.MaxProfit`
      - ``Gamma(alpha, beta)``
    * - :func:`~empulse.metrics.expected_cost_loss_acquisition`
      - :class:`~empulse.metrics.Cost`
      - Fixed (``contribution``)

Maximum Profit for Customer Acquisition (MPA)
=============================================

:func:`~empulse.metrics.mpa_score` treats the contribution of a conversion as a single known
amount.

Defaults: ``contribution=8000``, ``contact_cost=50``, ``sales_cost=500``, ``direct_selling=1``,
``commission=0.1``.

.. code-block:: python

    import numpy as np
    from empulse.metrics import mpa_score

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    profit = mpa_score(y_true, y_score)
    profit_indirect = mpa_score(y_true, y_score, direct_selling=0, commission=0.15)

Expected Maximum Profit for Customer Acquisition (EMPA)
=======================================================

New customers are not equally valuable, and their value is not known in advance.
:func:`~empulse.metrics.empa_score` models the contribution as a ``Gamma(alpha, beta)`` random
variable and integrates over it.

Defaults: ``alpha=12``, ``beta=1/0.0015`` (a mean contribution of
:math:`12 \times 666.7 = 8000`), ``contact_cost=50``, ``sales_cost=500``, ``direct_selling=1``,
``commission=0.1``.

.. code-block:: python

    from empulse.metrics import empa_score

    expected_profit = empa_score(y_true, y_score)

.. warning::
    ``beta`` is the **scale** of the Gamma distribution (mean = ``alpha * beta``), not the
    **rate** (mean = ``alpha / beta``) used by the removed ``empa`` function. The default was
    adjusted so that calling with no arguments reproduces the previous result, but an explicit
    non-default ``beta=`` means something different than it used to.

How many leads should you target?
---------------------------------

.. code-block:: python

    from empulse.metrics import classification_threshold

    target_fraction = empa_score.optimal_rate(y_true, y_score)
    threshold = classification_threshold(y_true, y_score, customer_threshold=target_fraction)

Expected Cost Loss for Acquisition
==================================

When you want to **minimise cost** from calibrated probabilities rather than maximise profit from
ranking scores, :func:`~empulse.metrics.expected_cost_loss_acquisition` applies the same business
model through the :class:`~empulse.metrics.Cost` strategy. Lower is better, and it always returns
the mean cost per lead.

Defaults: ``contribution=7000``, ``contact_cost=50``, ``sales_cost=500``, ``direct_selling=1``,
``commission=0.1``.

.. code-block:: python

    from empulse.metrics import expected_cost_loss_acquisition

    y_proba = np.array([0.05, 0.95, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65])

    mean_cost = expected_cost_loss_acquisition(y_true, y_proba)

Using an acquisition metric to train a model
============================================

.. code-block:: python

    from empulse.models import CSBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)

    model = CSBoostClassifier(loss=expected_cost_loss_acquisition)
    model.fit(X, y)

See :ref:`metric_class_in_model` for which models support which strategies.

See also
========

- :ref:`choosing_metric` — how to pick between cost, savings and profit metrics.
- :ref:`user_defined_value_metric` — build your own metric when none of these fit.
- :ref:`prebuilt_churn_metrics` and :ref:`prebuilt_credit_scoring_metrics` — the equivalent
  families for other use cases.

References
==========

.. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
       Development and application of consumer credit scoring models using profit-based
       classification measures. *European Journal of Operational Research*, 238(2), 505-513.
