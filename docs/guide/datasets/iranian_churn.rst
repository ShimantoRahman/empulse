.. _iranian_churn:

=====================
Iranian Churn Dataset
=====================

Summary
=======

A customer churn dataset from an Iranian telecom company, collected over 12 months [1]_ and
published in the UCI Machine Learning Repository [2]_. Each row is a customer, described by usage
and account features, with a label indicating whether they churned by the end of the period.

What makes this dataset well suited to value-driven modelling is its ``Customer Value`` column,
which Empulse exposes as a per-customer lifetime value (``clv``). Because the value of retaining a
customer varies, the profit-optimal customer to target is not simply the one most likely to churn —
which is exactly the situation the :ref:`maximum profit framework <prebuilt_churn_metrics>` was
designed for.

=================   ==============
Classes                          2
Churners                       495
Non-churners                  2755
Samples                       3150
Features                        12
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_iranian_churn`. It is downloaded from
the UCI repository on first use and cached under ``~/empulse_data`` (override with
``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'clv'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn

    dataset = fetch_iranian_churn(backend=pd)

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

All features are numeric, so a scaler is enough preprocessing for a linear model:

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(X, y, model__clv=dataset.instance_costs['clv'])

Because the cost matrix is the standard churn retention matrix, the dataset also works directly
with the prebuilt churn metrics:

.. code-block:: python

    from empulse.metrics import empc_score

    y_score = pipeline.predict_proba(X)[:, 1]
    expected_profit = empc_score(y, y_score, clv=dataset.instance_costs['clv'])
    target_fraction = empc_score.optimal_rate(y, y_score, clv=dataset.instance_costs['clv'])

Cost Matrix
===========

Contacting a customer costs a fraction :math:`f` of their value, whether or not they accept.
A contacted customer accepts the retention offer with probability :math:`\gamma`, in which case
their value is retained minus the incentive, a fraction :math:`d` of that value. Losing a customer
you did not contact costs their full value.

.. list-table::

    * -
      - Actual churner :math:`y_i = 1`
      - Actual non-churner :math:`y_i = 0`
    * - Predicted churner :math:`\hat{y}_i = 1`
      - ``tp_benefit`` :math:`= \gamma (CLV_i - d \cdot CLV_i - f \cdot CLV_i) - (1-\gamma) f \cdot CLV_i`
      - ``fp_cost`` :math:`= d \cdot CLV_i + f \cdot CLV_i`
    * - Predicted non-churner :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= CLV_i`
      - ``tn_benefit`` :math:`= 0`

The symbolic parameters carry these defaults, and can be overridden by passing their alias:

.. list-table::
    :widths: 30 20 50
    :header-rows: 1

    * - Alias
      - Default
      - Meaning
    * - ``accept_rate`` (:math:`\gamma`)
      - 0.3
      - Probability a contacted customer accepts the offer
    * - ``incentive_fraction`` (:math:`d`)
      - 0.05
      - Retention incentive, as a fraction of CLV
    * - ``contact_fraction`` (:math:`f`)
      - 0.01
      - Cost of contacting a customer, as a fraction of CLV

.. code-block:: python

    from empulse.metrics import Metric, Cost

    cost = Metric(dataset.cost_matrix, Cost())
    generous_offer = cost(
        y,
        y_score,
        accept_rate=0.5,
        incentive_fraction=0.10,
        clv=dataset.instance_costs['clv'],
    )

.. warning::
    132 of the 3150 customers have a ``Customer Value`` of exactly 0. For those rows every term of
    the cost matrix evaluates to 0, which makes the profit-optimal decision undefined for that
    customer. As a result :meth:`~empulse.metrics.Metric.optimal_threshold` and
    :meth:`~empulse.metrics.Metric.optimal_rate` raise a ``ValueError`` for the
    :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings` strategies on this
    dataset.

    The :class:`~empulse.metrics.MaxProfit` strategy is unaffected, because it derives the
    operating point from the ROC convex hull across the whole population rather than per customer.
    Use ``empc_score.optimal_rate(...)`` as shown above, or drop the zero-value customers if you
    need a cost-based threshold.

Data Description
================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Description
   * - ``call_failure``
     - Number of failed calls
   * - ``complains``
     - Whether the customer filed a complaint (0 = no, 1 = yes)
   * - ``subscription_length``
     - Total months of subscription
   * - ``charge_amount``
     - Ordinal attribute from 0 (lowest) to 9 (highest)
   * - ``seconds_of_use``
     - Total seconds of calls
   * - ``frequency_of_use``
     - Total number of calls
   * - ``frequency_of_sms``
     - Total number of text messages
   * - ``distinct_called_numbers``
     - Number of distinct phone numbers called
   * - ``age_group``
     - Ordinal age band from 1 (youngest) to 5 (oldest)
   * - ``tariff_plan``
     - Tariff plan (1 = pay as you go, 2 = contractual)
   * - ``status``
     - Subscription status (1 = active, 2 = non-active)
   * - ``age``
     - Age of the customer in years

The ``Customer Value`` column is not part of ``data``; it is returned separately as
``instance_costs['clv']`` because it is a cost driver rather than a predictive feature. Using it
as a feature would leak business value into the model instead of into the objective.

References
==========

.. [1] Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., & Keramati, A. (2020).
       Optimum profit-driven churn decision making: innovative artificial neural networks in
       telecom industry. *Neural Computing and Applications*, 32(18), 14929-14962.
       https://doi.org/10.1007/s00521-020-04850-6

.. [2] Iranian Churn Dataset. UCI Machine Learning Repository (ID 563).
       https://doi.org/10.24432/C5JW3Z
