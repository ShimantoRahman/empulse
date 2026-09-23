.. _cell2cell:

========================
Cell2Cell Customer Churn
========================

Summary
=======

Customer data of the US wireless operator Cell2Cell, released for the churn modelling tournament of
the Teradata Center for Customer Relationship Management at Duke University. It is the "Duke" data
that runs through the profit-driven churn literature [1]_ [2]_ [3]_. Each row is a subscriber,
described by usage, call quality, handset, household and retention-contact variables, with a label
indicating whether they churned.

The Duke center no longer distributes the data, so Empulse downloads the ``cell2celltrain.csv``
file from a public GitHub mirror, pinned to a fixed commit. The 156 customers without a
``MonthlyRevenue`` are dropped.

=================   ==============
Classes                          2
Churners                     14641
Non-churners                 36250
Samples                      50891
Features                        56
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_cell2cell`. It is downloaded on first
use and cached under ``~/empulse_data`` (override with ``$EMPULSE_DATA_HOME`` or the
``data_home`` argument), so later calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'monthly_revenue'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_cell2cell

    dataset = fetch_cell2cell(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

A few numeric features have missing values, and the categorical features need encoding. Pass the
cost matrix to the model as a :class:`~empulse.metrics.Metric` loss, and hand it the monthly
revenue at fit time:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    numeric = X.select_dtypes(include=['number']).columns
    categorical = X.select_dtypes(exclude=['number']).columns

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
            ('cat', make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing'),
                TargetEncoder(),
            ), categorical),
        ])),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(X, y, model__monthly_revenue=dataset.instance_costs['monthly_revenue'])

Cost Matrix
===========

The cost matrix is the churn retention matrix of Verbraken et al. [4]_, the framing every paper
above uses for this data. Contacting a customer costs a fixed amount :math:`f`, whether or not they
accept. A contacted churner accepts the retention offer with probability :math:`\gamma`, in which
case their value is retained minus the incentive, a fraction :math:`d` of that value. A churner who
is not contacted simply leaves, which costs the campaign nothing.

.. list-table::

    * -
      - Actual churner :math:`y_i = 1`
      - Actual non-churner :math:`y_i = 0`
    * - Predicted churner :math:`\hat{y}_i = 1`
      - ``tp_benefit`` :math:`= \gamma (CLV_i - d \cdot CLV_i - f) - (1-\gamma) f`
      - ``fp_cost`` :math:`= d \cdot CLV_i + f`
    * - Predicted non-churner :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= 0`
      - ``tn_benefit`` :math:`= 0`

The papers use a single average lifetime value of 200 for every customer [1]_. This version of the
data records each customer's monthly revenue, so Empulse makes the lifetime value
instance-dependent by counting :math:`m` months of it:

.. math::

    CLV_i = m \cdot \max(MonthlyRevenue_i, 0)

The 3 customers with a negative revenue (net credits) get a lifetime value of 0: a customer who
costs money has nothing to retain.

The symbolic parameters carry these defaults, and can be overridden by passing their alias:

.. list-table::
    :widths: 30 20 50
    :header-rows: 1

    * - Alias
      - Default
      - Meaning
    * - ``clv_months`` (:math:`m`)
      - 12
      - Months of revenue counted as lifetime value
    * - ``accept_rate`` (:math:`\gamma`)
      - 0.3
      - Probability a contacted churner accepts the offer, the mean of the Beta(6, 14)
        distribution used by the expected maximum profit measure
    * - ``incentive_fraction`` (:math:`d`)
      - 0.05
      - Retention incentive as a fraction of CLV, as an incentive of 10 on a CLV of 200 in
        Verbeke et al. [1]_
    * - ``contact_cost`` (:math:`f`)
      - 1
      - Cost of contacting a customer, as an absolute amount

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_cost = cost(y, y_score, **dataset.instance_costs)
    longer_lifetime = cost(y, y_score, clv_months=24, **dataset.instance_costs)

Data Description
================

The mirror does not document the variables individually; they fall into these groups.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Features
     - Description
   * - ``monthly_revenue``, ``monthly_minutes``, ``total_recurring_charge``,
       ``director_assisted_calls``, ``overage_minutes``, ``roaming_calls``,
       ``perc_change_minutes``, ``perc_change_revenues``
     - Revenue and usage, and their recent percentage change. ``monthly_revenue`` is also the
       cost driver.
   * - ``dropped_calls``, ``blocked_calls``, ``unanswered_calls``, ``customer_care_calls``,
       ``threeway_calls``, ``received_calls``, ``outbound_calls``, ``inbound_calls``,
       ``peak_calls_in_out``, ``off_peak_calls_in_out``, ``dropped_blocked_calls``,
       ``call_forwarding_calls``, ``call_waiting_calls``
     - Call volumes and call quality
   * - ``months_in_service``, ``unique_subs``, ``active_subs``, ``service_area``
     - Account tenure, number of subscriptions and service area
   * - ``handsets``, ``handset_models``, ``current_equipment_days``, ``handset_refurbished``,
       ``handset_web_capable``, ``handset_price``
     - Handset history and the current handset
   * - ``age_hh1``, ``age_hh2``, ``children_in_hh``, ``income_group``, ``homeownership``,
       ``marital_status``, ``occupation``, ``prizm_code``, ``truck_owner``, ``rv_owner``,
       ``owns_motorcycle``, ``owns_computer``, ``has_credit_card``, ``buys_via_mail_order``,
       ``responds_to_mail_offers``, ``opt_out_mailings``, ``non_us_travel``,
       ``new_cellphone_user``, ``not_new_cellphone_user``
     - Household demographics and lifestyle
   * - ``credit_rating``, ``adjustments_to_credit_rating``
     - Credit rating and adjustments to it
   * - ``retention_calls``, ``retention_offers_accepted``, ``made_call_to_retention_team``,
       ``referrals_made_by_subscriber``
     - Contacts with the retention team, and referrals
   * - churn (target)
     - Whether the customer churned (1 = yes, 0 = no)

References
==========

.. [1] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
       New insights into churn prediction in the telecommunication sector: A profit driven data
       mining approach. *European Journal of Operational Research*, 218(1), 211–229.

.. [2] Höppner, S., Stripling, E., Baesens, B., vanden Broucke, S., & Verdonck, T. (2020).
       Profit driven decision trees for churn prediction.
       *European Journal of Operational Research*, 284(3), 920–933.

.. [3] Maldonado, S., López, J., & Vairetti, C. (2020). Profit-based churn prediction based on
       Minimax Probability Machines. *European Journal of Operational Research*, 284(1), 273–284.

.. [4] Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing metric for
       measuring classification performance of customer churn prediction models.
       *IEEE Transactions on Knowledge and Data Engineering*, 25(5), 961–973.
