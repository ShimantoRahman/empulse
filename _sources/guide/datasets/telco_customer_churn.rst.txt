.. _telco_customer_churn:

===========================
Telco Customer Churn (IBM)
===========================

Summary
=======

Customer data of a telecommunications company from IBM's sample data sets, published on
Kaggle [1]_. Each row is a customer, described by the services they subscribe to, their account and
a few demographics, with a label indicating whether they left within the last month.

The 11 customers without a ``TotalCharges``, all with a tenure of 0 months, are dropped, as
in Vanderschueren et al. [2]_. Every customer's monthly charge is known, and it drives the cost
matrix: missing a churner loses a year of their charges, while a retention offer costs two months
of them.

=================   ==============
Classes                          2
Churners                      1869
Non-churners                  5163
Samples                       7032
Features                        19
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_telco_customer_churn`. It is
downloaded from OpenML on first use and cached under ``~/empulse_data`` (override with
``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'monthly_charges'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_telco_customer_churn

    dataset = fetch_telco_customer_churn(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

Most features are categorical. Pass the cost matrix to the model as a
:class:`~empulse.metrics.Metric` loss, and hand it the monthly charges at fit time:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = X.select_dtypes(include=['number']).columns
    categorical = X.select_dtypes(exclude=['number']).columns

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ])),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(X, y, model__monthly_charges=dataset.instance_costs['monthly_charges'])

Cost Matrix
===========

A churner who is not targeted leaves, which costs :math:`n_{FN}` months of their monthly charge
:math:`A_i`. Targeting a customer who would have stayed wastes a retention offer worth
:math:`n_{FP}` months of charges. Following Vanderschueren et al. [2]_, who take the scheme from
Petrides & Verbeke [3]_, correctly targeted churners and correctly ignored customers cost
nothing.

.. list-table::

    * -
      - Actual churner :math:`y_i = 1`
      - Actual non-churner :math:`y_i = 0`
    * - Predicted churner :math:`\hat{y}_i = 1`
      - ``tp_cost`` :math:`= 0`
      - ``fp_cost`` :math:`= n_{FP} \cdot A_i`
    * - Predicted non-churner :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= n_{FN} \cdot A_i`
      - ``tn_cost`` :math:`= 0`

The numbers of months are symbolic parameters with the defaults of the paper, and can be
overridden by passing their name:

.. list-table::
    :widths: 30 20 50
    :header-rows: 1

    * - Parameter
      - Default
      - Meaning
    * - ``fn_months`` (:math:`n_{FN}`)
      - 12
      - Months of charges lost when a churner is missed
    * - ``fp_months`` (:math:`n_{FP}`)
      - 2
      - Months of charges given away as a retention offer

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_cost = cost(y, y_score, **dataset.instance_costs)
    cheaper_offer = cost(y, y_score, fp_months=1, **dataset.instance_costs)

Data Description
================

.. list-table::
   :header-rows: 1
   :widths: 25 60 15

   * - Feature
     - Description
     - Type
   * - ``gender``
     - ``Female`` or ``Male``
     - categorical
   * - ``senior_citizen``
     - Whether the customer is a senior citizen (1 = yes, 0 = no)
     - binary
   * - ``partner``
     - Whether the customer has a partner
     - categorical
   * - ``dependents``
     - Whether the customer has dependents
     - categorical
   * - ``tenure``
     - Number of months the customer has stayed with the company
     - numeric
   * - ``phone_service``
     - Whether the customer has a phone service
     - categorical
   * - ``multiple_lines``
     - Whether the customer has multiple lines (or no phone service)
     - categorical
   * - ``internet_service``
     - Internet service provider: ``DSL``, ``Fiber optic`` or ``No``
     - categorical
   * - ``online_security``, ``online_backup``, ``device_protection``, ``tech_support``
     - Whether the customer has each add-on (or no internet service)
     - categorical
   * - ``streaming_tv``, ``streaming_movies``
     - Whether the customer streams TV or movies (or has no internet service)
     - categorical
   * - ``contract``
     - Contract term: ``Month-to-month``, ``One year`` or ``Two year``
     - categorical
   * - ``paperless_billing``
     - Whether the customer has paperless billing
     - categorical
   * - ``payment_method``
     - Electronic check, mailed check, bank transfer or credit card
     - categorical
   * - ``monthly_charges``
     - Amount charged to the customer each month; also the cost driver
     - numeric
   * - ``total_charges``
     - Total amount charged to the customer
     - numeric
   * - churn (target)
     - Whether the customer left within the last month (1 = yes, 0 = no)
     - binary

References
==========

.. [1] BlastChar. Telco Customer Churn. IBM Sample Data Sets, via Kaggle.
       https://www.kaggle.com/blastchar/telco-customer-churn

.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. *Information Sciences*, 594, 400–415.

.. [3] Petrides, G., & Verbeke, W. (2021). Cost-sensitive ensemble learning: a unifying framework.
       *Data Mining and Knowledge Discovery*, 1–28.
