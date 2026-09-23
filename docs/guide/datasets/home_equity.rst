.. _home_equity:

=====================
Home Equity (HMEQ)
=====================

Summary
=======

Baseline and loan performance information for 5,960 recent home equity loans, from the book
*Credit Risk Analytics* [1]_. A bank's consumer credit department wants to automate the approval
of home equity lines of credit, and the task is to predict which applicants eventually default or
become seriously delinquent.

Unlike most public credit scoring data, it records the amount each applicant asked for, which is
what an instance-dependent cost matrix needs. Ballegeer et al. [2]_ use it with the cost matrix of
Bahnsen et al. [3]_, which Empulse follows.

=================   ==============
Classes                          2
Defaulters                    1189
Non-defaulters                4771
Samples                       5960
Features                        12
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_home_equity`. It is downloaded from
OpenML on first use and cached under ``~/empulse_data`` (override with ``$EMPULSE_DATA_HOME`` or
the ``data_home`` argument), so later calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'cl'``, ``'fp_cost'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_home_equity

    dataset = fetch_home_equity(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

Most variables have missing values, so the data needs imputation before a linear model can use it.
Pass the cost matrix to the model as a :class:`~empulse.metrics.Metric` loss, and hand it the
instance costs at fit time:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = X.select_dtypes(include=['number']).columns
    categorical = X.select_dtypes(exclude=['number']).columns

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
            ('cat', make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing'),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ), categorical),
        ])),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(
        X,
        y,
        model__cl=dataset.instance_costs['cl'],
        model__fp_cost=dataset.instance_costs['fp_cost'],
    )

Cost Matrix
===========

.. list-table::

    * -
      - Actual positive :math:`y_i = 1`
      - Actual negative :math:`y_i = 0`
    * - Predicted positive :math:`\hat{y}_i = 1`
      - ``tp_cost`` :math:`= 0`
      - ``fp_cost`` :math:`= r_i + -\bar{r} \cdot \pi_0 + \bar{Cl} \cdot L_{gd} \cdot \pi_1`
    * - Predicted negative :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= Cl_i \cdot L_{gd}`
      - ``tn_cost`` :math:`= 0`

with
    - :math:`Cl_i` : the loan amount requested (``loan_amount``)
    - :math:`r_i` : the profit lost by rejecting what would have been a good loan
    - :math:`\bar{r}` : the average profit lost by rejecting a good loan
    - :math:`\pi_0` : the share of non-defaulters
    - :math:`\pi_1` : the share of defaulters
    - :math:`\bar{Cl}` : the average loan amount
    - :math:`L_{gd}` : the fraction of the loan amount lost when the borrower defaults

Rejecting a good applicant costs the profit their loan would have made, less what lending the money
to an average alternative applicant would have earned instead [3]_. The profit is computed with an
interest rate of 4.79%, a cost of funds of 2.94% and a term of 24 months, and is baked into
``'fp_cost'``.

The loss given default stays symbolic, with the default :math:`L_{gd} = 0.75` of Ballegeer et
al. [2]_. Override it by passing its alias ``loss_given_default`` when evaluating the metric:

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_lgd = cost(y, y_score, **dataset.instance_costs)
    higher_lgd = cost(y, y_score, loss_given_default=0.9, **dataset.instance_costs)

Data Description
================

The original column name is given in brackets.

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Feature
     - Description
     - Type
   * - ``loan_amount`` (``LOAN``)
     - Amount of the loan requested
     - numeric
   * - ``mortgage_due`` (``MORTDUE``)
     - Amount due on the existing mortgage
     - numeric
   * - ``property_value`` (``VALUE``)
     - Value of the current property
     - numeric
   * - ``reason`` (``REASON``)
     - ``DebtCon`` (debt consolidation) or ``HomeImp`` (home improvement)
     - categorical
   * - ``job`` (``JOB``)
     - Occupational category
     - categorical
   * - ``years_at_job`` (``YOJ``)
     - Years at the present job
     - numeric
   * - ``n_derogatory_reports`` (``DEROG``)
     - Number of major derogatory reports
     - numeric
   * - ``n_delinquent_credit_lines`` (``DELINQ``)
     - Number of delinquent credit lines
     - numeric
   * - ``oldest_credit_line_age`` (``CLAGE``)
     - Age of the oldest credit line, in months
     - numeric
   * - ``n_recent_credit_inquiries`` (``NINQ``)
     - Number of recent credit inquiries
     - numeric
   * - ``n_credit_lines`` (``CLNO``)
     - Number of credit lines
     - numeric
   * - ``debt_to_income`` (``DEBTINC``)
     - Debt-to-income ratio
     - numeric
   * - default (target)
     - Whether the applicant defaulted or was seriously delinquent (1 = yes, 0 = no)
     - binary

References
==========

.. [1] Baesens, B., Roesch, D., & Scheule, H. (2016). *Credit Risk Analytics: Measurement
       Techniques, Applications, and Examples in SAS*. John Wiley & Sons.

.. [2] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability of model
       explanations in instance-dependent cost-sensitive credit scoring.
       *European Journal of Operational Research*, 326(2), 630–640.

.. [3] Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014). Example-dependent cost-sensitive logistic
       regression for credit scoring. In *2014 13th International Conference on Machine Learning and
       Applications* (pp. 263–269).
