.. _south_german_credit:

===================
South German Credit
===================

Summary
=======

1,000 consumer loans granted by a southern German bank in 1973–1975, in the corrected version
published by Grömping [1]_. The widely used Statlog "German Credit" data comes from the same loans,
but many of its variables are wrongly coded; this version fixes them. The task is to predict whether
a loan is a bad credit risk.

Bad credits are heavily oversampled, so the 30% default rate is far above the bank's actual rate.
Ballegeer et al. [2]_ use this dataset with the cost matrix of Bahnsen et al. [3]_, taking the
credit amount as the credit line. Empulse follows them.

=================   ==============
Classes                          2
Bad credit                     300
Good credit                    700
Samples                       1000
Features                        20
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_south_german_credit`. It is
downloaded from the UCI Machine Learning Repository on first use and cached under
``~/empulse_data`` (override with ``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later
calls work offline.

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
    from empulse.datasets import fetch_south_german_credit

    dataset = fetch_south_german_credit(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

Every feature except ``duration``, ``amount`` and ``age`` is an integer category code, so one-hot
encode those before fitting a linear model:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = ['duration', 'amount', 'age']
    categorical = [c for c in X.columns if c not in numeric]

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
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
    - :math:`Cl_i` : the credit amount (``amount``, in Deutsche Mark)
    - :math:`r_i` : the profit lost by rejecting what would have been a good loan
    - :math:`\bar{r}` : the average profit lost by rejecting a good loan
    - :math:`\pi_0` : the share of good credits
    - :math:`\pi_1` : the share of bad credits
    - :math:`\bar{Cl}` : the average credit amount
    - :math:`L_{gd}` : the fraction of the credit amount lost when the borrower defaults

Rejecting a good applicant costs the profit their loan would have made, less what lending the money
to an average alternative applicant would have earned instead [3]_. The profit is computed with an
interest rate of 4.79%, a cost of funds of 2.94% and a term of 24 months, and is baked into
``'fp_cost'``.

Because :math:`\pi_1` enters the false positive cost, the oversampled bad credits make that cost
larger than it would be at the bank's real default rate.

The loss given default stays symbolic, with the default :math:`L_{gd} = 0.75` of Ballegeer et
al. [2]_. Override it by passing its alias ``loss_given_default`` when evaluating the metric:

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_lgd = cost(y, y_score, **dataset.instance_costs)
    higher_lgd = cost(y, y_score, loss_given_default=0.9, **dataset.instance_costs)

Data Description
================

Feature names are the English names of Grömping [1]_, whose code tables define every category
code.

.. list-table::
   :header-rows: 1
   :widths: 28 57 15

   * - Feature
     - Description
     - Type
   * - ``status``
     - Status of the checking account (1 = no account, 2 = negative balance, 3 = up to 200 DM,
       4 = 200 DM or more)
     - categorical
   * - ``duration``
     - Duration of the credit, in months
     - numeric
   * - ``credit_history``
     - History of compliance with previous credit contracts (0 = delays in the past, …,
       4 = all credits at this bank paid back duly)
     - categorical
   * - ``purpose``
     - Purpose of the credit (0 = others, 1 = new car, 2 = used car, …, 10 = business)
     - categorical
   * - ``amount``
     - Credit amount in Deutsche Mark; also the credit line
     - numeric
   * - ``savings``
     - Savings (1 = unknown or none, …, 5 = 1,000 DM or more)
     - categorical
   * - ``employment_duration``
     - Time with the current employer (1 = unemployed, …, 5 = 7 years or more)
     - categorical
   * - ``installment_rate``
     - Instalments as a share of disposable income (1 = 35% or more, …, 4 = under 20%)
     - categorical
   * - ``personal_status_sex``
     - Combined sex and marital status
     - categorical
   * - ``other_debtors``
     - Other debtors or guarantors (1 = none, 2 = co-applicant, 3 = guarantor)
     - categorical
   * - ``present_residence``
     - Time at the current residence (1 = under a year, …, 4 = 7 years or more)
     - categorical
   * - ``property``
     - Most valuable property (1 = unknown or none, …, 4 = real estate)
     - categorical
   * - ``age``
     - Age in years
     - numeric
   * - ``other_installment_plans``
     - Instalment plans with other providers (1 = bank, 2 = stores, 3 = none)
     - categorical
   * - ``housing``
     - Type of housing (1 = for free, 2 = rent, 3 = own)
     - categorical
   * - ``number_credits``
     - Number of credits at this bank (1 = one, …, 4 = six or more)
     - categorical
   * - ``job``
     - Quality of the job (1 = unemployed or unskilled non-resident, …, 4 = manager, self-employed
       or highly qualified)
     - categorical
   * - ``people_liable``
     - Number of people financially dependent on the debtor (1 = three or more, 2 = up to two)
     - categorical
   * - ``telephone``
     - Whether a telephone is registered in the customer's name (1 = no, 2 = yes)
     - categorical
   * - ``foreign_worker``
     - Whether the debtor is a foreign worker (1 = yes, 2 = no)
     - categorical
   * - default (target)
     - Whether the credit is a bad risk (1 = bad, 0 = good)
     - binary

References
==========

.. [1] Grömping, U. (2019). South German Credit Data: Correcting a widely used data set.
       Reports in Mathematics, Physics and Chemistry, Report 4/2019, Department II,
       Beuth University of Applied Sciences Berlin.

.. [2] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability of model
       explanations in instance-dependent cost-sensitive credit scoring.
       *European Journal of Operational Research*, 326(2), 630–640.

.. [3] Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014). Example-dependent cost-sensitive logistic
       regression for credit scoring. In *2014 13th International Conference on Machine Learning and
       Applications* (pp. 263–269).
