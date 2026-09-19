.. _give_me_some_credit:

===========================================
2011 Kaggle competition Give Me Some Credit
===========================================

Summary
=======

This is a Kaggle dataset from the credit agency Credit Fusion [1]_.
The goal is to predict whether a customer will default on a loan in the next two years.

Banks play a crucial role in market economies.
They decide who can get finance and on what terms and can make or break investment decisions.
For markets and society to function, individuals and companies need access to credit.
Credit scoring algorithms, which make a guess at the probability of default,
are the method banks use to determine whether or not a loan should be granted.

The dataset is fetched remotely from OpenML and cached locally on first use.

=================   ==============
Classes                          2
Defaulters                    7616
Non-defaulters              105299
Samples                     112915
Features                        10
=================   ==============

Using the Dataset
=================

The dataset can be fetched through the :func:`~empulse.datasets.fetch_give_me_some_credit` function.
This returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'cl'``, ``'fp_cost'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_give_me_some_credit

    dataset = fetch_give_me_some_credit(backend=pd)

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

The following code snippet demonstrates how to load the dataset and fit a model using the
:class:`~empulse.models.CSLogitClassifier`:

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_give_me_some_credit
    from empulse.models import CSLogitClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = fetch_give_me_some_credit(backend=pd)
    X, y = dataset.data, dataset.target
    cl = dataset.instance_costs['cl']
    fp_cost = dataset.instance_costs['fp_cost']
    fn_cost = cl * 0.75  # loss_given_default = 0.75

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier())
    ])
    pipeline.fit(
        X,
        y,
        model__fp_cost=fp_cost,
        model__fn_cost=fn_cost,
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
    - :math:`r_i` : loss in profit by rejecting what would have been a good loan
    - :math:`\bar{r}` : average loss in profit by rejecting what would have been a good loan
    - :math:`\pi_0` : percentage of defaulters
    - :math:`\pi_1` : percentage of non-defaulters
    - :math:`Cl_i` : credit line of the client
    - :math:`\bar{Cl}` : average credit line
    - :math:`L_{gd}` : the fraction of the loan amount which is lost if the client defaults

Using default parameters,
it is assumed that the interest rate is 4.79%, the cost of running the fund is 2.94%, the maximum credit line is 25,000,
the loss given default is 75%, the term length is 24 months, and the loan to income ratio is 3.
The default parameters are based on [2]_.

The interest rate, fund cost, maximum credit line, term length and loan-to-income ratio are
applied when the dataset is built, and are baked into the ``'fp_cost'`` and ``'cl'`` arrays
returned in ``instance_costs``.

The loss given default remains symbolic, so it can be overridden at evaluation time by passing
its alias ``loss_given_default`` to the metric:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from empulse.datasets import fetch_give_me_some_credit
    from empulse.metrics import Metric, Cost

    dataset = fetch_give_me_some_credit(backend=pd)

    # replace with your own model's predicted probabilities
    y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

    cost = Metric(dataset.cost_matrix, Cost())
    default_lgd = cost(dataset.target, y_score, **dataset.instance_costs)
    higher_lgd = cost(dataset.target, y_score, loss_given_default=0.9, **dataset.instance_costs)

Data Description
================

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Description
     - Type
   * - monthly_income
     - Monthly income of borrower
     - numeric
   * - debt_ratio
     - Monthly debt payments, alimony, living costs divided by monthly gross income
     - numeric
   * - revolving_utilization
     - Total balance on credit cards and personal lines of credit except real estate and
       no installment debt like car loans divided by the sum of credit limits
     - numeric
   * - age
     - Age of borrower in years
     - numeric
   * - n_dependents
     - Number of dependents in family excluding themselves (spouse, children etc.)
     - numeric
   * - n_open_credit_lines
     - Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)
     - numeric
   * - n_real_estate_loans
     - Number of mortgage and real estate loans including home equity lines of credit
     - numeric
   * - n_times_late_30_59_days
     - Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
     - numeric
   * - n_times_late_60_89_days
     - Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
     - numeric
   * - n_times_late_over_90_days
     - Number of times borrower has been 90 days or more past due.
     - numeric
   * - default
     - Whether a person experienced 90 days past due delinquency or worse ('yes' = 1, 'no' = 0)
     - binary

References
==========

.. [1] Credit Fusion and Will Cukierski. Give Me Some Credit.
       https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.

.. [2] A. Correa Bahnsen, D.Aouada, B, Ottersten,
       "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
       in Proceedings of the International Conference on Machine Learning and Applications, 2014.