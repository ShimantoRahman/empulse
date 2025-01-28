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

=================   ==============
Classes                          2
Defaulters                    7616
Non-defaulters              105299
Samples                     112915
Features                        10
=================   ==============

Using the Dataset
=================

The dataset can be loaded through the :func:`~empulse.datasets.load_give_me_some_credit` function.
This returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``tp_cost``: the cost of a true positive
- ``fp_cost``: the cost of a false positive
- ``fn_cost``: the cost of a false negative
- ``tn_cost``: the cost of a true negative
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    from empulse.datasets import load_give_me_some_credit

    dataset = load_give_me_some_credit()

Alternatively, the load function can also return the features, target, and costs separately,
by setting ``return_X_y_costs=True``.
Additionally, you can specify that you want the output in a :class:`pandas:pandas.DataFrame` format,
by setting ``as_frame=True``.

The following code snippet demonstrates how to load the dataset and fit a model using the
:class:`~empulse.models.CSLogitClassifier`:

.. code-block:: python

    from empulse.datasets import load_give_me_some_credit
    from empulse.models import CSLogitClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, tp_cost, fp_cost, fn_cost, tn_cost = load_give_me_some_credit(
        return_X_y_costs=True,
        as_frame=True
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier())
    ])
    pipeline.fit(
        X,
        y,
        model__tp_cost=tp_cost,
        model__fp_cost=fp_cost,
        model__fn_cost=fn_cost,
        model__tn_cost=tn_cost
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

These assumptions can be changed by passing your own values to the
:func:`~empulse.datasets.load_give_me_some_credit` function:

.. code-block:: python

    from empulse.datasets import load_give_me_some_credit

    X, y, tp_cost, fp_cost, fn_cost, tn_cost = load_give_me_some_credit(
        return_X_y_costs=True,
        interest_rate=0.0479,
        fund_cost=0.0294,
        max_credit_line=25000,
        loss_given_default=0.75,
        term_length_months=24,
        loan_to_income_ratio=3,
    )

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