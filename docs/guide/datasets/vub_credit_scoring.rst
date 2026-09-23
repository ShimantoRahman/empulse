.. _vub_credit_scoring:

==================
VUB Credit Scoring
==================

Summary
=======

This dataset holds loans granted by a Romanian non-banking financial institution, published in
anonymised form by the VUB Data Analytics Laboratory alongside Petrides et al. [1]_.
The goal is to predict whether a borrower will fall 45 or more days behind on a payment.

It has since become one of the standard benchmarks for instance-dependent cost-sensitive credit
scoring [2]_ [3]_. The dataset is bundled with Empulse and works offline.

=================   ==============
Classes                          2
Defaulters                    3206
Non-defaulters               15711
Samples                      18917
Features                        16
=================   ==============

Using the Dataset
=================

The dataset is loaded through the :func:`~empulse.datasets.load_vub_credit_scoring` function.
This returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'cl'``, ``'fp_cost'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import numpy as np
    import pandas as pd
    from empulse.datasets import load_vub_credit_scoring
    from empulse.metrics import Metric, Cost

    dataset = load_vub_credit_scoring(backend=pd)

    # replace with your own model's predicted probabilities
    y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

    cost = Metric(dataset.cost_matrix, Cost())
    default_lgd = cost(dataset.target, y_score, **dataset.instance_costs)
    higher_lgd = cost(dataset.target, y_score, loss_given_default=0.9, **dataset.instance_costs)

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
    - :math:`\pi_0` : percentage of non-defaulters
    - :math:`\pi_1` : percentage of defaulters
    - :math:`Cl_i` : credit line of the borrower
    - :math:`\bar{Cl}` : average credit line
    - :math:`L_{gd}` : the fraction of the loan amount which is lost if the borrower defaults

This is the credit scoring cost matrix of Bahnsen et al. [4]_, with an interest rate of 4.79%, a
cost of funds of 2.94%, a term of 24 months and a loss given default of 75%.
The loss given default stays symbolic; the other parameters are baked into ``'fp_cost'``.

Petrides et al. [1]_ derived their own costs from the institution's average return on investment
and loss given default per business channel. That is not possible from the published data: every
monetary column, including the loan amount and the ``Expected_loss`` and ``Expected_profit``
columns, is standardised to zero mean and unit variance. Following Vanderschueren et al. [2]_,
the loan amount is shifted to be strictly positive,
:math:`Cl_i = Loan\_amount_i - \min_j Loan\_amount_j + 10^{-9}`, and used as the credit line.
The resulting costs are in arbitrary units: compare them relative to each other, not as money.

Data Description
================

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Description
     - Type
   * - v1 – v8
     - Anonymised application variables
     - categorical
   * - has_fico
     - Whether the applicant has a FICO score
     - binary
   * - business_channel
     - Business channel through which the loan was granted (1, 2 or 3)
     - categorical
   * - fico_score
     - FICO score, standardised; 0 when missing
     - numeric
   * - loan_amount
     - Loan amount, standardised
     - numeric
   * - monthly_income
     - Monthly income, standardised
     - numeric
   * - age
     - Age of the borrower, standardised
     - numeric
   * - gearing_coefficient
     - Gearing coefficient, standardised
     - numeric
   * - max_gearing_ratio
     - Maximum gearing ratio, standardised
     - numeric
   * - default
     - Whether the borrower fell 45 or more days behind on a payment (1 = yes, 0 = no)
     - binary

References
==========

.. [1] Petrides, G., Moldovan, D., Coenen, L., Guns, T., & Verbeke, W. (2020).
       Cost-sensitive learning for profit-driven credit scoring.
       Journal of the Operational Research Society. https://doi.org/10.1080/01605682.2020.1843975

.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. Information Sciences, 594, 400–415.

.. [3] Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability
       of model explanations in instance-dependent cost-sensitive credit scoring.
       European Journal of Operational Research, 326(2), 630–640.

.. [4] A. Correa Bahnsen, D. Aouada, B. Ottersten,
       "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
       in Proceedings of the International Conference on Machine Learning and Applications, 2014.
