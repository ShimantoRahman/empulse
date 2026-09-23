.. _default_credit_card_clients:

===============================
Default of Credit Card Clients
===============================

Summary
=======

Credit card clients of a bank in Taiwan, collected by Yeh & Lien [1]_ and published in the UCI
Machine Learning Repository. Each row is a client, described by their credit limit, demographics
and six months of repayment history (April to September 2005). The task is to predict whether they
default on their next payment.

The credit limit of every client is known, which makes it a natural credit line for the
instance-dependent cost matrix of Bahnsen et al. [2]_. Vanderschueren et al. [3]_ use the dataset
this way, and Empulse follows them.

=================   ==============
Classes                          2
Defaulters                    6636
Non-defaulters               23364
Samples                      30000
Features                        23
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_default_credit_card_clients`. It is
downloaded from OpenML on first use and cached under ``~/empulse_data`` (override with
``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later calls work offline.

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
    from empulse.datasets import fetch_default_credit_card_clients

    dataset = fetch_default_credit_card_clients(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

All features are numeric codes or amounts. Pass the cost matrix to the model as a
:class:`~empulse.metrics.Metric` loss, and hand it the instance costs at fit time:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
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
    - :math:`Cl_i` : the client's credit limit (``limit_bal``)
    - :math:`r_i` : the profit lost by refusing a client who would have paid
    - :math:`\bar{r}` : the average profit lost by refusing such a client
    - :math:`\pi_0` : the share of non-defaulters
    - :math:`\pi_1` : the share of defaulters
    - :math:`\bar{Cl}` : the average credit limit
    - :math:`L_{gd}` : the fraction of the credit line lost when the client defaults

Refusing a good client costs the profit their credit would have made, less what lending the money
to an average alternative client would have earned instead [2]_. The profit is computed with an
interest rate of 4.79% and a cost of funds of 2.94% (the European rates Bahnsen et al. [2]_ use)
and the two-year term they fix for credit cards, as Vanderschueren et al. [3]_ do. It is baked into
``'fp_cost'``.

The loss given default stays symbolic, with the default :math:`L_{gd} = 0.75`. Override it by
passing its alias ``loss_given_default`` when evaluating the metric:

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_lgd = cost(y, y_score, **dataset.instance_costs)
    higher_lgd = cost(y, y_score, loss_given_default=0.9, **dataset.instance_costs)

Data Description
================

Amounts are in New Taiwan dollars. Descriptions follow Yeh & Lien [1]_.

.. list-table::
   :header-rows: 1
   :widths: 25 60 15

   * - Feature
     - Description
     - Type
   * - ``limit_bal``
     - Amount of credit given, including the family's supplementary credit; also the credit line
     - numeric
   * - ``sex``
     - 1 = male, 2 = female
     - categorical
   * - ``education``
     - 1 = graduate school, 2 = university, 3 = high school, 4 = others (the data also contains
       the undocumented codes 0, 5 and 6)
     - categorical
   * - ``marriage``
     - 1 = married, 2 = single, 3 = others (the data also contains the undocumented code 0)
     - categorical
   * - ``age``
     - Age in years
     - numeric
   * - ``pay_0``, ``pay_2`` – ``pay_6``
     - Repayment status from September back to April 2005: -1 = paid duly, 1–9 = months of delay
       (the data also contains -2 and 0)
     - numeric
   * - ``bill_amt1`` – ``bill_amt6``
     - Bill statement amount from September back to April 2005
     - numeric
   * - ``pay_amt1`` – ``pay_amt6``
     - Amount paid from September back to April 2005
     - numeric
   * - default (target)
     - Whether the client defaulted on their next payment (1 = yes, 0 = no)
     - binary

References
==========

.. [1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
       predictive accuracy of probability of default of credit card clients.
       *Expert Systems with Applications*, 36(2), 2473–2480.

.. [2] Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014). Example-dependent cost-sensitive logistic
       regression for credit scoring. In *2014 13th International Conference on Machine Learning and
       Applications* (pp. 263–269).

.. [3] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. *Information Sciences*, 594, 400–415.
