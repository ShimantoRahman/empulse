.. _ieee_fraud_detection:

========================
IEEE-CIS Fraud Detection
========================

Summary
=======

Real-world e-commerce transactions from the payment service provider Vesta, released for the
IEEE Computational Intelligence Society's fraud detection competition on Kaggle [1]_. Each row is
an online transaction, described by its amount, the card, addresses, e-mail domains, many
engineered features and, for part of the transactions, the device and network used. The task is to
flag fraudulent transactions.

The data is the competition's training set, the transaction table joined to the identity table.
As in Vanderschueren et al. [2]_, the transaction identifier and the timestamp offset are dropped.

=================   ==============
Classes                          2
Frauds                       20663
Legitimate                  569877
Samples                     590540
Features                       431
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_ieee_fraud_detection`. It is
downloaded from OpenML on first use, which is a large download, and cached under
``~/empulse_data`` (override with ``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later
calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'amount'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_ieee_fraud_detection

    dataset = fetch_ieee_fraud_detection(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

Most features have missing values, and fitting a model on all 431 of them and 590,540 rows is
slow. The example below imputes the numeric features of a sample of the transactions and trains
:class:`~empulse.models.CSBoostClassifier` on them:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSBoostClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    sample = X.sample(n=50_000, random_state=0).index
    X_sample = X.loc[sample].select_dtypes(include=['number'])
    y_sample = y.loc[sample]
    amount_sample = dataset.instance_costs['amount'][sample]

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('model', CSBoostClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    model.fit(X_sample, y_sample, model__amount=amount_sample)

Cost Matrix
===========

Flagging a transaction triggers an investigation with a fixed administrative cost :math:`c_f`,
whether or not it turns out to be fraud. A fraud that is not flagged costs its full amount
:math:`A_i`. Vanderschueren et al. [2]_ use this cost matrix, from Höppner et al. [3]_, for this
dataset.

.. list-table::

    * -
      - Actual fraud :math:`y_i = 1`
      - Actual legitimate :math:`y_i = 0`
    * - Predicted fraud :math:`\hat{y}_i = 1`
      - ``tp_cost`` :math:`= c_f`
      - ``fp_cost`` :math:`= c_f`
    * - Predicted legitimate :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= A_i`
      - ``tn_cost`` :math:`= 0`

The investigation cost is a symbolic parameter with the default :math:`c_f = 10` of both papers,
exposed under the alias ``investigation_cost``. Override it by passing the alias when evaluating
the metric:

.. code-block:: python

    y_score = model.predict_proba(X_sample)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_cost = cost(y_sample, y_score, amount=amount_sample)
    expensive_investigation = cost(y_sample, y_score, investigation_cost=50, amount=amount_sample)

Data Description
================

Most features are masked by Vesta. The groups below follow the description the competition host
published [1]_.

.. list-table::
   :header-rows: 1
   :widths: 35 50 15

   * - Features
     - Description
     - Type
   * - ``transaction_amt``
     - Transaction amount in US dollars; also the false-negative cost
     - numeric
   * - ``product_cd``
     - Product code of the transaction
     - categorical
   * - ``card1`` – ``card6``
     - Payment card information, such as card type, category, issuing bank and country
     - categorical
   * - ``addr1``, ``addr2``
     - Billing address region and country
     - categorical
   * - ``dist1``, ``dist2``
     - Distances, for example between addresses
     - numeric
   * - ``p_email_domain``, ``r_email_domain``
     - E-mail domains of the purchaser and the recipient
     - categorical
   * - ``c1`` – ``c14``
     - Counts, such as how many addresses are associated with the card
     - numeric
   * - ``d1`` – ``d15``
     - Time deltas, such as days since the previous transaction
     - numeric
   * - ``m1`` – ``m9``
     - Matches, such as between the names on the card and the address
     - categorical
   * - ``v1`` – ``v339``
     - Features engineered by Vesta, including rankings, counts and entity relations
     - numeric
   * - ``id_01`` – ``id_11``
     - Identity information: network connection and digital signature
     - numeric
   * - ``id_12`` – ``id_38``, ``device_type``, ``device_info``
     - Identity information: network connection, browser, operating system and device
     - categorical
   * - fraud (target)
     - Whether the transaction is fraudulent (1 = fraud, 0 = legitimate)
     - binary

Identity information is only available for about a quarter of the transactions; for the others,
those features are missing.

References
==========

.. [1] IEEE Computational Intelligence Society and Vesta Corporation. IEEE-CIS Fraud Detection.
       Kaggle, 2019. https://www.kaggle.com/c/ieee-fraud-detection

.. [2] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. *Information Sciences*, 594, 400–415.

.. [3] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       *European Journal of Operational Research*, 297(1), 291–300.
