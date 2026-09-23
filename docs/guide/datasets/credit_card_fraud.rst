.. _credit_card_fraud:

=================================
Credit Card Fraud Detection (ULB)
=================================

Summary
=======

Card transactions made by European cardholders over two days in September 2013, collected during
a research collaboration between Worldline and the Machine Learning Group of the Université Libre
de Bruxelles [1]_. The task is to flag fraudulent transactions. For confidentiality, all features
except the transaction amount are principal components of the original variables.

Fraud is rare: the raw data holds 492 frauds among 284,807 transactions. Following Höppner et
al. [2]_ and Vanderschueren et al. [3]_, the 1,825 transactions with a zero amount are removed,
since missing them costs nothing. That leaves 282,982 transactions, 465 of them fraudulent.

=================   ==============
Classes                          2
Frauds                         465
Legitimate                  282517
Samples                     282982
Features                        29
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_credit_card_fraud`. It is downloaded
from OpenML on first use (about 150 MB) and cached under ``~/empulse_data`` (override with
``$EMPULSE_DATA_HOME`` or the ``data_home`` argument), so later calls work offline.

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
    from empulse.datasets import fetch_credit_card_fraud

    dataset = fetch_credit_card_fraud(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

All features are numeric. Pass the cost matrix to the model as a :class:`~empulse.metrics.Metric`
loss, and hand it the transaction amounts at fit time:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(X, y, model__amount=dataset.instance_costs['amount'])

Cost Matrix
===========

Flagging a transaction triggers an investigation with a fixed administrative cost :math:`c_f`,
whether or not it turns out to be fraud. A fraud that is not flagged costs its full amount
:math:`A_i` [2]_ [3]_.

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

The investigation cost is a symbolic parameter with the default :math:`c_f = 10` used in both
papers, exposed under the alias ``investigation_cost``. Override it by passing the alias when
evaluating the metric:

.. code-block:: python

    y_score = pipeline.predict_proba(X)[:, 1]

    cost = Metric(dataset.cost_matrix, Cost())
    default_cost = cost(y, y_score, **dataset.instance_costs)
    expensive_investigation = cost(y, y_score, investigation_cost=50, **dataset.instance_costs)

Data Description
================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Description
   * - ``v1`` – ``v28``
     - Principal components of the original, confidential transaction variables
   * - ``amount``
     - Transaction amount in euros; also the false-negative cost
   * - fraud (target)
     - Whether the transaction is fraudulent (1 = fraud, 0 = legitimate)

The ``Time`` column of the original data, the seconds elapsed since the first transaction, is
dropped.

References
==========

.. [1] Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
       Calibrating probability with undersampling for unbalanced classification.
       In *2015 IEEE Symposium Series on Computational Intelligence* (pp. 159–166).

.. [2] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       *European Journal of Operational Research*, 297(1), 291–300.

.. [3] Vanderschueren, T., Verdonck, T., Baesens, B., & Verbeke, W. (2022).
       Predict-then-optimize or predict-and-optimize? An empirical evaluation
       of cost-sensitive learning strategies. *Information Sciences*, 594, 400–415.
