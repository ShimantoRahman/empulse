.. _kddcup09_churn:

=====================================
KDD Cup 2009 Orange Customer Churn
=====================================

Summary
=======

Customer relationship management data from the French telecom operator Orange, released for the
ACM KDD Cup 2009 [1]_. Each row is a customer, and the task is to predict whether they will switch
provider. This is the small version of the competition data, hosted on OpenML.

All 230 variables are anonymised, and none of them is a revenue or customer value, so the cost of a
retention campaign cannot differ between customers. The dataset therefore uses the churn cost
matrix with the average customer lifetime value that Verbeke et al. [2]_ calibrated for the
telecom sector, the same setting under which the data has been benchmarked in the profit-driven
churn literature [2]_ [3]_.

=================   ==============
Classes                          2
Churners                      3672
Non-churners                 46328
Samples                      50000
Features                       230
=================   ==============

Using the Dataset
=================

The dataset is fetched through :func:`~empulse.datasets.fetch_kddcup09_churn`. It is downloaded
from OpenML on first use and cached under ``~/empulse_data`` (override with ``$EMPULSE_DATA_HOME``
or the ``data_home`` argument), so later calls work offline.

It returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'clv'``, 200 for every customer)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_kddcup09_churn

    dataset = fetch_kddcup09_churn(backend=pd)
    X, y = dataset.data, dataset.target

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

Many variables are mostly missing and the nominal ones have thousands of levels, so the data needs
imputation and an encoding that copes with high cardinality before a linear model can use it:

.. code-block:: python

    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    numeric = X.select_dtypes(include=['number']).columns
    nominal = X.select_dtypes(exclude=['number']).columns

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()), numeric),
            ('cat', make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing', keep_empty_features=True),
                TargetEncoder(),
            ), nominal),
        ])),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost()))),
    ])
    pipeline.fit(X, y, model__clv=dataset.instance_costs['clv'])

The defaults of :func:`~empulse.metrics.empc_score` are the same parameters (a CLV of 200, an
incentive of 10 and a contact cost of 1), with the acceptance rate drawn from Beta(6, 14) instead
of fixed at its mean. The expected maximum profit measure of the literature therefore needs no
extra arguments:

.. code-block:: python

    from empulse.metrics import empc_score

    y_score = pipeline.predict_proba(X)[:, 1]
    expected_profit = empc_score(y, y_score)

Cost Matrix
===========

Contacting a customer costs a fixed amount :math:`f`, whether or not they accept. A contacted
churner accepts the retention offer with probability :math:`\gamma`, in which case their value is
retained minus the incentive, a fraction :math:`d` of that value. A churner who is not contacted
simply leaves, which costs the campaign nothing [4]_.

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

:math:`CLV_i` is 200 for every customer. The symbolic parameters carry these defaults, which are
those of Verbeke et al. [2]_, and can be overridden by passing their alias:

.. list-table::
    :widths: 30 20 50
    :header-rows: 1

    * - Alias
      - Default
      - Meaning
    * - ``accept_rate`` (:math:`\gamma`)
      - 0.3
      - Probability a contacted churner accepts the offer, the mean of the Beta(6, 14)
        distribution used by the expected maximum profit measure
    * - ``incentive_fraction`` (:math:`d`)
      - 0.05
      - Retention incentive as a fraction of CLV, an incentive of 10 on a CLV of 200
    * - ``contact_cost`` (:math:`f`)
      - 1
      - Cost of contacting a customer, as an absolute amount

.. code-block:: python

    cost = Metric(dataset.cost_matrix, Cost())
    generous_offer = cost(
        y,
        y_score,
        accept_rate=0.5,
        incentive_fraction=0.10,
        clv=dataset.instance_costs['clv'],
    )

Data Description
================

The variables are anonymised by Orange and cannot be interpreted.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Description
   * - ``var1`` – ``var190``
     - Numeric variables. Many are mostly missing.
   * - ``var191`` – ``var229``
     - Nominal variables, some with thousands of levels.
   * - ``var230``
     - Numeric, but missing for every customer (as is the nominal ``var209``).

References
==========

.. [1] Guyon, I., Lemaire, V., Boullé, M., Dror, G., & Vogel, D. (2009).
       Analysis of the KDD Cup 2009: Fast scoring on a large Orange customer database.
       In *Proceedings of KDD-Cup 2009 Competition*, JMLR Workshop and Conference Proceedings, 7, 1–22.

.. [2] Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).
       New insights into churn prediction in the telecommunication sector: A profit driven data
       mining approach. *European Journal of Operational Research*, 218(1), 211–229.

.. [3] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B., & Snoeck, M. (2018).
       Profit maximizing logistic model for customer churn prediction using genetic algorithms.
       *Swarm and Evolutionary Computation*, 40, 116–130.

.. [4] Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing metric for
       measuring classification performance of customer churn prediction models.
       *IEEE Transactions on Knowledge and Data Engineering*, 25(5), 961–973.
