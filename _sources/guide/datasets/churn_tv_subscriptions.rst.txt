.. _churn_tv_subscriptions:

==================================
Churn in a TV Subscription Company
==================================

Summary
=======

This is a private dataset provided by a TV cable provider [1]_. The dataset consists of
active customers during the first semester of 2014. The total dataset contains 9,410
individual registries, each one with 45 attributes, including a churn label indicating whenever
a customer is a churner. This label was created internally in the company, and can be
regarded as highly accurate. In the dataset only 455 customers are churners, leading to a
churn ratio of 4.83 %.

The features names are anonymized to protect the privacy of the customers.

=================   ==============
Classes                          2
Churners                       455
Non-churners                  8955
Samples                       9410
Features                        45
=================   ==============

Using the Dataset
=================

The dataset can be loaded through the :func:`~empulse.datasets.load_churn_tv_subscriptions` function.
This returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` over the four outcome terms
- ``instance_costs``: a dict of per-instance cost arrays
  (``'tp_benefit'``, ``'tn_benefit'``, ``'fp_cost'``, ``'fn_cost'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import load_churn_tv_subscriptions

    dataset = load_churn_tv_subscriptions(backend=pd)

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

Unlike the other churn datasets, this one ships **precomputed** costs: its cost matrix is not
built from symbolic business parameters, so each of the four outcome terms is supplied directly
as an array in ``instance_costs``. That makes it a good fit for the plain cost arguments of the
models, which take costs rather than benefits — so the two benefit terms are negated:

.. code-block:: python

    import pandas as pd
    from empulse.datasets import load_churn_tv_subscriptions
    from empulse.models import CSLogitClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = load_churn_tv_subscriptions(backend=pd)
    X, y = dataset.data, dataset.target
    costs = dataset.instance_costs

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier())
    ])
    pipeline.fit(
        X,
        y,
        model__tp_cost=-costs['tp_benefit'],
        model__tn_cost=-costs['tn_benefit'],
        model__fp_cost=costs['fp_cost'],
        model__fn_cost=costs['fn_cost'],
    )

To *evaluate* a model instead, wrap the cost matrix in a :class:`~empulse.metrics.Metric` and pass
the instance costs straight through:

.. code-block:: python

    from empulse.metrics import Metric, Cost

    cost = Metric(dataset.cost_matrix, Cost())
    y_score = pipeline.predict_proba(X)[:, 1]
    score = cost(y, y_score, **dataset.instance_costs)

Cost Matrix
===========

.. list-table::

    * -
      - Actual positive :math:`y_i = 1`
      - Actual negative :math:`y_i = 0`
    * - Predicted positive :math:`\hat{y}_i = 1`
      - ``tp_cost`` :math:`= \gamma_i d_i + (1 - \gamma_i) (CLV_i + c_i)`
      - ``fp_cost`` :math:`= d_i + c_i`
    * - Predicted negative :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= CLV_i`
      - ``tn_cost`` :math:`= 0`

with
    - :math:`\gamma_i` : probability of the customer accepting the retention offer
    - :math:`CLV_i` : customer lifetime value of the retained customer
    - :math:`d_i` : cost of incentive offered to the customer
    - :math:`c_i` : cost of contacting the customer

References
==========

.. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           `"A novel cost-sensitive framework for customer churn predictive modeling"
           <http://www.decisionanalyticsjournal.com/content/pdf/s40165-015-0014-6.pdf>`__,
           Decision Analytics, 2:5, 2015.