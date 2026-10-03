.. _upsell_bank_telemarketing:

==================================
Bank Telemarketing Upsell Campaign
==================================

Summary
=======

This dataset is related to a direct marketing campaigns (phone calls) of a Portuguese banking institution.
The marketing campaigns were based on phone calls.
Often, more than one contact to the same client was required,
in order to access if the product (bank term deposit) would be or not subscribed.

Features recorded before the contact event are removed from the original dataset [1]_ to avoid data leakage.
Only clients with a positive balance are considered, since clients in debt are not eligible for term deposits.

=================   ==============
Classes                          2
Subscribers                   4787
Non-subscribers              33144
Samples                      37931
Features                        10
=================   ==============

Other relevant information can be found in [2]_ and [3]_.

Using the Dataset
=================

The dataset can be loaded through the :func:`~empulse.datasets.load_upsell_bank_telemarketing` function.
This returns a :class:`~empulse.datasets.Dataset` object with the following attributes:

- ``data``: the feature matrix
- ``target``: the target vector
- ``cost_matrix``: a :class:`~empulse.metrics.CostMatrix` with default values pre-filled
- ``instance_costs``: a dict of per-instance cost drivers (``'balance'``)
- ``feature_names``: the feature names
- ``target_names``: the target names
- ``DESCR``: the full description of the dataset

.. code-block:: python

    import pandas as pd
    from empulse.datasets import load_upsell_bank_telemarketing

    dataset = load_upsell_bank_telemarketing(backend=pd)

The ``backend`` argument selects the dataframe library used for ``data`` and ``target``.
Pass the module itself — ``backend=pd`` for pandas or ``backend=pl`` for polars.

The cost matrix is symbolic: only the customer ``balance`` varies per instance, while the
interest rate, term deposit fraction and contact cost are parameters with defaults.
Pass the cost matrix to the model as a :class:`~empulse.metrics.Metric` loss, and hand it the
instance costs at fit time:

.. code-block:: python

    import pandas as pd
    from empulse.datasets import load_upsell_bank_telemarketing
    from empulse.metrics import Metric, Cost
    from empulse.models import CSLogitClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    dataset = load_upsell_bank_telemarketing(backend=pd)
    X, y = dataset.data, dataset.target

    numeric = X.select_dtypes(include=['number']).columns
    categorical = X.select_dtypes(exclude=['number']).columns

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('cat', TargetEncoder(), categorical),
        ])),
        ('model', CSLogitClassifier(loss=Metric(dataset.cost_matrix, Cost())))
    ])
    pipeline.fit(X, y, model__balance=dataset.instance_costs['balance'])

Cost Matrix
===========

.. list-table::

    * -
      - Actual positive :math:`y_i = 1`
      - Actual negative :math:`y_i = 0`
    * - Predicted positive :math:`\hat{y}_i = 1`
      - ``tp_cost`` :math:`= c`
      - ``fp_cost`` :math:`= c`
    * - Predicted negative :math:`\hat{y}_i = 0`
      - ``fn_cost`` :math:`= r \cdot d_i \cdot b_i`
      - ``tn_cost`` :math:`= 0`

with
    - :math:`c` : cost of contacting the client
    - :math:`r` : interest rate of the term deposit
    - :math:`d_i` : fraction of the client's balance that is deposited in the term deposit
    - :math:`b_i` : client's balance

Using default parameters, it is assumed that :math:`c = 1`, :math:`r = 0.02463333`, :math:`d_i = 0.25` for all clients.
The default parameters are based on [4]_.

These assumptions are symbolic parameters of the cost matrix, exposed under the aliases
``interest_rate``, ``term_deposit_fraction`` and ``contact_cost``.
Override any of them by passing the alias when evaluating the metric:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from empulse.datasets import load_upsell_bank_telemarketing
    from empulse.metrics import Metric, Cost

    dataset = load_upsell_bank_telemarketing(backend=pd)

    # replace with your own model's predicted probabilities
    y_score = np.random.default_rng(0).uniform(size=len(dataset.target))

    cost = Metric(dataset.cost_matrix, Cost())
    score = cost(
        dataset.target,
        y_score,
        interest_rate=0.05,
        term_deposit_fraction=0.30,
        contact_cost=10,
        **dataset.instance_costs,
    )

Data Description
================

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Description
     - Type
   * - age
     - Age of the client
     - numeric
   * - balance
     - Average yearly balance
     - numeric
   * - previous
     - Number of contacts performed before this campaign and for this client
     - numeric
   * - job
     - Type of job (e.g., 'admin.', 'blue-collar', 'entrepreneur', etc.)
     - categorical
   * - marital
     - Marital status ('divorced', 'married', 'single')
     - categorical
   * - education
     - Education level ('primary', 'secondary', 'tertiary', 'unknown')
     - categorical
   * - has_credit_in_default
     - Has credit in default? ('yes' = 1, 'no' = 0)
     - binary
   * - has_housing_loan
     - Has housing loan? ('yes' = 1, 'no' = 0)
     - binary
   * - has_personal_loan
     - Has personal loan? ('yes' = 1, 'no' = 0)
     - binary
   * - previous_outcome
     - Outcome of the previous marketing campaign ('success', 'failure', 'other', 'unknown')
     - categorical
   * - subscribed
     - Has the client subscribed a term deposit? ('yes' = 1, 'no' = 0)
     - binary

References
==========

.. [1] Moro, S., Rita, P., & Cortez, P. (2014).
       Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
.. [2] Moro, S., Cortez, P., & Rita, P. (2014).
       A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
.. [3] S. Moro, R. Laureano and P. Cortez.
       Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.
       In P. Novais et al. (Eds.),
       Proceedings of the European Simulation and Modelling Conference
       - ESM'2011, pp. 117-121, Guimaraes, Portugal, October, 2011. EUROSIS. [bank.zip]
.. [4] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
       "Improving Credit Card Fraud Detection with Calibrated Probabilities",
       in Proceedings of the fourteenth SIAM International Conference on Data Mining,
       677-685, 2014.
