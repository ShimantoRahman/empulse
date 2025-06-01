.. _metric_class_in_model:

=====================================
Use your custom metric inside a model
=====================================

Once you have defined your own cost-sensitive or value-driven metric using the :class:`empulse.metrics.Metric` class,
you can use it inside the Empulse models.
The supported models are able to convert your custom metric into a loss function
that can be used during training which is appropriate for the model type.

For example defining a custom metric and using it in a model can be done as follows:

.. code-block:: python

    import numpy as np
    import sympy
    from empulse.metrics import Metric
    from empulse.models import CSBoostClassifier
    from sklearn.datasets import make_classification


    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    clvs = np.random.uniform(100, 200, size=y.shape[0])

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    cost_loss = (
        Metric('cost')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias('accept_rate', gamma)
        .alias('incentive_cost', d)
        .alias('contact_cost', f)
        .build()
    )

    model = CSBoostClassifier(loss=cost_loss)
    model.fit(X, y, accept_rate=0.3, incentive_cost=10, clv=clvs, contact_cost=1)
    y_proba = model.predict_proba(X)[:, 1]


Currently only a subset of the models in Empulse support custom metrics, but more models will be added in the future.
The table below summarizes the Empulse models that support custom metrics:

.. list-table:: Models that support custom metrics
    :widths: 20 20 20 20
    :header-rows: 1

    * - Model
      - Cost
      - Savings
      - Maximum profit
    * - :class:`~empulse.models.CSLogitClassifier`
      - ✅
      - ❌
      - ❌
    * - :class:`~empulse.models.CSBoostClassifier`
      - ✅
      - ❌
      - ❌
    * - :class:`~empulse.models.CSThresholdClassifier`
      - ✅
      - ✅
      - ✅ (no stochastic variables)
    * - :class:`~empulse.models.ProfLogitClassifier`
      - ✅
      - ✅
      - ✅

