======================================
Cost-sensitive and Value-driven models
======================================

These models take the cost matrix into training, so they optimise business value directly instead
of being scored on it afterwards. Each one mirrors a familiar scikit-learn estimator with a
cost-sensitive objective, and each accepts costs either as plain
``tp_cost``/``tn_cost``/``fp_cost``/``fn_cost`` values or as a full
:class:`~empulse.metrics.Metric` passed as ``loss``.

Not sure which to pick? :doc:`../getting_started/overview` has a decision table; in short,
:doc:`models/csboost` is the strongest default, :doc:`models/linear_models` is the interpretable
choice, and :doc:`models/tree_models` sits in between.

Two pages cover meta-estimators rather than models in their own right:
:doc:`models/threshold_tuning` wraps an already-fitted estimator to fix its decision threshold,
and :doc:`models/robustcs` wraps a cost-sensitive model to guard it against outliers in noisy
cost estimates.

.. toctree::
    :maxdepth: 2

    models/linear_models.rst
    models/csboost.rst
    models/tree_models.rst
    models/threshold_tuning.rst
    models/robustcs.rst
