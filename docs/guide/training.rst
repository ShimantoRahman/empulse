.. _training:

=================
Training on costs
=================

These models take the cost matrix into training, so they optimise business value directly instead
of being scored on it afterwards. Each one mirrors a familiar scikit-learn estimator with a
cost-sensitive objective.

All of them accept costs the same two ways — plain ``tp_cost``/``tn_cost``/``fp_cost``/``fn_cost``
values, or a :class:`~empulse.metrics.Metric` passed as ``loss`` — described once in
:ref:`specifying_costs`, and all of them route per-row costs through pipelines the same way, in
:ref:`instance_based_cv`. The pages below cover only what differs between them.

Not sure which to pick? :doc:`../getting_started/overview` has a decision table; in short,
:doc:`training/csboost` is the strongest default, :doc:`training/linear_models` is the interpretable
choice, and :doc:`training/tree_models` sits in between.

:doc:`training/robustcs` is a meta-estimator rather than a model in its own right: it wraps a
cost-sensitive model to guard it against outliers in noisy cost estimates.

.. toctree::
    :maxdepth: 2

    training/linear_models.rst
    training/csboost.rst
    training/tree_models.rst
    training/minimax_models.rst
    training/robustcs.rst
