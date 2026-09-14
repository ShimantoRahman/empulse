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

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Linear Cost-Sensitive Models
        :link: cslogit
        :link-type: ref

        Cost-sensitive logistic regression: the interpretable choice, with a coefficient per
        feature you can read.

    .. grid-item-card:: Cost-Sensitive Gradient Boosting
        :link: csboost
        :link-type: ref

        XGBoost, LightGBM and CatBoost trained on your cost matrix. The strongest default.

    .. grid-item-card:: Tree-Based Cost-Sensitive Models
        :link: cstree
        :link-type: ref

        Trees, forests and ProfTree, which split on business value rather than on impurity.

    .. grid-item-card:: Minimax and symbolic models
        :link: profmpm
        :link-type: ref

        Models that optimise the worst case over an uncertain class prior, and symbolic
        alternatives.

    .. grid-item-card:: Robust Cost-Sensitive Classification
        :link: robustcs
        :link-type: ref

        A meta-estimator rather than a model: wraps a cost-sensitive model to guard it against
        outliers in noisy cost estimates.

.. toctree::
    :maxdepth: 2
    :hidden:

    training/linear_models.rst
    training/csboost.rst
    training/tree_models.rst
    training/minimax_models.rst
    training/robustcs.rst
