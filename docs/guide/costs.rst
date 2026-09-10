.. _defining_costs:

==============
Defining costs
==============

Everything in Empulse starts from one object: a :class:`~empulse.metrics.CostMatrix` recording what
each of the four classification outcomes is worth. Metrics score with it, models train on it,
samplers resample by it and the bundled datasets ship one.

This stage covers writing that matrix down, handing it to an estimator, and getting per-row cost
values safely through a pipeline. Once you have a cost matrix, :doc:`measuring` turns it into a
number.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: The cost matrix
        :link: cost_matrix
        :link-type: ref

        The four outcomes, the cost/benefit sign convention, and the builder API — symbols,
        aliases, defaults and stochastic terms.

    .. grid-item-card:: Handing costs to an estimator
        :link: specifying_costs
        :link-type: ref

        Plain ``fp_cost``/``fn_cost`` values or a full ``Metric`` as ``loss``: which to use, and
        the rules that govern both.

    .. grid-item-card:: Costs that differ per row
        :link: instance_based_cv
        :link-type: ref

        Metadata routing carries per-instance values through pipelines, cross-validation folds and
        hyperparameter searches.

    .. grid-item-card:: Worked cost matrices
        :link: user_defined_value_metric
        :link-type: ref

        Complete examples built from business parameters, including uncertain parameters and
        mixture distributions.

.. toctree::
    :maxdepth: 2

    costs/cost_matrix.rst
    costs/specifying_costs.rst
    costs/metadata_routing.rst
    costs/recipes.rst
