=============
API Reference
=============

The class and function reference for Empulse, organised by module. These pages document the raw
specifications: every parameter, attribute and method.

For the reasoning behind a component and worked examples of using it, see the :doc:`guide` — the
signatures alone rarely convey when you should reach for something.

.. grid:: 1 2 2 3
    :gutter: 3

    .. grid-item-card:: :octicon:`meter;1.5em;sd-mr-1` Metrics
        :link: reference/metrics
        :link-type: doc

        ``empulse.metrics``

        Cost matrices, metric strategies, and the prebuilt churn, acquisition
        and credit scoring metrics.

    .. grid-item-card:: :octicon:`beaker;1.5em;sd-mr-1` Models
        :link: reference/models
        :link-type: doc

        ``empulse.models``

        Cost-sensitive classifiers, threshold meta-estimators, and the bias
        mitigation wrappers.

    .. grid-item-card:: :octicon:`filter;1.5em;sd-mr-1` Samplers
        :link: reference/samplers
        :link-type: doc

        ``empulse.samplers``

        Cost-proportionate and fairness-aware resampling, in the
        imbalanced-learn style.

    .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Optimizers
        :link: reference/optimizers
        :link-type: doc

        ``empulse.optimizers``

        Solvers and learning-rate schedules driving the logit-family and
        evolutionary models.

    .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Datasets
        :link: reference/datasets
        :link-type: doc

        ``empulse.datasets``

        Loaders for the five real-world cost-sensitive datasets, each shipping
        its own cost matrix.

.. note::
    Empulse has no top-level re-exports. Import from the submodules — ``from empulse.models import
    CSBoostClassifier`` — rather than from ``empulse`` directly.

.. toctree::
    :maxdepth: 2
    :hidden:

    reference/metrics.rst
    reference/models.rst
    reference/samplers.rst
    reference/optimizers.rst
    reference/datasets.rst
