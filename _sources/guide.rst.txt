==========
User Guide
==========

These pages cover each part of Empulse in depth: what every component does, the parameters that
matter, and how it fits into a scikit-learn workflow.

They are written as reference material to dip into once you know roughly what you need. If you are
new to the package, read the :doc:`getting_started/quickstart` first, then the :doc:`tutorial`,
which works through a complete problem end to end. :doc:`getting_started/overview` maps common
problems to the component that solves them.

Start with **Metrics**: everything else consumes a cost matrix or a metric. The models train on
them, the samplers resample by them, and the datasets ship them.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`meter;1.5em;sd-mr-1` Metrics
        :link: guide/metrics_guide
        :link-type: doc

        Write down what each outcome is worth, and turn it into a number.
        Cost matrices, the three strategies, and the prebuilt metrics for
        churn, acquisition and credit scoring.

    .. grid-item-card:: :octicon:`beaker;1.5em;sd-mr-1` Models
        :link: guide/models_guide
        :link-type: doc

        Classifiers that train on your cost matrix instead of accuracy, plus
        the meta-estimators for threshold tuning and robustness to noisy
        costs.

    .. grid-item-card:: :octicon:`filter;1.5em;sd-mr-1` Preprocessing
        :link: guide/preprocessing_guide
        :link-type: doc

        Make any estimator cost-sensitive by changing the data rather than
        the algorithm: cost-proportionate sampling and bias mitigation.

    .. grid-item-card:: :octicon:`git-branch;1.5em;sd-mr-1` Instance-dependent costs
        :link: guide/instance_based_cv
        :link-type: doc

        When every row has its own cost, metadata routing carries those
        values correctly through pipelines and cross-validation folds.

    .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Datasets
        :link: guide/datasets_guide
        :link-type: doc

        Five real-world cost-sensitive datasets for benchmarking, each
        shipping the cost matrix of the business problem it came from.

.. toctree::
    :maxdepth: 2
    :numbered:
    :hidden:

    guide/metrics_guide.rst
    guide/models_guide.rst
    guide/preprocessing_guide.rst
    guide/instance_based_cv.rst
    guide/datasets_guide.rst
