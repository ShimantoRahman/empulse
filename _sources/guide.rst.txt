==========
User Guide
==========

These pages cover each part of Empulse in depth: what every component does, the parameters that
matter, and how it fits into a scikit-learn workflow.

They are written as reference material to dip into once you know roughly what you need. If you are
new to the package, read the :doc:`getting_started/quickstart` first, then the :doc:`tutorial`,
which works through a complete problem end to end. :doc:`getting_started/overview` maps common
problems to the component that solves them.

The guide follows the order you actually work in. You write down what the outcomes are worth, turn
that into a number, train on it, decide where to draw the line, and — if you would rather not touch
the model at all — change the data instead. Every later stage consumes what the earlier ones
produced, so reading top to bottom works; so does jumping straight to the stage you are stuck on.

.. themed-figure:: empulse_spine
    :alt: A cost matrix combines with a strategy to make a metric, which is then used to
        evaluate, train and decide; the cost matrix also feeds resampling directly.

    One definition, four uses. The stage numbers match the sections below.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`pencil;1.5em;sd-mr-1` 1. Defining costs
        :link: guide/costs
        :link-type: doc

        Write down what each of the four outcomes is worth. Cost matrices, the two ways to hand
        those costs to an estimator, and how per-row values travel through pipelines.

    .. grid-item-card:: :octicon:`meter;1.5em;sd-mr-1` 2. Measuring in money
        :link: guide/measuring
        :link-type: doc

        Turn a cost matrix into a number. The six strategies, what a ``Metric`` object can do, and
        the ready-made metrics for churn, acquisition and credit scoring.

    .. grid-item-card:: :octicon:`beaker;1.5em;sd-mr-1` 3. Training on costs
        :link: guide/training
        :link-type: doc

        Classifiers that optimise business value during training instead of being scored on it
        afterwards: linear, boosting, trees and ensembles, minimax, and robustness to noisy costs.

    .. grid-item-card:: :octicon:`git-compare;1.5em;sd-mr-1` 4. Deciding who to act on
        :link: guide/deciding
        :link-type: doc

        A score is not a decision. Calibration, and picking the cut-off — or the fraction of the
        population — that maximises value.

    .. grid-item-card:: :octicon:`filter;1.5em;sd-mr-1` 5. Changing the data instead
        :link: guide/preprocessing
        :link-type: doc

        Make any estimator cost-sensitive without touching its objective: class imbalance,
        cost-proportionate sampling and bias mitigation.

    .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` 6. Datasets
        :link: guide/datasets
        :link-type: doc

        Cost-sensitive datasets for benchmarking, each shipping the cost matrix of
        the business problem it came from.

.. toctree::
    :maxdepth: 2
    :numbered:
    :hidden:

    guide/costs.rst
    guide/measuring.rst
    guide/training.rst
    guide/deciding.rst
    guide/preprocessing.rst
    guide/datasets.rst
