.. meta::
    :description lang=en:
        Empulse is a scikit-learn based Python package for cost-sensitive and
        value-driven (also known as profit-driven) machine learning.
        Solve your imbalanced data and other cost-sensitive learning data science problems
        with a sklearn compatible library.

=======
Empulse
=======

**Not every mistake costs the same.**

Accuracy, F1 and AUC treat a false positive and a false negative as equally bad. Your business
does not. Flagging a loyal customer as a churner wastes a discount; missing a real churner loses
their entire lifetime value. A model tuned for accuracy quietly optimises the wrong thing.

Empulse lets you write down what each outcome is actually worth, then use that same definition to
**evaluate** models, **train** them, and **set their decision threshold** — as a normal
scikit-learn estimator.

.. code-block:: python

    from empulse.metrics import Cost, CostMatrix, Metric
    from empulse.models import CSBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, random_state=42)

    # 1. Write down what each mistake costs
    cost_matrix = (
        CostMatrix()
        .add_fp_cost('wasted_discount')
        .add_fn_cost('lost_customer')
        .set_default(wasted_discount=5, lost_customer=100)
    )

    # 2. Turn it into a metric
    expected_cost = Metric(cost_matrix, Cost())

    # 3. Train a model that optimises it directly
    model = CSBoostClassifier(loss=expected_cost).fit(X, y)

    cost_per_customer = expected_cost(y, model.predict_proba(X)[:, 1])

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Getting Started
        :link: getting_started
        :link-type: doc

        Install Empulse and get a working cost-sensitive model in five minutes.
        Start here if you are new.

    .. grid-item-card:: :octicon:`book;1.5em;sd-mr-1` Tutorial
        :link: tutorial
        :link-type: doc

        A complete worked example on a real churn dataset, from a cost-blind
        baseline to a deployed, profit-optimised pipeline.

    .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` User Guide
        :link: guide
        :link-type: doc

        Task-oriented guides for each model, metric, sampler and dataset.
        Go here once you know what you need.

    .. grid-item-card:: :octicon:`code-square;1.5em;sd-mr-1` API Reference
        :link: api
        :link-type: doc

        The full class and function reference, with every parameter
        documented.

What can you do with Empulse?
=============================

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: Measure profit, not accuracy

        Ready-made metrics for :ref:`churn <prebuilt_churn_metrics>`,
        :ref:`acquisition <prebuilt_acquisition_metrics>` and
        :ref:`credit scoring <prebuilt_credit_scoring_metrics>`, or
        :ref:`define your own <user_defined_value_metric>`.

    .. grid-item-card:: Train on your cost matrix

        :ref:`Logistic regression <cslogit>`, :ref:`gradient boosting <csboost>`,
        :ref:`trees and ensembles <cstree>` that optimise business value
        during training.

    .. grid-item-card:: Set the right threshold

        :ref:`Pick the cut-off <threshold_tuning>` that maximises profit
        instead of defaulting to 0.5.

    .. grid-item-card:: Per-customer costs

        Costs that differ per row are :ref:`routed through pipelines and
        cross-validation <instance_based_cv>` automatically.

    .. grid-item-card:: Handle noisy costs

        :ref:`RobustCSClassifier <robustcs>` detects and corrects outliers in
        instance-dependent costs.

    .. grid-item-card:: Reweight the data instead

        :ref:`Cost-proportionate sampling <cost_sampling>` and
        :ref:`bias mitigation <bias_mitigation>` make any estimator
        cost-sensitive.

Everything follows scikit-learn conventions, so Empulse estimators drop into
:class:`~sklearn.pipeline.Pipeline`, :class:`~sklearn.model_selection.GridSearchCV` and
:func:`~sklearn.model_selection.cross_val_score` unchanged, and every metric can be wrapped with
:func:`~sklearn.metrics.make_scorer`.

.. toctree::
    :maxdepth: 1
    :hidden:

    getting_started.rst
    tutorial.rst
    guide.rst
    api.rst
    project_info.rst
