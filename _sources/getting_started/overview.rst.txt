.. _overview:

========================
Which tool do I need?
========================

Empulse has a lot of surface area. This page maps problems to the component that solves them, so
you can skip straight to the right guide.

Everything rests on one idea: write down what each classification outcome is worth, then reuse
that definition everywhere. That definition is a :class:`~empulse.metrics.CostMatrix`, and pairing
it with a strategy makes it a :class:`~empulse.metrics.Metric` you can score with, train on, and
threshold by.

Start from your situation
=========================

.. list-table::
    :widths: 46 54
    :header-rows: 1

    * - I want to...
      - Use
    * - Score a model I already have, in money rather than accuracy
      - A prebuilt metric (:ref:`churn <prebuilt_churn_metrics>`,
        :ref:`acquisition <prebuilt_acquisition_metrics>`,
        :ref:`credit scoring <prebuilt_credit_scoring_metrics>`) or your own
        :ref:`Metric <user_defined_value_metric>`
    * - Train a model that optimises business value directly
      - :ref:`CSLogitClassifier <cslogit>`, :ref:`CSBoostClassifier <csboost>`,
        :ref:`CSTreeClassifier and ensembles <cstree>`
    * - Keep my existing model and only fix the decision threshold
      - :ref:`CSThresholdClassifier / CSRateClassifier <threshold_tuning>`
    * - Keep my existing model and change the training data instead
      - :ref:`CostSensitiveSampler <cost_sampling>`
    * - Handle costs that differ for every row
      - :ref:`Instance-dependent costs and metadata routing <instance_based_cv>`
    * - Handle cost estimates that are noisy or contain outliers
      - :ref:`RobustCSClassifier <robustcs>`
    * - Make predictions independent of a sensitive attribute
      - :ref:`Bias mitigation <bias_mitigation>`
    * - Benchmark on realistic data
      - :ref:`Bundled datasets <datasets>`

Choosing a metric strategy
==========================

A cost matrix on its own is not a number. A **strategy** decides how it becomes one, and the right
choice depends on what you have and what you want.

.. list-table::
    :widths: 18 16 26 40
    :header-rows: 1

    * - Strategy
      - Direction
      - Expects
      - Use when
    * - :class:`~empulse.metrics.Cost`
      - Lower is better
      - Calibrated probabilities
      - You want the expected cost per instance, in currency.
    * - :class:`~empulse.metrics.Savings`
      - Higher is better
      - Calibrated probabilities
      - You want that same cost relative to a naive baseline, as a 0-1 ratio that is comparable
        across datasets.
    * - :class:`~empulse.metrics.MaxProfit`
      - Higher is better
      - Ranking scores
      - The threshold is not fixed yet and you want the profit at the best possible cut-off —
        optionally averaging over uncertain business parameters.

.. warning::
    :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings` assume ``y_score`` holds
    calibrated probabilities; :class:`~empulse.metrics.MaxProfit` only needs a ranking. Passing the
    wrong kind does not raise an error, it just returns a misleading number. See
    :ref:`choosing_metric` for the details.

Not every model supports every strategy — :ref:`metric_class_in_model` has the compatibility table.

Choosing a model
================

.. list-table::
    :widths: 30 70
    :header-rows: 1

    * - Model
      - Best when
    * - :ref:`CSLogitClassifier <cslogit>`
      - You want a linear, interpretable model with coefficients you can explain.
    * - :ref:`CSBoostClassifier <csboost>`
      - You want the strongest predictive performance and have XGBoost, LightGBM or CatBoost
        installed. Usually the best default.
    * - :ref:`CSTreeClassifier <cstree>`
      - You want a single interpretable tree split on a cost-sensitive criterion.
    * - :ref:`CSForestClassifier / CSBaggingClassifier <csforest>`
      - You want the robustness of an ensemble of cost-sensitive trees.
    * - :ref:`ProfLogitClassifier <proflogit>` / :ref:`ProfTreeClassifier <proftree>`
      - Your objective is non-smooth (typically :class:`~empulse.metrics.MaxProfit`) and needs a
        gradient-free optimizer.
    * - :ref:`B2BoostClassifier <b2boost>`
      - You have a B2B churn problem and want the cost matrix pre-wired.
    * - :ref:`RobustCSClassifier <robustcs>`
      - Your instance-dependent costs are estimates that may contain outliers.

Two ways to specify costs
=========================

Every cost-sensitive model accepts costs in either of two forms.

**Plain costs** — quickest, when your costs are just four numbers or four arrays:

.. code-block:: python

    from empulse.models import CSBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, random_state=42)

    model = CSBoostClassifier()
    model.fit(X, y, fp_cost=5, fn_cost=100)

**A Metric** — when costs are built from business parameters, or you want the same definition used
for scoring and training:

.. code-block:: python

    from empulse.metrics import Cost, CostMatrix, Metric

    cost_matrix = (
        CostMatrix()
        .add_fp_cost('discount')
        .add_fn_cost('lost_value')
        .set_default(discount=5, lost_value=100)
    )
    model = CSBoostClassifier(loss=Metric(cost_matrix, Cost()))
    model.fit(X, y)

Prefer the second when a parameter varies per instance, when you want to tune a business parameter
by cross-validation, or when the cost formula is more than a single number per outcome.

Where next
==========

- :doc:`../tutorial` — a complete worked example end to end.
- :ref:`choosing_metric` — the concepts behind cost matrices and strategies.
- :doc:`../guide` — reference-depth guides for every component.
