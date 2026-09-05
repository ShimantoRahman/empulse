=======
Metrics
=======

Metrics are the foundation of Empulse. A :class:`~empulse.metrics.CostMatrix` records what each
classification outcome is worth, and pairing it with a :class:`~empulse.metrics.MetricStrategy`
produces a :class:`~empulse.metrics.Metric` you can score with, train on, and derive a decision
threshold from.

Start with :doc:`metrics/choosing_metric` for the concepts, then
:doc:`metrics/user_defined_value_metric` to build a cost matrix for your own problem and
:doc:`metrics/metric_class_in_model` to train on it.

If your use case is customer churn, acquisition or credit scoring, the last three pages cover
ready-made metrics that already encode the standard cost matrix for that domain — you may not need
to write one at all.

.. toctree::
    :maxdepth: 2

    metrics/choosing_metric.rst
    metrics/user_defined_value_metric.rst
    metrics/metric_class_in_model.rst
    metrics/prebuilt_churn_metrics.rst
    metrics/prebuilt_acquisition_metrics.rst
    metrics/prebuilt_credit_scoring_metrics.rst
