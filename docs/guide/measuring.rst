.. _measuring:

==================
Measuring in money
==================

A cost matrix on its own is not a number. A **strategy** decides how it becomes one, and the pair
becomes a :class:`~empulse.metrics.Metric` you can call like any scikit-learn scoring function.

If your problem is customer churn, acquisition or credit scoring, the last three pages ship the
standard cost matrix for that domain — you may not need to write one at all.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Choosing a strategy
        :link: choosing_metric
        :link-type: ref

        The six strategies, what each one assumes about your scores, and which models can train on
        them.

    .. grid-item-card:: Working with metric objects
        :link: metric_objects
        :link-type: ref

        What a ``Metric`` can do beyond returning a score: optimal thresholds and rates, missing
        parameters, LaTeX.

    .. grid-item-card:: Customer Churn Metrics
        :link: prebuilt_churn_metrics
        :link-type: ref

        The standard retention cost matrix, and the ready-made EMPC and MPC metrics built on it.

    .. grid-item-card:: Customer Acquisition Metrics
        :link: prebuilt_acquisition_metrics
        :link-type: ref

        The standard acquisition cost matrix, and the EMPA and MPA metrics built on it.

    .. grid-item-card:: Credit Scoring Metrics
        :link: prebuilt_credit_scoring_metrics
        :link-type: ref

        The standard credit scoring cost matrix, and the EMPCS and MPCS metrics built on it.

.. toctree::
    :maxdepth: 2
    :hidden:

    measuring/strategies.rst
    measuring/metric_objects.rst
    measuring/churn.rst
    measuring/acquisition.rst
    measuring/credit_scoring.rst
