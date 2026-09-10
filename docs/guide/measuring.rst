.. _measuring:

==================
Measuring in money
==================

A cost matrix on its own is not a number. A **strategy** decides how it becomes one, and the pair
becomes a :class:`~empulse.metrics.Metric` you can call like any scikit-learn scoring function.

Start with :doc:`measuring/strategies` to choose between the six strategies, then
:doc:`measuring/metric_objects` for what a metric object can do beyond returning a score. If your
problem is customer churn, acquisition or credit scoring, the last three pages ship the standard
cost matrix for that domain — you may not need to write one at all.

.. toctree::
    :maxdepth: 2

    measuring/strategies.rst
    measuring/metric_objects.rst
    measuring/churn.rst
    measuring/acquisition.rst
    measuring/credit_scoring.rst
