.. module:: empulse.metrics

empulse.metrics
===============

The :mod:`~empulse.metrics` module contains a collection of metrics for evaluating the performance of
models in the context of customer churn, credit scoring, and acquisition.


Build your own cost-sensitive metric
------------------------------------

Metrics can be built by combining a :class:`~empulse.metrics.CostMatrix` with a :class:`~empulse.metrics.MetricStrategy`.
The cost matrix defines the costs and benefits associated with different outcomes,
while the metric strategy defines how to compute the metric based on the cost matrix and the model's predictions.

Cost Matrix
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   CostMatrix

Metric Strategies
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   MetricStrategy
   MaxProfit
   EmpiricalMaxProfit
   AUEPC
   Cost
   LogCost
   Savings

Metrics
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   BaseMetric
   Metric
   MixtureComponent
   MixtureMetric

General Metrics
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   max_profit_score
   lift_score
   cost_loss
   expected_cost_loss
   expected_log_cost_loss
   savings_score
   expected_savings_score

Customer Acquisition Metrics
----------------------------

See :ref:`prebuilt_acquisition_metrics` for an explanation of the cost-benefit matrix.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empa_score
   mpa_score
   expected_cost_loss_acquisition


Customer Churn Metrics
----------------------

See :ref:`prebuilt_churn_metrics` for an explanation of the cost-benefit matrix.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empc_score
   mpc_score
   empb_score
   auepc_score
   expected_cost_loss_churn

Credit Scoring Metrics
----------------------

See :ref:`prebuilt_credit_scoring_metrics` for an explanation of the cost-benefit matrix.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empcs_score
   mpcs_score

Helper Functions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   classification_threshold
