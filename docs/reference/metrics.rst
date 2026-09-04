.. module:: empulse.metrics

empulse.metrics
===============

The :mod:`~empulse.metrics` module contains a collection of metrics for evaluating the performance of
models in the context of customer churn, credit scoring, and acquisition.


Build your own cost-sensitive metric
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   BaseMetric
   Metric
   MetricStrategy
   MixtureComponent
   MixtureMetric
   MaxProfit
   EmpiricalMaxProfit
   AUEPC
   CostMatrix
   Cost
   LogCost
   Savings

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

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empa_score
   mpa_score
   expected_cost_loss_acquisition


Customer Churn Metrics
----------------------

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
