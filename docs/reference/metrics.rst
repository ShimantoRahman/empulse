.. module:: empulse.metrics

empulse.metrics
===============

The :mod:`~empulse.metrics` module contains a collection of metrics for evaluating the performance of
models in the context of customer churn, credit scoring, and acquisition.

General Metrics
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   Metric
   max_profit
   max_profit_score
   lift_score
   cost_loss
   expected_cost_loss
   expected_log_cost_loss
   make_objective_aec
   savings_score
   expected_savings_score

Customer Acquisition Metrics
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empa
   empa_score
   mpa
   mpa_score
   expected_cost_loss_acquisition
   make_objective_acquisition


Customer Churn Metrics
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empc
   empc_score
   mpc
   mpc_score
   empb
   empb_score
   auepc_score
   expected_cost_loss_churn
   make_objective_churn

Credit Scoring Metrics
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   empcs
   empcs_score
   mpcs
   mpcs_score

Helper Functions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: base.rst

   classification_threshold
