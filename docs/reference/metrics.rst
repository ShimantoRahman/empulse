.. module:: empulse.metrics

Metrics
=======

The :mod:`~empulse.metrics` module contains a collection of metrics for evaluating the performance of
models in the context of customer churn, credit scoring, and acquisition.

General Metrics
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   emp
   emp_score
   mp
   mp_score
   lift_score

Acquisition Metrics
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   empa
   empa_score
   mpa
   mpa_score
   mpa_cost_score
   make_objective_acquisition


Customer Churn Metrics
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   empc
   empc_score
   mpc
   mpc_score
   mpc_cost_score
   make_objective_churn

Credit Scoring Metrics
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   empcs
   empcs_score
   mpcs
   mpcs_score

Helper Functions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   classification_threshold
