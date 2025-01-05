`Unreleased`_
=============

- ...


`0.5.1`_ (05-01-2025)
=====================
- Fix documentation build issue

`0.5.0`_ (05-01-2025)
=====================

- Added supported for python 3.13
- Added cost-sensitive models
    - CSLogitClassifier
    - CSBoostClassifier
    - RobustCSClassifier
    - CSThresholdClassifier
- Added cost-sensitive metrics
    - cost_loss
    - expected_cost_loss
    - expected_log_cost_loss
    - savings_score
    - expected_savings_score
- Added cost-sensitive sampler
    - CostSensitiveSampler
- Added datasets module
-  rename metric arguments which expect target score from y_pred to y_score and
   target probabilities from y_pred to y_proba.
- Allow all cost-sensitive models and samplers to accept cost parameters during initialization


.. _Unreleased: https://github.com/ShimantoRahman/empulse/compare/0.5.0...main
.. _0.5.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.5.0