`Unreleased`_
=============

- ...

`0.5.2`_ (XX-01-2025)
=====================
- |Enhancement| Rework user guide documentation to better explain the usage of value-driven
  and cost-sensitive models and metrics

`0.5.1`_ (05-01-2025)
=====================
- |Fix| Fix documentation build issue

`0.5.0`_ (05-01-2025)
=====================

- |MajorFeature| Added supported for python 3.13
- |MajorFeature| Added cost-sensitive models
    - CSLogitClassifier
    - CSBoostClassifier
    - RobustCSClassifier
    - CSThresholdClassifier
- |MajorFeature| Added cost-sensitive metrics
    - cost_loss
    - expected_cost_loss
    - expected_log_cost_loss
    - savings_score
    - expected_savings_score
- |MajorFeature| Added datasets module
- |Feature| Added cost-sensitive sampler
    - CostSensitiveSampler
- |API| rename metric arguments which expect target score from y_pred to y_score and
   target probabilities from y_pred to y_proba.
- |Enhancement| Allow all cost-sensitive models and samplers to accept cost parameters during initialization


.. _Unreleased: https://github.com/ShimantoRahman/empulse/compare/0.5.2...main
.. _0.5.2: https://github.com/ShimantoRahman/empulse/releases/tag/0.5.2
.. _0.5.1: https://github.com/ShimantoRahman/empulse/releases/tag/0.5.1
.. _0.5.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.5.0

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |MajorFeature| replace:: :raw-html:`<span class="badge text-bg-success">Major Feature</span>` :raw-latex:`{\small\sc [Major Feature]}`
.. |Feature| replace:: :raw-html:`<span class="badge text-bg-success">Feature</span>` :raw-latex:`{\small\sc [Feature]}`
.. |Efficiency| replace:: :raw-html:`<span class="badge text-bg-info">Efficiency</span>` :raw-latex:`{\small\sc [Efficiency]}`
.. |Enhancement| replace:: :raw-html:`<span class="badge text-bg-info">Enhancement</span>` :raw-latex:`{\small\sc [Enhancement]}`
.. |Fix| replace:: :raw-html:`<span class="badge text-bg-danger">Fix</span>` :raw-latex:`{\small\sc [Fix]}`
.. |API| replace:: :raw-html:`<span class="badge text-bg-warning">API Change</span>` :raw-latex:`{\small\sc [API Change]}`