`Unreleased`_
=============

- ...

`0.6.0`_ (28-01-2025)
=====================

- |MajorFeature| Add :class:`~empulse.metrics.Metric` to easily build your own value-driven and cost-sensitive metrics
- |Feature| Add support for LightGBM and Catboost models in :class:`~empulse.models.CSBoostClassifier` and
  :class:`~empulse.models.B2BoostClassifier`
- |API| :func:`~empulse.metrics.make_objective_churn` and :func:`~empulse.metrics.make_objective_acquisition`
  now take a ``model`` argument to calculate the objective for either XGBoost, LightGBM or Catboost models.
- |API| XGBoost is now an optional dependency together with LightGBM and Catboost. To install the package with
  XGBoost, LightGBM and Catboost support, use the following command: ``pip install empulse[optional]``
- |API| Renamed ``y_pred_baseline`` and ``y_proba_baseline`` to ``baseline`` in :func:`~empulse.metrics.savings_score`
  and :func:`~empulse.metrics.expected_savings_score`. It now accepts the following arguments:

  - If ``'zero_one'``, the baseline model is a naive model that predicts all zeros or all ones
    depending on which is better.
  - If ``'prior'``, the baseline model is a model that predicts the prior probability of
    the majority or minority class depending on which is better (not available for savings score).
  - If array-like, target probabilities of the baseline model.

- |Feature| Add parameter validation for all models and samplers
- |API| Make all arguments of dataset loaders keyword-only
- |Fix| Update the descriptions attached to each dataset to match information found in the user guide
- |Fix| Improve type hints for functions and classes

`0.5.2`_ (12-01-2025)
=====================
- |Feature| Allow :func:`~empulse.metrics.savings_score` and :func:`~empulse.metrics.expected_savings_score`
  to calculate the savings score over the baseline model instead of a naive model,
  by setting the ``y_pred_baseline`` and ``y_proba_baseline`` parameters, respectively.
- |Enhancement| Reworked the user guide documentation to better explain the usage of value-driven
  and cost-sensitive models, samplers and metrics
- |API| :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.ProfLogitClassifier`
  by default do not perform soft-thresholding on the regression coefficients.
  This can be enabled by setting the ``soft_threshold`` parameter to True.
- |Fix| Prevent division by zero errors in :func:`~empulse.metrics.expected_cost_loss`

`0.5.1`_ (05-01-2025)
=====================
- |Fix| Fixed documentation build issue

`0.5.0`_ (05-01-2025)
=====================

- |MajorFeature| Added supported for python 3.13
- |MajorFeature| Added cost-sensitive models
    - :class:`~empulse.models.CSLogitClassifier`
    - :class:`~empulse.models.CSBoostClassifier`
    - :class:`~empulse.models.RobustCSClassifier`
    - :class:`~empulse.models.CSThresholdClassifier`
- |MajorFeature| Added cost-sensitive metrics
    - :func:`~empulse.metrics.cost_loss`
    - :func:`~empulse.metrics.expected_cost_loss`
    - :func:`~empulse.metrics.expected_log_cost_loss`
    - :func:`~empulse.metrics.savings_score`
    - :func:`~empulse.metrics.expected_savings_score`
- |MajorFeature| Added :mod:`empulse.datasets` module
- |Feature| Added :class:`~empulse.samplers.CostSensitiveSampler`
- |Enhancement| Allow all cost-sensitive models and samplers to accept cost parameters during initialization
- |API| Renamed metric arguments which expect target score from y_pred to y_score and
  target probabilities from y_pred to y_proba


.. _Unreleased: https://github.com/ShimantoRahman/empulse/compare/0.6.0...main
.. _0.6.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.6.0
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