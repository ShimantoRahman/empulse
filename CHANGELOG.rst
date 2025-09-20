`Unreleased`_
=============

`0.10.3`_ (20-09-2025)
======================

- |Fix| Fix build issue on Windows.

`0.10.0`_ (20-09-2025)
======================

- |Efficiency| Changed to Cython implementation for the loss functions and impurity measures of
  :class:`~empulse.models.CSLogitClassifier`, :class:`~empulse.models.CSBoostClassifier`,
  :class:`~empulse.models.CSTreeClassifier`, and :class:`~empulse.models.CSForestClassifier`.
  This improves the training time and memory efficiency of these models significantly.
  Training time speedups observed were
  up to 300x for :class:`~empulse.models.CSTreeClassifier` and :class:`~empulse.models.CSForestClassifier`,
  30x for :class:`~empulse.models.CSLogitClassifier`, and 1.5x for :class:`~empulse.models.CSBoostClassifier`
  depending on the dataset size and parameters.
- |API| Changed arguments to :class:`~empulse.models.CSTreeClassifier`, :class:`~empulse.models.CSForestClassifier`, and
  :class:`~empulse.models.CSBaggingClassifier` to be in line with scikit-learn's decision tree and ensemble models.
- |API| :class:`~empulse.models.CSForestClassifier`, and :class:`~empulse.models.CSBaggingClassifier`
  no longer support stacking combination method. Use :class:`~sklearn.ensemble.StackingClassifier` instead for stacking.
- |API| Extracted the construction of the cost matrix into a separate class
  :func:`~empulse.metrics.CostMatrix` away from :class:`~empulse.metrics.Metric`
  to allow reusing the cost matrix in custom metrics.
- |API| :class:`~empulse.models.ProfLogitClassifier` no longer uses the EMPC metric by default.
  Users now need to explicitely pass a loss to the model.
- |API| :class:`~empulse.models.CSLogitClassifier` no longer accepts any callable as loss function.
  Users now need to pass a :class:`~empulse.metrics.Metric` instance for a custom loss function.
- |Feature| :func:`~empulse.metrics.savings_score` and :func:`~empulse.metrics.expected_savings_score`
  now accept two more baseline options `'one'` and `'zero'`
  to always predict the positive and negative class, respectively.
- |Feature| Metrics with with the :class:`~empulse.metrics.Savings` strategy now also accepts baseline options like
  :func:`~empulse.metrics.savings_score` and :func:`~empulse.metrics.expected_savings_score`.
- |Enhancement| Models which use a :class:`~empulse.metrics.Metric` instance as their loss function
  with the :class:`~empulse.metrics.Cost` or :class:`~empulse.metrics.Savings`
  strategy as their loss function now are pickleable.
  The :class:`~empulse.metrics.MaxProfit` strategy will be updated to be pickleable in a future release.
- |Enhancement| Models which use a :class:`~empulse.metrics.Metric` instance as their loss function
  can now request arguments necessary for the metric to be passed during the fit method through Metadata Routing.
- |Fix| Fix :class:`~empulse.models.CSLogitClassifier` not properly calculating gradient penalty.
- |Fix| Fix default values not being properly when using aliases in :class:`~empulse.metrics.CostMatrix`.
- |Fix| Fix :class:`~empulse.metrics.Metric` throwing errors when certain terms cancelled out.

`0.9.0`_ (15-06-2025)
=====================

- |Feature| Added :meth:`~empulse.metrics.Metric.optimal_threshold` and
  :meth:`~empulse.metrics.Metric.optimal_rate` methods to calculate the optimal threshold(s)
  and optimal predicted positive rate for a given metric.
  This is useful for determining the best decision threshold and predicted positive rate
  for a cost-sensitive or value-driven model.
- |Feature| :class:`~empulse.models.CSTreeClassifier`, :class:`~empulse.models.CSForestClassifier`, and
  :class:`~empulse.models.CSBaggingClassifier` can now take
  a :class:`~empulse.metrics.Metric` instance as their criterion to optimize.
- |Feature| :class:`~empulse.models.CSThresholdClassifier` can now take
  a :class:`~empulse.metrics.Metric` instance to choose the optimal decision threshold.
- |Feature| :class:`~empulse.models.RobustCSClassifier` can now take estimators with a
  :class:`~empulse.metrics.Metric` instance as the loss function or criterion.
  :class:`~empulse.models.RobustCSClassifier` will treat any cost marked as outlier sensitive.
  This can be done by using the :meth:`~empulse.metrics.Metric.mark_outlier_sensitive` method.
- |Feature| Allow savings metrics to be used in :class:`~empulse.models.CSBoostClassifier` and
  :class:`~empulse.models.CSLogitClassifier` as the objective function.
  Internally, the expected cost loss is used to train the model,
  since the expected savings score is just a transformation of the expected cost loss.
- |API| `kind` argument to :class:`~empulse.metrics.Metric` has been replaced by `strategy`.
  The :class:`~empulse.metrics.Metric` class now takes a :class:`~empulse.metrics.MetricStrategy` instance.
  This change allows for more flexibility in defining the metric strategy.
  The currently available strategies are:

    - :class:`~empulse.metrics.MaxProfit` for the expected maximum profit score
    - :class:`~empulse.metrics.Cost` for the expected cost loss
    - :class:`~empulse.metrics.Savings` for the expected savings score

- |Fix| Fix error when importing Empulse without any optional dependencies installed.
- |Fix| Fix :class:`~empulse.models.CSLogitClassifier` not properly using the gradient
  when using a custom loss function from :class:`~empulse.metrics.Metric`.
- |Fix| Fix models throwing errors when differently shaped costs are passed to the fit or predict method.
- |Fix| Fix sympy distribution parameters not being properly translated to scipy distribution parameters when
  using the :class:`~empulse.metrics.MaxProfit` strategy (formerly `kind='max profit'`)
  with the quasi monte-carlo integration method.

`0.8.0`_ (01-06-2025)
=====================

- |Feature| :class:`~empulse.models.CSBoostClassifier`, :class:`~empulse.models.CSLogitClassifier`, and
  :class:`~empulse.models.ProfLogitClassifier` can now take
  a :class:`~empulse.metrics.Metric` instance as their loss function.
  Internally, the metric instance is converted to the appropriate loss function for the model.
  For more information, read the :ref:`User Guide <metric_class_in_model>`.
- |Feature| Type hints are now available for all functions and classes.
- |Enhancement| Add support for more than one stochastic variable when building maximum profit metrics with
  :class:`~empulse.metrics.Metric`
- |Enhancement| Allow :class:`~empulse.metrics.Metric` to be used as a context manager.
  This ensures the metric is always built after defining the cost-benefit elements.
- |Fix| Fix datasets not properly being packaged together with the package
- |Fix| Fix :class:`~empulse.models.RobustCSClassifier` when array-like parameters are passed to fit method.
- |Fix| Fix boosting models being biased towards the positive class.

`0.7.0`_ (05-02-2025)
=====================

- |MajorFeature| Add :class:`~empulse.models.CSTreeClassifier`, :class:`~empulse.models.CSForestClassifier`,
  and :class:`~empulse.models.CSBaggingClassifier` to support cost-sensitive decision tree and ensemble models
- |Enhancement| Add support for scikit-learn 1.5.2 (previously Empulse only supported scikit-learn 1.6.0 and above).
- |API| Removed the ``emp_score`` and ``emp`` functions from the :mod:`~empulse.metrics` module.
  Use the :func:`~empulse.metrics.Metric` class instead to define custom expected maximum profit measures.
  For more information, read the :ref:`User Guide <user_defined_value_metric>`.
- |API| Removed numba as a dependency for Empulse. This will reduce the installation time and the size of the package.
- |Fix| Fix :func:`~empulse.metrics.Metric` when defining stochastic variable with fixed values.
- |Fix| Fix :func:`~empulse.metrics.Metric` when stochastic variable has infinite bounds.
- |Fix| Fix :func:`~empulse.models.CSThresholdClassifier`
  when costs of predicting positive and negative classes are equal.
- |Fix| Fix documentation linking issues to sklearn

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


.. _Unreleased: https://github.com/ShimantoRahman/empulse/compare/0.10.3...main
.. _0.10.3: https://github.com/ShimantoRahman/empulse/releases/tag/0.10.3
.. _0.10.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.10.0
.. _0.9.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.9.0
.. _0.8.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.8.0
.. _0.7.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.7.0
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