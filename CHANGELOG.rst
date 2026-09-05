`Unreleased`_
=============

- |Feature| Added :class:`~empulse.metrics.EmpiricalMaxProfit` and :class:`~empulse.metrics.AUEPC`
  strategies for building custom metrics that compute the empirical (convex-hull-based) maximum
  profit and the area under the empirical profit curve, respectively.
- |API| ``empc``, ``mpc``, ``empa``, ``mpa``, ``empcs``, ``mpcs``, ``empb``, ``make_objective_churn``,
  ``make_objective_acquisition``, and their supporting ``AECObjectiveChurn``, ``AECMetricChurn``,
  ``AECObjectiveAcquisition``, and ``AECMetricAcquisition`` classes have been removed.
  :func:`~empulse.metrics.empc_score`,
  :func:`~empulse.metrics.mpc_score`, :func:`~empulse.metrics.empa_score`,
  :func:`~empulse.metrics.mpa_score`, :func:`~empulse.metrics.empcs_score`,
  :func:`~empulse.metrics.mpcs_score`, :func:`~empulse.metrics.empb_score`, and
  :func:`~empulse.metrics.auepc_score` are now prebuilt :class:`~empulse.metrics.Metric`/
  :class:`~empulse.metrics.MixtureMetric` instances instead of hand-written functions.
  The decision threshold (previously returned alongside the score by the removed
  ``empc``/``mpc``/``empa``/``mpa``/``empcs``/``mpcs``/``empb`` functions) can still be obtained
  with the ``.optimal_rate(...)`` method on the corresponding ``*_score`` instance.
  Training a boosting model directly on any of these metrics (previously done through
  ``make_objective_churn``/``make_objective_acquisition``) is now done by passing the metric
  as the ``loss`` argument to :class:`~empulse.models.CSBoostClassifier`.
- |API| :func:`~empulse.metrics.expected_cost_loss_churn` and
  :func:`~empulse.metrics.expected_cost_loss_acquisition` now always return the mean cost per
  instance (previously they returned the summed cost by default, with an optional
  ``normalize=True`` argument to switch to the mean).
- |API| :func:`~empulse.metrics.empa_score`'s ``beta`` parameter now represents the *scale* of the
  Gamma-distributed contribution (mean = ``alpha * beta``), instead of the *rate*
  (mean = ``alpha / beta``) used by the previous ``empa``/``empa_score`` functions. The default
  value has been adjusted accordingly, so calls that rely on the default are unaffected; only
  explicit non-default ``beta=`` overrides behave differently.
- |API| The generic (domain-agnostic) native functions ``max_profit``, ``make_objective_aec``, and
  the supporting ``AECObjective`` and ``AECMetric`` classes have been removed.
  :func:`~empulse.metrics.max_profit_score`, :func:`~empulse.metrics.expected_cost_loss`,
  :func:`~empulse.metrics.expected_log_cost_loss`, and :func:`~empulse.metrics.expected_savings_score`
  are now prebuilt :class:`~empulse.metrics.Metric` instances instead of hand-written functions
  (:func:`~empulse.metrics.cost_loss` and :func:`~empulse.metrics.savings_score`, the hard-label
  variants that auto-threshold continuous scores, are unaffected and remain plain functions, since
  there is no :class:`~empulse.metrics.Metric`/:class:`~empulse.metrics.MetricStrategy` equivalent
  for that behavior). The decision threshold (previously returned alongside the score by the
  removed ``max_profit`` function) can still be obtained with the ``.optimal_rate(...)`` method on
  :func:`~empulse.metrics.max_profit_score`. Training a boosting or logistic-regression model
  directly on any of these metrics (previously done through ``make_objective_aec``) is now done by
  passing the metric as the ``loss`` argument to a cost-sensitive model, e.g.
  :class:`~empulse.models.CSBoostClassifier` or :class:`~empulse.models.CSLogitClassifier`.
- |API| :func:`~empulse.metrics.max_profit_score` now takes ``tp_cost``/``tn_cost`` parameters
  (costs) instead of the previous native ``max_profit``/``max_profit_score`` functions'
  ``tp_benefit``/``tn_benefit`` parameters (benefits): ``tp_cost = -tp_benefit`` and
  ``tn_cost = -tn_benefit``. ``fp_cost``/``fn_cost`` are unchanged. This matches the parameter
  naming already used throughout the rest of the package (e.g. the cost-sensitive models' default
  loss).
- |API| :func:`~empulse.metrics.expected_cost_loss` and :func:`~empulse.metrics.expected_log_cost_loss`
  now always return the mean cost per instance (previously they returned the summed cost by
  default, with an optional ``normalize=True`` argument to switch to the mean). This also affects
  the default (unweighted-``loss``) out-of-bag weighted-voting behavior of
  :class:`~empulse.models.CSForestClassifier` and :class:`~empulse.models.CSBaggingClassifier`,
  which use :func:`~empulse.metrics.expected_cost_loss` as their fallback per-estimator weight.
- |Fix| Cost-sensitive models can now be trained with a ``loss`` metric whose cost matrix names one
  of its symbols (or aliases) ``tp_cost``, ``tn_cost``, ``fp_cost`` or ``fn_cost``. Those names
  collide with the dedicated ``fit``/``predict`` parameters of the same name, so the value bound to
  the parameter and was silently discarded instead of reaching the metric, and training failed with
  ``TypeError: _lambdifygenerated() missing 1 required positional argument``. This affected the cost
  matrices shipped with :func:`~empulse.datasets.load_churn_tv_subscriptions`,
  :func:`~empulse.datasets.load_credit_scoring_pakdd` and
  :func:`~empulse.datasets.fetch_give_me_some_credit`. Such values are now routed to the metric, and
  a cost argument that the metric does not use raises a warning instead of being dropped silently.
  Note that ``__init__``-time costs are still not forwarded to a metric loss, since they default to
  ``0.0`` and would silently zero out a cost matrix term of the same name.
- |Fix| Fix :class:`~empulse.models.RobustCSClassifier` not properly handling outlier sensitive costs
  when passing a custom loss function from :class:`~empulse.metrics.Metric`.
- |Fix| Fix :func:`~empulse.metrics.empb_score` and :func:`~empulse.metrics.auepc_score` not always
  incurring the contact cost for customers who are not contacted.
- |Fix| Fix :func:`~empulse.metrics.expected_savings_score`'s (and the
  :class:`~empulse.metrics.Savings` strategy's) ``baseline='prior'`` option: it previously computed
  the baseline cost by hard-thresholding the constant prior probability (via the same logic as
  :func:`~empulse.metrics.cost_loss`), which usually made it silently degenerate to the same result
  as ``baseline='zero_one'``/``'one'``. It now correctly evaluates the expected cost of predicting
  the prior probability of the majority or minority class, whichever is cheaper, as documented.
- |Fix| :class:`~empulse.metrics.Metric` no longer shares mutable strategy state when the same
  :class:`~empulse.metrics.MetricStrategy` instance (e.g. a configured
  :class:`~empulse.metrics.MaxProfit`) is passed to more than one ``Metric``. Previously, building
  a second ``Metric`` from an already-used strategy instance silently rebuilt that strategy in
  place, so the first ``Metric`` would start returning values computed from the second metric's
  cost matrix instead of its own. Each ``Metric`` now builds and owns an independent copy of the
  strategy it is given.
- |Fix| :class:`~empulse.metrics.Metric` no longer stays linked to the mutable
  :class:`~empulse.metrics.CostMatrix` it was constructed from. Previously, modifying the
  ``CostMatrix`` object after building a ``Metric`` from it silently desynchronized the metric's
  advertised cost expressions (e.g. ``metric.fn_cost``) from what it actually computed, so a
  parameter the metric claimed to depend on could be silently ignored when scoring. ``Metric`` now
  takes an independent copy of its cost matrix at construction time.
- |Fix| :class:`~empulse.metrics.Metric` now raises a ``ValueError`` at construction time if the
  cost matrix uses a symbol name or alias reserved for internal use (``y``, ``s``, ``F_0``,
  ``F_1``, ``pi_0``, ``pi_1``, ``N``, ``i``). Previously, a colliding symbol name was either
  silently fused with the identically-named internal variable (e.g. a user symbol named ``F_0`` in
  a :class:`~empulse.metrics.MaxProfit` metric), producing a wrong score with no warning, or raised
  a confusing internal ``TypeError`` only once the metric was called (e.g. a symbol named ``y`` or
  ``s``).
- |Fix| :class:`~empulse.metrics.Metric` now raises a ``ValueError`` when called with both a
  symbol's own name and one of its aliases (or with two different aliases for the same symbol).
  Previously, whichever keyword argument was seen last silently won, so the computed score
  depended on the order the keyword arguments happened to be passed in.
- |Fix| :class:`~empulse.metrics.Metric` now raises a ``ValueError`` at construction time if an
  alias's target, or a :meth:`~empulse.metrics.CostMatrix.set_default` parameter name, does not
  match any symbol used in the cost matrix. This also catches the ordering footgun documented on
  :meth:`~empulse.metrics.CostMatrix.set_default`, where a default keyed by an alias name that was
  set *before* the alias was registered used to be silently stored under that raw (untranslated)
  name and then silently ignored, instead of ever being applied.
- |Fix| :class:`~empulse.metrics.Metric` now validates ``y_true``/``y_score``.
- |Fix| :class:`~empulse.metrics.Metric` now raises a ``ValueError`` naming the offending
  parameter when an instance-dependent (array-like) ``cost_matrix`` parameter's length doesn't
  match ``y_true``/``y_score`` and isn't a single value broadcastable to every sample. Previously
  a length mismatch surfaced as an unrelated numpy broadcasting error, e.g. ``operands could not
  be broadcast together with shapes (3,) (5,)``, that didn't say which parameter was wrong.
- |Fix| :meth:`Metric.optimal_threshold() <empulse.metrics.Metric.optimal_threshold>` and
  :meth:`~empulse.metrics.Metric.optimal_rate` now raise a ``ValueError`` when the cost matrix is
  degenerate for the given parameters (``fp_cost + tn_benefit + fn_cost + tp_benefit`` evaluates
  to 0, making the optimal threshold undefined).
- |Fix| :class:`~empulse.models.CSThresholdClassifier` and :class:`~empulse.models.CSRateClassifier`
  no longer silently skip learning a cost-sensitive threshold/rate when fitted with an aliased
  :class:`~empulse.metrics.Metric` with every required parameter supplied through its alias.
- |Fix| :class:`~empulse.models.CSBaggingClassifier` with ``combination='weighted_voting'`` no
  longer raises a shape error when a sub-estimator draws a feature subset (``max_features < 1.0``
  or ``bootstrap_features=True``).
- |Fix| :class:`~empulse.models.CSForestClassifier` with ``combination='weighted_voting'`` no
  longer produces ``NaN`` out-of-bag estimator weights when ``max_samples`` is an ``int``.
  ``numbers.Integral`` is a subclass of ``numbers.Real``.
- |Fix| Out-of-bag weighted voting (:class:`~empulse.models.CSForestClassifier` and
  :class:`~empulse.models.CSBaggingClassifier` with ``combination='weighted_voting'``) no longer
  gives *more* weight to *worse* estimators, and no longer raises when instance-dependent
  (array-like) costs are used together with weighted voting.
- |Fix| :class:`~empulse.models.CSTreeClassifier` and  :class:`~empulse.models.CSForestClassifier`
  no longer raises when fitted with a metric strategy containing a stochastic (``sympy.stats``) variable.
- |Fix| :class:`~empulse.models.CSBoostClassifier` no longer mutates a caller-supplied
  ``fit_params`` dict in place.
- |Fix| :class:`~empulse.models.CSTreeClassifier` no longer mutates a user-supplied custom
  ``criterion`` instance in place.
- |Fix| :class:`~empulse.models.CSBoostClassifier`'s LightGBM and CatBoost backends now start
  from the intended probability. The internal base-score nudge was applied as a raw (log-odds)
  score for LightGBM's ``init_score``/CatBoost's ``baseline`` but as a probability for XGBoost's
  ``base_score``, and neither LightGBM nor CatBoost persist that offset into the saved model, so
  ``predict_proba`` now adds it back manually before converting to a probability.
- |Fix| :class:`~empulse.models.BiasRelabelingClassifier`, :class:`~empulse.models.BiasResamplingClassifier`,
  and :class:`~empulse.models.BiasReweighingClassifier` now raise a clear ``ValueError`` when
  ``sensitive_feature`` does not have the same length as ``y``, instead of silently proceeding.
- |Fix| :class:`~empulse.models.CSThresholdClassifier` and :class:`~empulse.models.CSRateClassifier`
  no longer mutate a user-supplied ``calibrator`` estimator instance in place; a clone is
  configured and fitted instead, so the constructor argument is safe to reuse or refit.
- |Fix| :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.ProfLogitClassifier`
  documented ``soft_threshold`` as defaulting to ``False``; the actual default is ``True``.

`0.11.1`_ (08-05-2026)
======================

- |Fix| Fix empulse not running properly on scikit-learn versions lower than 1.6.

`0.11.0`_ (08-05-2026)
======================

- |Feature| Added :class:`~empulse.models.ProfTreeClassifier` to optimize a cost-sensitive metric
  using evolutionary trees (similar to :class:`~empulse.models.ProfLogitClassifier`).
- |Feature| Added :class:`~empulse.models.CSRateClassifier` to optimize for a specific predicted positive rate.
- |Feature| Allow :class:`~empulse.models.CSThresholdClassifier` to optimize the decision threshold at training time.
  Users can now either specify the decision threshold to use at prediction time or
  let the model choose the optimal threshold during training.
- |Feature| (Experimental)
  When using the :class:`~empulse.metrics.MaxProfit` strategy, only one stochastic variable is present,
  and the cost/profit function is a polynomials of the stochastic variable:
  the metric will now be computed exactly instead of using numerical integration.
  This is currently supported for stochastic variables following
  the Normal, Log Normal, Uniform, Beta, Gamma, Chi Squared, Exponential, Weibull, Pareto, and Triangular distributions.
- |Feature| (Experimental) Added support for the MaxProfit strategy metrics to be optimized
  through gradient descent methods in
  :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.CSBoostClassifier`.
  This is currently an experimental feature and is not recommended for use in production.
- |API| Updated the :class:`~empulse.models.ProfLogitClassifier` interface to be more consistent
  with other models in the package. By default optimizes the maximum profit metric.
- |API| :class:`~empulse.models.CSLogitClassifier` no longer takes a string argument for the loss function
  to be more consistent with other models in the package. Default value for the loss is None.
- |API| :class:`~empulse.metrics.MaxProfit` now takes a numpy Generator instead of a RandomState instance.
- |Enhancement| Metrics built with the :class:`~empulse.metrics.MaxProfit`
  strategy can now handle instance-dependent costs. They will automatically be averaged over the instances.
  Mathematically this is equivalent to recomputing the EMP score for each instance and then averaging the scores.
- |Enhancement| Metrics built with the :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings`
  strategies can now handle stochastic cost parameters. If the distribution allows it, the mean cost will be computed.
- |Fix| Fix :class:`~empulse.models.CSTreeClassifier` and :class:`~empulse.models.CSForestClassifier`
  not properly training when costs were negative.
- |Fix| Fix integration bounds inconsistently being calculated
  when the :class:`~empulse.metrics.MaxProfit` strategy was chosen.
- |Fix| Fix :class:`~empulse.models.CSBoostClassifier` throwing errors
  when one or two of the Boosting libraries were not installed (XGBoost, LGBM & Catboost).
- |Fix| Add __name__ attribute to :class:`~empulse.metrics.Metric` class to fix issues with scikit-learn compatibility.
- |Fix| Fix metadata routing not working for scikit-learn>=1.8.0
- |Fix| Fix :class:`~empulse.metrics.MaxProfit` strategy not calculating Log Normal distributed variables correctly
  when using quasi monte carlo.
- |Fix| Fix some models not properly being able to be pickled when using a custom metric as the loss function.
- |Fix| Fix some distributions not correctly computing the expected maximum profit score when using the
  :class:`~empulse.metrics.MaxProfit` strategy when using monte carlo or quasi monte carlo method.

`0.10.4`_ (20-09-2025)
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


.. _Unreleased: https://github.com/ShimantoRahman/empulse/compare/0.11.1...main
.. _0.11.1: https://github.com/ShimantoRahman/empulse/releases/tag/0.11.1
.. _0.11.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.11.0
.. _0.10.4: https://github.com/ShimantoRahman/empulse/releases/tag/0.10.4
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