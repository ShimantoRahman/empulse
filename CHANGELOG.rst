`Unreleased`_
=============

Metrics
-------

- |Fix| :class:`~empulse.metrics.MaxProfit` with stochastic variables now works when the metric's
  symbols were declared with assumptions, e.g. ``sympy.symbols('a b', positive=True)``. Parameter
  values were substituted into the expressions by name, which sympy turns into symbols without
  assumptions that do not match the metric's own, so they were left in place. Depending on the
  integration method, this raised a ``TypeError`` for a distribution whose support depends on its
  parameters (such as a uniform one), for the ``'quad'``, ``'monte-carlo'`` and
  ``'quasi-monte-carlo'`` integration methods, and for profits that are not polynomial in the
  stochastic variable. Distribution parameters with assumptions also skipped part of their
  validation.
- |Fix| :class:`~empulse.metrics.MaxProfit` with stochastic variables now accepts distributions
  whose parameters are expressions, e.g. ``sympy.stats.Beta('v', 2 * a, b)``. Each argument of the
  distribution was treated as the name of a parameter, so the metric raised
  ``ValueError: Metric expected a value for 2*a`` whatever values were passed. It now asks for the
  symbols the arguments are built from (here ``a`` and ``b``).
- |Fix| :class:`~empulse.metrics.MaxProfit` with a single stochastic variable and the default
  ``integration_method='auto'`` no longer raises ``KeyError`` when a parameter of the variable's
  distribution also appears elsewhere in the cost matrix, e.g.
  ``sympy.stats.Gamma('v', a, b) * clv - a``. The distribution's parameters were removed from the
  values used for the rest of the profit.
- |Fix| Deterministic :class:`~empulse.metrics.MaxProfit` metrics (and so
  :func:`~empulse.metrics.mpc_score`, :func:`~empulse.metrics.mpa_score`,
  :func:`~empulse.metrics.mpcs_score` and :func:`~empulse.metrics.empcs_score`) no longer return
  ``nan`` on data with a single class. Their score is now the better of targeting no one and
  targeting everyone, and their optimal rate and threshold follow from it.
- |Fix| :class:`~empulse.metrics.MaxProfit` with stochastic variables (and so
  :func:`~empulse.metrics.empc_score` and :func:`~empulse.metrics.empa_score`) now scores data with
  a single class as the better of targeting no one and targeting everyone. It only considered
  targeting no one, so e.g. :func:`~empulse.metrics.empc_score` scored ``0.0`` rather than ``56.0``
  when every customer churns.
- |Fix| :class:`~empulse.metrics.MaxProfit` with stochastic variables (and so
  :func:`~empulse.metrics.empc_score`, :func:`~empulse.metrics.empa_score` and
  :func:`~empulse.metrics.empcs_score`) now raises a ``ValueError`` when ``y_true`` and ``y_score``
  are empty or differ in length. Empty inputs crashed the Python interpreter with a segmentation
  fault, and inputs of different lengths raised an unhelpful ``IndexError``.
- |Fix| :class:`~empulse.metrics.MaxProfit` with stochastic variables (and so
  :func:`~empulse.metrics.empc_score`, :func:`~empulse.metrics.empa_score` and
  :func:`~empulse.metrics.empcs_score`) no longer underestimates the maximum profit on large
  datasets. The ROC convex hull it integrates over treated any turn of the curve smaller than a fixed
  tolerance as a straight line, but the points of a curve of n samples lie about 1/n apart, so from
  roughly 10,000 samples on it dropped vertices of the hull: scores on 100,000 samples could come out
  about 1% too low. The hull is now computed exactly from the integer counts of the curve. This also
  affects the optimal rate and threshold of these metrics, the training objectives of
  :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.CSBoostClassifier` with
  them, and :class:`~empulse.models.ProfTreeClassifier` with a stochastic
  :class:`~empulse.metrics.MaxProfit` loss. Results on smaller datasets are normally unchanged.
- |Fix| :class:`~empulse.metrics.EmpiricalMaxProfit`, :class:`~empulse.metrics.EmpiricalMinCost`
  and :class:`~empulse.metrics.AUEPC` (and so :func:`~empulse.metrics.empb_score` and
  :func:`~empulse.metrics.auepc_score`) now handle tied scores. No threshold can separate samples
  with equal scores, so each group of ties is now targeted all at once, as
  :class:`~empulse.metrics.MaxProfit` already did. Previously the profit curve was accumulated one
  sample at a time, so the result depended on the order the tied samples happened to be in: the
  same data could score 84.0 or 9.0 with :func:`~empulse.metrics.empb_score`. AUEPC now scores the
  expected profit curve under random tie-breaking. Scores on data with ties change as a result;
  scores on data without ties are unchanged.
- |Fix| :func:`~empulse.metrics.auepc_score` and :class:`~empulse.metrics.AUEPC` no longer raise
  ``ZeroDivisionError`` when the oracle's cumulative profit turns negative after its first sample.
  A curve of a single point now scores that point's ratio. The curve also stops where the oracle's
  profit reaches exactly zero, rather than dividing by that zero, and a dataset in which no sample
  is profitable scores ``0.0``.
- |Fix| :class:`~empulse.metrics.MaxProfit` with a single stochastic variable no longer raises
  ``IndexError`` when the variable's distribution mixes numeric and symbolic parameters (e.g.
  ``sympy.stats.Beta('gamma', 6, beta)``) or has two equal numeric parameters (e.g.
  ``sympy.stats.Beta('gamma', 6, 6)``). This affected the score, the optimal rate and threshold,
  and the training objectives of :class:`~empulse.models.CSLogitClassifier` and
  :class:`~empulse.models.CSBoostClassifier`.
- |Fix| :class:`~empulse.metrics.MaxProfit` with ``integration_method='quasi-monte-carlo'`` now
  samples :func:`~sympy.stats.Arcsin` and :func:`~sympy.stats.PowerFunction` random variables over
  their correct support. Their upper bound was passed to SciPy as the width of the support, so
  ``Arcsin('x', 2, 5)`` was sampled on ``[2, 7]`` instead of ``[2, 5]``.
- |Fix| :class:`~empulse.metrics.MaxProfit` with ``integration_method='quad'`` and four or more
  stochastic variables now integrates each variable over its own support. The variables were
  paired with the supports in reverse order.
- |Fix| :func:`~empulse.metrics.lift_score` no longer raises ``ZeroDivisionError`` when
  ``fraction * n_samples`` rounds to zero; the top fraction now always contains at least one
  sample. ``fraction=0`` is now rejected with a ``ValueError``.
- |Fix| :meth:`~empulse.metrics.MixtureMetric.__call__`,
  :meth:`~empulse.metrics.MixtureMetric.optimal_rate` and
  :meth:`~empulse.metrics.MixtureMetric.optimal_threshold` now pass ``validate`` on to their
  component metrics, which previously re-validated their parameters on every call.
- |Fix| The documentation of :func:`~empulse.metrics.auepc_score` no longer lists
  ``optimal_threshold`` and ``optimal_rate``, which AUEPC does not support. Use
  :func:`~empulse.metrics.empb_score`'s methods instead.
- |Efficiency| A compiled cost expression no longer re-derives the names of its free symbols every
  time it is evaluated; they are computed once per expression. This matters where the rest of an
  evaluation is cheap, such as scoring a :class:`~empulse.models.ProfTreeClassifier` tree from its
  leaves with a stochastic :class:`~empulse.metrics.MaxProfit` metric (about 10% faster).
- |Efficiency| The ROC convex hull behind :class:`~empulse.metrics.MaxProfit` with stochastic
  variables (and so :func:`~empulse.metrics.empc_score`, :func:`~empulse.metrics.empa_score` and
  :func:`~empulse.metrics.empcs_score`) is computed 8-16x faster: in about 3 ms instead of 27 ms on
  100,000 samples, and in about 1.5 us instead of 22 us on 20. It no longer sorts the curve's points
  a second and third time, sorts the samples with an unstable sort, and runs in C++ without calling
  back into NumPy except to sort large inputs. The hull took nearly all the time of these scores on
  large datasets and up to a fifth of each gradient boosting round of
  :class:`~empulse.models.CSBoostClassifier` with them; a
  :class:`~empulse.models.ProfTreeClassifier` generation with them is about 5% faster.
- |Efficiency| :class:`~empulse.metrics.MaxProfit` with one stochastic variable (and so
  :func:`~empulse.metrics.empc_score`, :func:`~empulse.metrics.empa_score` and
  :func:`~empulse.metrics.empcs_score`) scores a ROC convex hull 10-25x faster, in about 15 us
  instead of 150-400 us for a hull of 20 points. Splitting the variable's support into the regions
  where one threshold is optimal and integrating the profit over each now runs in C++, for the ten
  distributions with closed-form partial moments and profits up to quadratic in the variable; other
  profits and distributions are scored as before. Scoring the hull took most of the time of fitting
  a :class:`~empulse.models.ProfTreeClassifier` with such a metric, which is now about 3x faster.
- |Efficiency| A :class:`~empulse.metrics.Metric` now finds the names of its parameters once rather
  than on every call, which walked all four cost expressions each time. Calls on 1,000 samples are
  about 10-40% faster; the names are found again if the metric's cost matrix is changed.
- |Efficiency| A :class:`~empulse.metrics.Metric` called with ``validate=False`` now also skips the
  checks of the labels, which the models that pass it have already checked when they were fitted,
  and only checks that the scores are finite. This makes the metric evaluations inside training
  loops (such as :class:`~empulse.models.ProfTreeClassifier`, :class:`~empulse.models.ProfSRClassifier`
  and the out-of-bag weighting of :class:`~empulse.models.CSForestClassifier` and
  :class:`~empulse.models.CSBaggingClassifier`) 1.7-3.8x faster on up to 10,000 samples, together
  with the entry above. Results are unchanged, and scores that are ``nan`` or infinite still raise.
- |Efficiency| The compiled logistic losses behind cost metrics are 1.3-6.6x faster. They drive
  :class:`~empulse.models.CSLogitClassifier`, :class:`~empulse.models.ProfLogitClassifier` with a
  cost loss, and the gradient boosting of :class:`~empulse.models.CSBoostClassifier`. Their matrix
  products now go to BLAS, and their exponentials are computed by numpy's vectorized ``exp`` for
  all samples at once. Fitting ``CSLogitClassifier`` on 100,000 samples with 50 features went from
  0.81 to 0.25 seconds, and ``ProfLogitClassifier`` on 10,000 samples with 20 features from 2.3 to
  1.1 seconds. The derivatives are also exact where the predicted probability is close to 1. They
  used to be computed as ``p * (1 - p)``, which rounds to 0 once the margin exceeds about 37.

Models
------

- |Fix| :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.ProfLogitClassifier`
  with a cost loss now fit read-only data, such as the memory-mapped arrays joblib passes to
  parallel workers in ``GridSearchCV(n_jobs=...)``. With ``fit_intercept=False`` they raised
  ``ValueError: buffer source array is read-only``.
- |Fix| :class:`~empulse.models.CSBoostClassifier` and :class:`~empulse.models.B2BoostClassifier`
  with a ``CatBoostClassifier`` estimator no longer train on distorted sample weights. The backend
  passed each row's index as its sample weight, so that the objective could look up that row's
  costs, but CatBoost also trains on the weights: the first row was ignored, later rows counted
  progressively more, and the model depended on the order of the training rows. Training now uses
  an equivalent weighted problem, with each row weighted by how much its costs depend on the
  prediction. On one example the test-set expected cost fell from 2.02 to 0.83.
- |Enhancement| :class:`~empulse.models.CSBoostClassifier` now accepts ``sample_weight`` with a
  ``CatBoostClassifier`` estimator.
- |API| :class:`~empulse.models.CSBoostClassifier` with a ``CatBoostClassifier`` estimator now
  raises a ``ValueError`` for :class:`~empulse.metrics.MaxProfit` and
  :class:`~empulse.metrics.LogCost` losses. CatBoost computes the objective on chunks of the
  training rows: MaxProfit's gradient depends on every row, so it was computed incorrectly, and
  LogCost relied on the row indices removed above. Use an ``XGBClassifier`` or ``LGBMClassifier``
  estimator for these losses. It also raises a ``ValueError`` when the costs make the same
  prediction strictly cheapest for every training sample, as CatBoost cannot train on a single
  class.
- |Fix| :class:`~empulse.models.CSForestClassifier` (with the default ``bootstrap=True``) and
  :class:`~empulse.models.CSTreeClassifier` (with ``class_weight``) now build correct trees with
  the cost-sensitive criteria. The criteria ignored sample weights when totalling a node's costs
  but not when splitting it, so the right child's costs were wrong. Forests pass their bootstrap
  draws as sample weights, so every bootstrapped tree was affected: a tree fitted on a bootstrap
  sample disagreed with one fitted on the same rows written out on 80% of its predictions.
- |Fix| :class:`~empulse.models.CSTreeClassifier` and :class:`~empulse.models.CSForestClassifier`
  with ``criterion='entropy'`` or ``criterion='log_loss'`` now grow past a single split. The criterion's child impurities were
  missing a minus sign and came out negative, and scikit-learn stops splitting any node whose
  impurity is not positive, so every tree stopped at depth 1 whatever ``max_depth`` was.
- |Fix| :class:`~empulse.models.ProfTreeClassifier` now minimizes a custom ``loss`` instead of
  maximizing it. This affected every loss other than a deterministic
  :class:`~empulse.metrics.MaxProfit` metric, including stochastic MaxProfit metrics such as
  :func:`~empulse.metrics.empc_score`. With an expected-cost loss, the fitted tree used to be a
  single leaf predicting the class prior, costing more than predicting 0.5 for every sample.
- |Fix| :class:`~empulse.models.ProfTreeClassifier` no longer crashes the Python process when a
  feature has a single distinct value. Drawing a split for such a feature took a random integer
  modulo zero, which raised a floating-point exception (SIGFPE) in C. Splits are now drawn from the
  features with at least two distinct values; if there are none, the tree is a single leaf.
- |Fix| :class:`~empulse.models.ProfTreeClassifier`'s ``alpha`` complexity penalty now counts the
  tree's actual nodes. The count was updated incrementally and drifted: it grew when a split was
  refused at ``max_depth`` and did not shrink when nodes were pruned for violating
  ``min_samples_split`` or ``min_samples_leaf``, so a fitted tree of 5 nodes could be penalized
  as 25. Fits with the default ``alpha=0`` are unaffected.
- |Fix| :meth:`CSTreeClassifier.predict <empulse.models.CSTreeClassifier.predict>` now returns the
  original class labels. It returned the 0/1 encoding the inner tree is fitted on, so with labels
  such as ``'no'``/``'yes'`` or ``-1``/``1`` it predicted ``0`` and ``1``, values not in ``classes_``.
- |Fix| ``pos_label`` now makes its class the positive class in
  :class:`~empulse.models.CSThresholdClassifier` and :class:`~empulse.models.CSRateClassifier`:
  the class the costs refer to, whose probability is thresholded or ranked, and which is predicted
  above the threshold or within the targeted fraction. The threshold was computed for
  ``classes_[1]`` but scores above it were given the ``pos_label`` class, so ``pos_label=classes_[0]``
  inverted every prediction of :class:`~empulse.models.CSThresholdClassifier`, and
  :class:`~empulse.models.CSRateClassifier` ignored ``pos_label``. A ``pos_label`` that is not one of
  the classes now raises a ``ValueError``.
- |Fix| :class:`~empulse.models.BiasRelabelingClassifier` (and :class:`~empulse.samplers.BiasRelabler`)
  and :class:`~empulse.models.BiasReweighingClassifier` now work with labels other than ``0``/``1``.
  The relabeler chose which samples to relabel by comparing the labels to ``0`` and ``1`` and
  relabelled them with those literal values, so ``-1``/``1`` labels gained a third class ``0`` and
  ``predict`` raised an ``IndexError``. The reweighing classifier computed its weights from the raw
  labels, so ``-1``/``1`` labels got different weights and string labels raised a ``TypeError``.
  Both now work on the 0/1 encoding of the target, with ``classes_[1]`` as the positive class; a
  custom reweighing ``strategy`` now receives that encoding too.
- |Fix| :class:`~empulse.models.RobustCSClassifier` now detects outliers in the costs of both
  classes whatever the labels are. It selected the positive samples with ``y > 0`` and the negative
  ones with ``y == 0``, so with labels such as ``-1``/``1`` or ``'no'``/``'yes'`` the costs of the
  negative class (``fp_cost``, ``tn_cost``, and metric parameters that only affect them) were
  silently never cleaned, and with labels such as ``2``/``5`` every sample counted as positive. The
  greater of the two labels is now the positive class, as in the cost-sensitive models it wraps.
- |Fix| :class:`~empulse.models.CSForestClassifier` now rejects a metric passed as ``criterion``
  during parameter validation, like :class:`~empulse.models.CSTreeClassifier`, instead of accepting
  it and failing later in ``fit`` with an "Unknown criterion" error. Like
  :class:`~empulse.models.CSTreeClassifier`, it now also accepts a cost impurity instance.
- |Fix| :class:`~empulse.models.ProfTreeClassifier`'s early stopping (``patience`` and
  ``tolerance``) now works when the fitness is negative, as it is with costs but no benefits, or with
  a custom ``loss``. A new best tree had to beat ``fitness * (1 + tolerance)``, which is below the
  current best when the fitness is negative, so a tie counted as an improvement, the patience never
  ran out and every fit ran for ``max_iter`` generations. The tolerance is now scaled by the
  magnitude of the fitness; fits with a positive fitness are unaffected.
- |Efficiency| :class:`~empulse.models.ProfTreeClassifier` fits roughly 8x faster with its default
  maximum profit fitness (no ``loss``, or a deterministic :class:`~empulse.metrics.MaxProfit`
  metric). Each candidate tree's maximum profit is now computed from its leaf counts: every sample
  in a leaf gets the same score, so ranking the leaves gives the same ROC curve as predicting and
  sorting every training sample, which is what each evaluation did before. Routing samples through
  the tree also no longer creates a memoryview per sample. The fitted trees are unchanged.
- |Efficiency| :class:`~empulse.models.ProfTreeClassifier` now uses ``n_jobs``, which it accepted
  but ignored: each generation's trees are fitted, and with the default maximum profit fitness also
  evaluated, in parallel over ``n_jobs`` threads (about 3x faster on 4 cores). The random variations
  are still drawn serially, so the fitted tree does not depend on ``n_jobs``. ``n_jobs`` now also
  accepts ``None`` and negative values, which count back from the number of processors as in
  scikit-learn (``-1`` uses all of them). The package falls back to a single thread when it is built
  without OpenMP; set ``EMPULSE_DISABLE_OPENMP=1`` to build it that way deliberately.
- |Efficiency| :class:`~empulse.models.ProfTreeClassifier` refits only what changed. Each new tree
  is a copy of a fitted tree with one subtree changed by crossover, growing or mutating a split, so
  only the samples that reach that subtree are routed through it again, and pruning a split needs
  no refit at all. This makes each generation about 1.7x faster; the fitted trees are unchanged.
- |Efficiency| :class:`~empulse.models.ProfTreeClassifier` with a stochastic
  :class:`~empulse.metrics.MaxProfit` or :class:`~empulse.metrics.MinCost` metric as ``loss`` (e.g.
  the expected maximum profit) fits about 4x faster. The metric depends on a tree's predictions only
  through the ROC convex hull and the class prior, which follow from each leaf's numbers of positive
  and negative samples, so each tree is now scored from its leaves rather than by predicting every
  training sample. The metric's parameters are also resolved once per fit instead of once per tree,
  and the labels and predictions are no longer re-validated on every evaluation. The fitted trees
  are unchanged. Other losses are still evaluated on every sample's prediction.
- |Efficiency| :class:`~empulse.models.ProfLogitClassifier` and
  :class:`~empulse.models.CSLogitClassifier` with a :class:`~empulse.metrics.MaxProfit` loss no
  longer compute a gradient for optimizers that do not use it, such as the default
  :class:`~empulse.optimizers.GeneticAlgorithmOptimizer`. Each candidate model is now scored by the
  metric itself: with 10,000 samples, :class:`~empulse.models.ProfLogitClassifier` fits about 2.9x
  faster with a deterministic metric and 1.9x faster with the expected maximum profit, to the same
  coefficients.
- |Enhancement| :class:`~empulse.models.ProfLogitClassifier` (and the logit models with any
  optimizer that does not use gradients) now supports every :class:`~empulse.metrics.MaxProfit`
  loss. Losses with a stochastic variable whose distribution is not strictly positive, such as a
  uniform or normal one, or with several stochastic variables, raised a ``NotImplementedError``,
  since only the gradient was restricted to the others.

Optimizers
----------

- |API| :class:`~empulse.optimizers.Optimizer` has a new ``requires_gradient`` property, which the
  logit models read to decide whether to build an objective that also computes its gradient. It is
  ``True`` by default, ``False`` for :class:`~empulse.optimizers.GeneticAlgorithmOptimizer`, and
  follows ``use_jacobian`` for :class:`~empulse.optimizers.ScipyOptimizer`. A custom optimizer that
  only calls ``logit_loss`` can return ``False`` to get the faster objective.

Datasets
--------

- |Feature| Added ten dataset loaders from the cost-sensitive learning literature, each shipping
  the cost matrix its source papers use:

  - :func:`~empulse.datasets.fetch_credit_card_fraud` and
    :func:`~empulse.datasets.fetch_ieee_fraud_detection` for fraud detection;
  - :func:`~empulse.datasets.fetch_kdd98` for direct marketing;
  - :func:`~empulse.datasets.fetch_telco_customer_churn`, :func:`~empulse.datasets.fetch_cell2cell`
    and :func:`~empulse.datasets.fetch_kddcup09_churn` for customer churn;
  - :func:`~empulse.datasets.load_vub_credit_scoring`, :func:`~empulse.datasets.fetch_home_equity`,
    :func:`~empulse.datasets.fetch_south_german_credit` and
    :func:`~empulse.datasets.fetch_default_credit_card_clients` for credit scoring.

  :func:`~empulse.datasets.load_vub_credit_scoring` is bundled with the package; the others are
  downloaded on first use and cached.

`0.12.0`_ (19-09-2026)
======================

Metrics
-------

- |MajorFeature| Added :class:`~empulse.metrics.MixtureMetric` and
  :class:`~empulse.metrics.MixtureComponent`, which express a weighted linear combination of
  :class:`~empulse.metrics.BaseMetric` instances. This is exact, by linearity of expectation,
  and is how :func:`~empulse.metrics.empcs_score` now expresses a distribution that mixes a
  point mass with a continuous piece — something a single :class:`~empulse.metrics.Metric`
  cannot express, since SymPy cannot symbolically integrate over such a distribution.
- |Feature| Added :class:`~empulse.metrics.BaseMetric`, the abstraction every ``loss``
  parameter throughout the package now accepts. :class:`~empulse.metrics.Metric` and
  :class:`~empulse.metrics.MixtureMetric` both implement it, so either can be passed
  interchangeably as a model's training objective.
- |Feature| Added the :class:`~empulse.metrics.LogCost` strategy for building custom expected
  log cost (weighted cross-entropy) metrics. :func:`~empulse.metrics.expected_log_cost_loss`
  is now built from it.
- |Feature| Metrics built with the :class:`~empulse.metrics.Cost` and
  :class:`~empulse.metrics.Savings` strategies now accept stochastic (``sympy.stats``) cost
  parameters, reducing them to their mean. Previously this raised
  ``NotImplementedError: Random variables are not supported for the savings metric.`` at
  :class:`~empulse.metrics.Metric` construction time.
- |Feature| Added :meth:`~empulse.metrics.CostMatrix.constrain` and
  :meth:`~empulse.metrics.MixtureMetric.constrain`, which declare the values a cost-matrix
  parameter is allowed to take. A parameter can be given inclusive ``lower``/``upper`` bounds, or a
  callable can express a condition spanning several parameters at once
  (``constrain(lambda p: p['clv'] > p['incentive_cost'], message=...)``).
- |Feature| Added :class:`~empulse.metrics.Profit`, :class:`~empulse.metrics.MinCost` and
  :class:`~empulse.metrics.EmpiricalMinCost`, the sign-flipped siblings of
  :class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.MaxProfit` and
  :class:`~empulse.metrics.EmpiricalMaxProfit`. Each computes the same quantity as its partner
  and reports it with the opposite sign, so a cost matrix can be read as costs to minimize or
  as profits to maximize. The choice is presentational: the optimal threshold and rate are
  unchanged, and models fit identically on either member of a pair.
- |Feature| Added :class:`~empulse.metrics.EmpiricalMaxProfit` and :class:`~empulse.metrics.AUEPC`
  strategies for building custom metrics that compute the empirical (convex-hull-based) maximum
  profit and the area under the empirical profit curve, respectively.
- |Enhancement| :class:`~empulse.metrics.Metric` now warns (``UserWarning``) at construction
  when the cost matrix has no terms, or its terms cancel to exactly zero — such a metric always
  evaluates to ``0.0`` regardless of its parameters.
- |Enhancement| :class:`~empulse.metrics.Metric` now warns when called with a parameter its
  cost matrix does not use, instead of silently ignoring it — catching a typo'd keyword
  argument. ``sample_weight`` and strategy-specific extras (e.g. :class:`~empulse.metrics.Savings`'
  ``baseline``) are exempt.
- |Enhancement| Multi-letter symbols now render upright in the LaTeX representations of
  :class:`~empulse.metrics.CostMatrix` and :class:`~empulse.metrics.Metric`, so a product such as
  ``clv * r`` reads as two variables rather than as one named ``clvr``. Greek names such as
  ``gamma`` are unaffected.
- |Efficiency| :class:`~empulse.metrics.MaxProfit` with ``integration_method='auto'`` now uses
  quasi-Monte Carlo for any number of stochastic variables whose distributions can be sampled,
  rather than only above two. Two stochastic variables used to go to nested quadrature.
  Pass ``random_state`` for a result reproducible across :class:`~empulse.metrics.Metric` instances;
  repeated calls on one instance were already identical.
- |Efficiency| Deterministic :class:`~empulse.metrics.MaxProfit` metrics (a profit function
  polynomial in the stochastic variable) no longer compute the ROC convex hull, and evaluate
  the profit function vectorized over all operating points instead of in a Python loop.
- |Feature| Quasi-Monte Carlo integration now covers eleven more distributions:
  :func:`~sympy.stats.BoundedPareto`, :func:`~sympy.stats.Dagum`,
  ``ExponentialPower`` (not published in SymPy's own documentation, so no link is possible),
  :func:`~sympy.stats.Frechet`,
  :func:`~sympy.stats.Gompertz`, :func:`~sympy.stats.LogLogistic`,
  :func:`~sympy.stats.RaisedCosine`, :func:`~sympy.stats.Rayleigh`,
  :func:`~sympy.stats.Reciprocal`, :func:`~sympy.stats.Weibull` and
  :func:`~sympy.stats.WignerSemicircle`. These previously fell through to plain Monte Carlo, which
  is around a thousand times less accurate for the same sampling budget.
- |Feature| :meth:`~empulse.metrics.Metric.optimal_rate` and
  :meth:`~empulse.metrics.Metric.optimal_threshold` now always compute the exact
  EMP when one stochastic variable is present.
- |Feature| :class:`~empulse.metrics.MaxProfit` accepts profit functions of any shape in the
  stochastic variable. Previously a cost matrix using :func:`sympy.exp`, :func:`sympy.log` or a
  square root raised ``PolynomialError`` from inside SymPy when the :class:`~empulse.metrics.Metric`
  was constructed.
- |API| ``check_input`` has been removed from every prebuilt metric that is now a
  :class:`~empulse.metrics.Metric`/:class:`~empulse.metrics.MixtureMetric` instance
  (:func:`~empulse.metrics.empc_score`, :func:`~empulse.metrics.mpc_score`,
  :func:`~empulse.metrics.empb_score`, :func:`~empulse.metrics.auepc_score`,
  :func:`~empulse.metrics.empa_score`, :func:`~empulse.metrics.mpa_score`,
  :func:`~empulse.metrics.empcs_score`, :func:`~empulse.metrics.mpcs_score`,
  :func:`~empulse.metrics.max_profit_score`, :func:`~empulse.metrics.expected_cost_loss`,
  :func:`~empulse.metrics.expected_log_cost_loss`, :func:`~empulse.metrics.expected_savings_score`,
  :func:`~empulse.metrics.expected_cost_loss_churn` and
  :func:`~empulse.metrics.expected_cost_loss_acquisition`; :func:`~empulse.metrics.cost_loss` and
  :func:`~empulse.metrics.savings_score` are unaffected and keep it). These prebuilt metrics now
  reject parameter values that put them outside their mathematical domain via
  :meth:`~empulse.metrics.CostMatrix.constrain`, instead of returning a meaningless number, but
  the enforced domain is not identical to the one 0.11.1's ``check_input=True`` enforced.
  :func:`~empulse.metrics.auepc_score` has also lost its ``normalize`` parameter (it defaulted
  to ``True``, so scores computed with the default are unaffected).
- |API| :class:`~empulse.metrics.MaxProfit`'s ``alpha_growth`` and ``alpha_max`` constructor
  parameters have been removed (previous defaults ``1.1`` and ``100.0``); passing either now
  raises a ``TypeError``. ``alpha`` is now a constant temperature, and the annealing schedule
  that ``alpha_growth``/``alpha_max`` used to control is now expressed with an ``alpha_schedule``
  on the optimizer (see *Optimizers* below).
- |API| The :class:`~empulse.metrics.MetricStrategy` extension API used to write custom
  strategies has changed, which will break third-party subclasses: ``logit_objective()`` now
  returns a :class:`~empulse.metrics.LogitObjective` (a new public ABC exposing ``logit_loss``,
  ``logit_gradient``, ``logit_loss_gradient`` and ``logit_gradient_steps``) instead of a bare
  tuple; ``prepare_logit_objective()`` and ``build_logit_objective()`` have been removed; and a
  new ``requires_dynamic_boost_objective`` property controls whether
  :class:`~empulse.models.CSBoostClassifier` uses the dynamic (per-iteration) or static
  boosting gradient path for the strategy.
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
- |Fix| :class:`~empulse.metrics.CostMatrix` now rejects a string term that uses a name SymPy
  reserves for its own objects, instead of quietly substituting that object. ``add_fp_cost('E')``
  used to become Euler's number, so the term disappeared from the metric's parameter list and the
  metric returned a plausible but meaningless score that no caller could influence; ``'I'`` made
  the cost complex, and ``'gamma'``/``'beta'`` raised a ``TypeError`` about ``FunctionClass`` from
  inside SymPy.
- |Fix| :class:`~empulse.metrics.Metric` now raises if the cost matrix contains two distinct
  symbols that share a name, naming both spellings. ``sympy.Symbol('clv')`` and
  ``sympy.Symbol('clv', positive=True)`` are different variables to SymPy and do not cancel, and a
  term given as a string always produces the assumption-free one. Mixing them used to surface as
  ``SyntaxError: duplicate argument 'clv' in function definition`` pointing at SymPy's generated
  source.
- |Fix| Lambdified expressions now bind their arguments in a deterministic order. The internal
  ``PicklableLambda`` used to derive that order from ``list(expression.free_symbols)`` — a set,
  whose iteration order is process-dependent under Python's hash randomization — and now sorts by
  symbol name instead. This affects any metric whose expression is lambdified without an
  explicit variable list.
- |Fix| :class:`~empulse.metrics.MaxProfit` now returns the correct expected maximum profit when
  the profit function is a polynomial of degree two or higher in the stochastic variable.
- |Fix| The LaTeX rendering of a :class:`~empulse.metrics.MaxProfit` metric (shown by
  ``metric._repr_latex_()``, e.g. in a notebook) had the wrong sign on its false-positive and
  false-negative terms: it negated the true-positive and true-negative benefits but left the
  costs untouched, so the rendered formula was neither the profit nor the cost. It now renders
  the profit being maximized. Only the displayed formula was affected; the computed metric
  value was always correct.
- |Fix| Fix :func:`~empulse.metrics.empb_score` and :func:`~empulse.metrics.auepc_score` only
  incurring the contact cost for churners who accepted the retention incentive. The contact cost
  is now incurred for every contacted churner, whether or not they accept; only the retention
  benefit net of the incentive cost remains contingent on acceptance.
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

Models
------

- |MajorFeature| Added :class:`~empulse.models.ProfMPMClassifier` and
  :class:`~empulse.models.ProfMEMPMClassifier`, profit-driven minimax probability machines
  (Maldonado, López and Vairetti, 2020). Both learn a linear decision boundary that maximizes
  the worst-case expected profit implied by each class's empirical mean and covariance, via the
  multivariate Chebyshev-Cantelli inequality; ``ProfMEMPMClassifier`` allows the two classes to
  have different worst-case accuracy bounds, ``ProfMPMClassifier`` shares one bound taken from
  the tighter of the two. Their ``loss`` must use the :class:`~empulse.metrics.MaxProfit`
  strategy; instance-dependent costs are averaged to a scalar and stochastic cost parameters are
  replaced by their mean before fitting.
- |MajorFeature| Added :class:`~empulse.models.ProfSRClassifier`, a profit-driven symbolic
  regression classifier that evolves a population of mathematical expressions with genetic
  programming to maximize a cost-sensitive metric. Requires the new optional ``gplearn``
  dependency, installable with ``pip install empulse[symbolic]``.
- |Feature| :class:`~empulse.models.CSThresholdClassifier.threshold_` and
  :class:`~empulse.models.CSRateClassifier.rate_` are now real properties returning the fitted
  decision rule (or ``None`` if it wasn't learned). They were documented as fitted attributes
  in 0.11.1 but never actually set under those names.
- |Feature| :class:`~empulse.samplers.CostSensitiveSampler` can now take a
  :class:`~empulse.metrics.Metric` instance as its ``loss``. When set, ``fit_resample`` accepts
  the cost matrix's parameters as keyword arguments (routed through scikit-learn's metadata
  routing, like the cost-sensitive models) and derives ``fp_cost``/``fn_cost`` from the metric,
  taking precedence over any ``fp_cost``/``fn_cost`` passed directly.
- |Feature| Every cost-sensitive estimator's ``loss`` now accepts any
  :class:`~empulse.metrics.BaseMetric`, so a :class:`~empulse.metrics.MixtureMetric` instance
  (e.g. :func:`~empulse.metrics.empcs_score`) can be passed directly as a model's training
  objective.
- |Fix| :class:`~empulse.models.CSLogitClassifier` and
  :class:`~empulse.models.ProfLogitClassifier` no longer return an all-zero coefficient vector,
  predicting exactly 0.5 for every sample, at the default ``C=1.0, l1_ratio=1.0``. The
  elastic-net penalty was added unnormalized to a data loss that had already been averaged over
  the samples, making it roughly ``n_samples`` times stronger than scikit-learn's at the same
  ``C``. Since the expected cost is linear in the predicted probability its gradient is bounded,
  so with modest costs ``w = 0`` genuinely satisfied the L1 optimality condition and L-BFGS-B
  stopped after zero iterations.
- |API| ``C`` now scales as ``objective_scale / (C * n_samples)``, where ``objective_scale`` is
  the mean magnitude of the per-sample cost gradient. Dividing by ``n_samples`` makes ``C`` mean
  what it means in :class:`~sklearn:sklearn.linear_model.LogisticRegression`, and the objective
  scale makes the regularization path invariant to rescaling the cost matrix, so a ``C`` grid
  tuned on costs in euros transfers to the same costs in cents. **A given** ``C`` **no longer
  selects the same model as in previous releases**; re-tune any hard-coded value or ``C`` grid.
- |API| Removed the ``soft_threshold`` parameter from
  :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.ProfLogitClassifier`.
  Use ``l1_ratio=1.0`` for sparsity, which now produces exact zeros.
- |Enhancement| A cost passed to a model as a length-1 array is now broadcast to every sample
  instead of raising a shape error, and the error message for a genuine length mismatch names
  the offending parameter and the lengths that would have been accepted.
- |API| :class:`~empulse.models.CSBaggingClassifier` and :class:`~empulse.models.CSForestClassifier`
  with ``combination='weighted_voting'`` now require the base estimator to implement
  ``predict_proba``, and raise at fit time if it doesn't. Previously a base estimator without
  ``predict_proba`` silently fell back to ``predict()`` for the out-of-bag weighting.
- |API| ``soft_threshold`` now defaults to ``True`` (previously ``False``) on
  :class:`~empulse.models.CSLogitClassifier` and :class:`~empulse.models.ProfLogitClassifier`.
  With the default ``l1_ratio=1.0`` (a pure L1 penalty) and zero-initialized coefficients, the raw
  L1 subgradient used when ``soft_threshold=False`` can make L-BFGS-B stall at exactly zero for
  every coefficient, silently producing a model that predicts a constant probability regardless
  of the input. Soft-thresholding the coefficients before evaluating the penalty avoids this.
  Models fitted with default arguments now produce different (properly fit, rather than
  degenerate) coefficients than in 0.11.1; pass ``soft_threshold=False`` explicitly to restore
  the previous default.
- |API| Because :class:`~empulse.models.ProfTreeClassifier` no longer draws from the C library's
  process-global ``rand()``/``srand()`` (see below), it produces a different — equally valid —
  tree for a given ``random_state`` than it did in earlier releases. Results within this release
  are reproducible as before.
- |API| When :class:`~empulse.models.CSBoostClassifier` is used with the CatBoost backend and a
  ``loss`` metric that is maximized (e.g. :func:`~empulse.metrics.empc_score`), the value CatBoost
  reports for the evaluation metric is now negated, so that lower is better for every metric. This
  affects CatBoost's training output and ``best_score_`` only; the model that is selected, and
  early stopping, are unchanged.
- |Fix| :class:`~empulse.models.ProfTreeClassifier` no longer draws from the C library's
  process-global ``rand()``/``srand()``. Every fit now owns its generator state, so one fit can no
  longer reseed another, and ``random_state=None`` no longer derives its seed from a one-second
  resolution clock (two fits starting in the same second used to share a seed).
- |Fix| ``fit`` now deep-copies the ``loss`` metric before use, so two estimators sharing one
  module-level prebuilt metric (e.g. two models both given ``empc_score`` as ``loss``), or one
  estimator fit twice from the same ``loss`` instance, no longer read or overwrite each other's
  memoised strategy state (boosting objective, Monte Carlo grid, RNG).
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
- |Fix| :class:`~empulse.models.CSForestClassifier` now offsets negative costs the same way
  :class:`~empulse.models.CSTreeClassifier` already did, so a forest trained with a benefit
  (e.g. ``tp_cost=-200``) now trains correctly instead of raising during node-impurity
  computation (which requires ``node_impurity >= 0``).
- |Fix| :class:`~empulse.models.CSBoostClassifier`'s ``predict_proba`` raised
  ``TypeError: isinstance() arg 2 must be a type`` in any environment without LightGBM installed,
  regardless of which backend was actually in use for the fitted model.
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
- |Fix| :class:`~empulse.models.CSTreeClassifier`'s ``predict``/``predict_proba`` now validate
  their input, so a wrong number of features raises the standard scikit-learn error instead of
  being forwarded raw to the underlying tree.
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
- |Fix| :class:`~empulse.models.RobustCSClassifier` now raises a ``ValueError`` naming both keys
  instead of silently dropping one of them, when its ``loss`` metric's cost matrix gives two
  aliases to the same underlying symbol.
- |Fix| :class:`~empulse.models.RobustCSClassifier`'s ``classes_`` property now raises the
  standard "not fitted" error via ``check_is_fitted`` instead of an unrelated ``AttributeError``.
- |Fix| Metadata routing (``set_fit_request``/``set_predict_request``/``set_fit_resample_request``)
  for a ``loss``'s cost-matrix parameter names is now resolved per instance instead of being
  installed on the class. Every cost-sensitive estimator and :class:`~empulse.samplers.CostSensitiveSampler`
  used to rewrite a class-level descriptor from ``__init__``, so constructing a second instance of
  the same class with a different ``loss`` silently changed which parameter names *every earlier
  instance* of that class accepted -- e.g. ``a = CSBoostClassifier(loss=empc_score)`` followed by
  ``b = CSBoostClassifier(loss=mpcs_score)`` made ``a.set_fit_request(clv=True)`` raise
  ``TypeError``, because ``b``'s construction had overwritten the class-level accepted keys.
  Accepted keys are now computed from each instance's own ``loss``, so unrelated estimators (and a
  loss swapped in later via ``set_params``) no longer interfere with each other.
- |Fix| :class:`~empulse.samplers.CostSensitiveSampler` now takes a per-fit deep copy of its
  ``loss`` metric before ``fit_resample``, matching every cost-sensitive model, so a shared
  module-level prebuilt metric (e.g. :func:`~empulse.metrics.empc_score`) no longer leaks
  memoized strategy state between two samplers, or two calls, that use it.

Optimizers
----------

- |Fix| :class:`~empulse.optimizers.LBFGSBOptimizer` now minimizes a non-smooth elastic-net
  objective (``l1_ratio > 0``) exactly, through a split-variable reformulation (``w = u - v``
  with ``u, v >= 0``) that turns the L1 term into a linear function L-BFGS-B handles natively
  through its box constraints. Previously it was handed ``sign(w)`` as a subgradient, which made
  the origin look stationary; it now converges across the whole ``l1_ratio`` range, produces
  exact zeros, and matches an independent proximal-gradient solver.
  :class:`~empulse.optimizers.ScipyOptimizer` is deliberately unchanged, so user-supplied
  ``bounds`` keep working there; it performs plain subgradient descent and will not produce
  exact zeros.
- |Enhancement| :class:`~empulse.optimizers.LBFGSBOptimizer`'s ``tolerance`` is now relative to
  the objective magnitude rather than absolute, so a cost matrix expressed in euros and the same
  one expressed in cents converge to the same model.

- |MajorFeature| ``empulse.optimizers`` has been substantially expanded, from exporting only
  :class:`~empulse.optimizers.Generation` to a full set of optimizers and schedules: the
  :class:`~empulse.optimizers.Optimizer` base class, first-order minibatch optimizers
  :class:`~empulse.optimizers.SGD`, :class:`~empulse.optimizers.Adam` and
  :class:`~empulse.optimizers.RMSProp`, :class:`~empulse.optimizers.ScipyOptimizer` (wraps any
  :func:`scipy.optimize.minimize` method) and :class:`~empulse.optimizers.LBFGSBOptimizer`,
  evolutionary optimizers :class:`~empulse.optimizers.GeneticAlgorithmOptimizer`,
  :class:`~empulse.optimizers.MemeticOptimizer` (genetic algorithm with Lamarckian gradient local
  search) and :class:`~empulse.optimizers.LamarckianGeneration`, and the learning-rate/temperature
  schedule family :class:`~empulse.optimizers.Schedule`, :class:`~empulse.optimizers.ConstantSchedule`,
  :class:`~empulse.optimizers.LinearSchedule`, :class:`~empulse.optimizers.ExponentialSchedule`,
  :class:`~empulse.optimizers.StepSchedule`, :class:`~empulse.optimizers.CosineAnnealingSchedule` and
  :class:`~empulse.optimizers.WarmupSchedule` (:class:`~empulse.optimizers.ExponentialSchedule` and
  :class:`~empulse.optimizers.StepSchedule` also gained a ``max_value`` upper-bound parameter).
- |API| :class:`~empulse.models.ProfLogitClassifier` and :class:`~empulse.models.CSLogitClassifier`
  no longer take ``optimize_fn``/``optimizer_params`` constructor parameters. Both now take a
  single ``optimizer`` parameter, an :class:`~empulse.optimizers.Optimizer` instance
  (defaulting to :class:`~empulse.optimizers.GeneticAlgorithmOptimizer` for
  ``ProfLogitClassifier`` and :class:`~empulse.optimizers.LBFGSBOptimizer` for
  ``CSLogitClassifier``, matching each model's previous default behavior). For example,
  ``ProfLogitClassifier(optimizer_params={'max_iter': 10})`` becomes
  ``ProfLogitClassifier(optimizer=GeneticAlgorithmOptimizer(max_iter=10))``.
  :class:`~empulse.models.ProfLogitClassifier`'s ``n_jobs`` constructor parameter has moved
  onto :class:`~empulse.optimizers.GeneticAlgorithmOptimizer` for the same reason.

Datasets
--------

- |MajorFeature| Dataset loaders are now backend-agnostic via `narwhals
  <https://narwhals-dev.github.io/narwhals/>`_: every loader takes a required, keyword-only
  ``backend`` argument — the dataframe *module* itself, e.g. ``backend=pandas`` or
  ``backend=polars`` — and returns data in that library's native types. A bare
  ``pip install empulse`` no longer requires any particular dataframe library to be installed;
  you choose and supply one yourself.
- |Feature| Added :func:`~empulse.datasets.fetch_iranian_churn`, a churn dataset from the UCI
  Machine Learning Repository (3,150 customers, 495 churners, 12 features), downloaded on first
  use and cached locally. It ships a churn-retention cost matrix driven by each customer's
  value, with overridable economic parameters ``incentive_fraction``, ``contact_cost`` and
  ``accept_rate``.
- |Feature| Added :func:`~empulse.datasets.get_data_home`, which resolves and creates the
  directory downloaded datasets are cached in: an explicit ``data_home`` argument, else the
  ``EMPULSE_DATA_HOME`` environment variable, else ``~/empulse_data``.
- |API| :func:`~empulse.datasets.load_give_me_some_credit` has been renamed to
  :func:`~empulse.datasets.fetch_give_me_some_credit`: rather than being bundled with the
  package, it is now downloaded from OpenML on first use and cached (subsequent calls read the
  cache and do not hit the network). It gains ``data_home`` and ``download_if_missing``
  parameters, matching :func:`~empulse.datasets.fetch_iranian_churn`.
- |API| The ``as_frame`` and ``return_X_y_costs`` parameters have been removed from every
  dataset loader — loaders always return a :class:`~empulse.datasets.Dataset`, and there is no
  longer a plain-numpy return path (call ``.to_numpy()`` on the returned frame if you need one).
  The per-loader economic keyword arguments (e.g. ``interest_rate``, ``fund_cost``,
  ``max_credit_line``, ``loss_given_default``, ``term_length_months``, ``loan_to_income_ratio``,
  ``term_deposit_fraction``, ``contact_cost``) have also been removed from the loaders'
  signatures. :class:`~empulse.datasets.Dataset` no longer carries fixed
  ``tp_cost``/``tn_cost``/``fp_cost``/``fn_cost`` values; instead it carries a symbolic
  ``cost_matrix`` (a :class:`~empulse.metrics.CostMatrix`) and an ``instance_costs`` dict of
  per-sample values. The dataset's economic parameters that used to be set at load time are now
  passed at metric-call time instead, e.g.::

      # before
      dataset = load_credit_scoring_pakdd(as_frame=True, loss_given_default=0.6)
      metric(dataset.target, y_score, ...)

      # now
      import pandas as pd
      dataset = load_credit_scoring_pakdd(backend=pd)
      metric = Metric(dataset.cost_matrix, Cost())
      metric(dataset.target, y_score, loss_given_default=0.6, **dataset.instance_costs)

Packaging and dependencies
---------------------------

- |MajorFeature| Empulse now supports the free-threaded build of CPython 3.14. ``cp314t`` wheels
  are published alongside the regular ones, and all eleven Cython extension modules declare the
  ``freethreading_compatible`` directive, so importing Empulse no longer re-enables the global
  interpreter lock for the whole process. Fitting or scoring separate estimator and
  :class:`~empulse.metrics.Metric` instances on separate threads is supported and gives the same
  results as running them sequentially. Note that ``pip install empulse[boosting]`` does not
  resolve on 3.14t until CatBoost publishes a free-threaded wheel; XGBoost and LightGBM install
  fine.
- |API| ``pandas`` is no longer a required dependency; ``narwhals`` is used instead. A bare
  ``pip install empulse`` no longer installs a dataframe library — none of the optional extras
  supply one either, so using a dataset loader requires installing pandas, polars or another
  narwhals-supported library yourself and passing it as ``backend=``.
- |API| The minimum supported scikit-learn version is now 1.9.0 (previously 1.5.2), and the
  minimum supported imbalanced-learn version is now 0.14.2 (previously 0.13.0).
- |API| The optional-dependency extras have been reorganized: ``pip install empulse[boosting]``
  now installs XGBoost, LightGBM and CatBoost (what ``empulse[optional]`` used to install), a new
  ``pip install empulse[symbolic]`` installs ``gplearn`` (required by
  :class:`~empulse.models.ProfSRClassifier`), and ``pip install empulse[optional]`` now installs
  every optional extra.

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


.. _Unreleased: https://github.com/ShimantoRahman/empulse/compare/0.12.0...main
.. _0.12.0: https://github.com/ShimantoRahman/empulse/releases/tag/0.12.0
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