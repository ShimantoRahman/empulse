.. _csthreshold:
.. _csrate:
.. _threshold_tuning:

===========================
Threshold Tuning
===========================

After training a probabilistic classifier you typically predict the *positive* class for every
sample whose score exceeds 0.5.  That default threshold is almost never optimal when
misclassification costs are asymmetric.  Empulse provides two dedicated meta-estimators
for analytic threshold / rate selection, and the :class:`~empulse.metrics.Metric` class
integrates seamlessly with scikit-learn's :class:`~sklearn:sklearn.model_selection.TunedThresholdClassifierCV`
for cross-validated threshold search.

.. list-table:: Choosing an approach
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Analytic (empulse)
     - Cross-validated search (sklearn)
   * - Class
     - :class:`~empulse.models.CSThresholdClassifier` / :class:`~empulse.models.CSRateClassifier`
     - :class:`~sklearn:sklearn.model_selection.TunedThresholdClassifierCV`
   * - How it works
     - Derives the decision boundary analytically from the cost matrix at fit time
     - Scans candidate thresholds via cross-validation and picks the best one
   * - Computation speed
     - Fast — a single closed-form computation
     - Slower — depends on ``cv`` splits × candidate thresholds


.. _csthreshold_classifier:

CSThresholdClassifier
=====================

:class:`~empulse.models.CSThresholdClassifier` wraps any probabilistic base classifier.
During ``fit`` it calibrates the probabilities (optional but recommended, sigmoid by default),
then computes the cost-optimal decision threshold analytically.  During ``predict`` it applies
that stored threshold — or recomputes it on-the-fly when you pass fresh cost information.

Quick Start
-----------

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.models import CSThresholdClassifier

    X, y = make_classification(n_samples=1000, random_state=0)

    model = CSThresholdClassifier(
        estimator=LogisticRegression(),
        fp_cost=5,   # cost of a false positive (e.g. wasted marketing spend)
        fn_cost=1,   # cost of a false negative (e.g. missed churner)
    )
    model.fit(X, y)

    print(f"Optimal threshold: {model.threshold_:.4f}")
    y_pred = model.predict(X)


Cost Matrix
-----------

The classifier accepts the same four cost terms as all cost-sensitive Empulse models.

Constant costs
~~~~~~~~~~~~~~

Pass a scalar to apply the same cost to every sample:

.. code-block:: python

    from empulse.models import CSThresholdClassifier
    from sklearn.linear_model import LogisticRegression

    # Low recall penalty, high precision penalty
    model = CSThresholdClassifier(
        LogisticRegression(),
        tp_cost=10,   # benefit of catching a churner
        fp_cost=2,    # cost of contacting a non-churner
        fn_cost=0,
        tn_cost=0,
    )

Instance-dependent costs
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass per-sample cost arrays to ``fit`` when each observation has its own cost profile
(e.g., individual Customer Lifetime Values):

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.models import CSThresholdClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 1000, size=len(y))

    model = CSThresholdClassifier(
        LogisticRegression(),
    ).set_fit_request(tp_cost=True)

    model.fit(X, y, tp_cost=clv)
    # For instance-dependent costs, multiple thresholds are learned
    print(model.threshold_)  # array of shape (n_samples,)

.. note::
    Instance-dependent costs require
    :ref:`metadata routing <sklearn:metadata_routing>` to be enabled via
    ``sklearn.set_config(enable_metadata_routing=True)``.

Custom Metric
~~~~~~~~~~~~~

For non-standard business objectives, pass a :class:`~empulse.metrics.Metric` instance
as the ``loss`` parameter.  The model will learn the decision threshold that maximises /
minimises that metric:

.. code-block:: python

    import sympy
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.metrics import Metric, MaxProfit, CostMatrix
    from empulse.models import CSThresholdClassifier

    clv, incentive_cost, contact_cost, accept_rate = sympy.symbols(
        'clv incentive_cost contact_cost accept_rate'
    )

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(accept_rate * (clv - incentive_cost - contact_cost))
        .add_tp_benefit((1 - accept_rate) * -contact_cost)
        .add_fp_cost(incentive_cost + contact_cost)
    )
    profit_metric = Metric(cost_matrix, MaxProfit())

    X, y = make_classification(n_samples=1000, random_state=0)

    model = CSThresholdClassifier(
        LogisticRegression(),
        loss=profit_metric,
    )
    model.fit(X, y, clv=200, incentive_cost=10, contact_cost=1, accept_rate=0.3)
    print(f"Optimal threshold: {model.threshold_:.4f}")

Probability Calibration
-----------------------

Analytic thresholds are only meaningful when the model outputs well-calibrated
probabilities.  ``CSThresholdClassifier`` ships with an optional internal calibration
step controlled by the ``calibrator`` parameter:

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from empulse.models import CSThresholdClassifier

    # sigmoid calibration (default) — fast, suitable for Platt scaling
    model_sigmoid = CSThresholdClassifier(
        GradientBoostingClassifier(),
        calibrator='sigmoid',
        fp_cost=5,
        fn_cost=1,
    )

    # isotonic calibration — more flexible, needs larger datasets
    model_isotonic = CSThresholdClassifier(
        GradientBoostingClassifier(),
        calibrator='isotonic',
        fp_cost=5,
        fn_cost=1,
    )

    # No calibration — use only when probabilities are already well-calibrated
    model_none = CSThresholdClassifier(
        LogisticRegression(),
        calibrator=None,
        fp_cost=5,
        fn_cost=1,
    )

Override the Threshold at Predict Time
----------------------------------------

You can supply different costs at inference time without re-fitting the model.  This is
useful when costs vary by deployment context (e.g. different campaigns):

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.models import CSThresholdClassifier

    X, y = make_classification(n_samples=500, random_state=0)

    model = CSThresholdClassifier(LogisticRegression(), fp_cost=5, fn_cost=1).fit(X, y)

    # Use the threshold learned at fit time
    y_pred_default = model.predict(X)

    # Override: higher false-positive penalty at inference (more conservative)
    y_pred_conservative = model.predict(X, fp_cost=20, fn_cost=1)

    # Count how many fewer positives the conservative threshold produces
    print(f"Standard positives  : {y_pred_default.sum()}")
    print(f"Conservative positives: {y_pred_conservative.sum()}")

.. note::
    Overriding at predict time recomputes the threshold analytically from the new
    costs — no re-fitting occurs.  This does **not** work with ``MaxProfit``-based
    metrics because that strategy requires label information from the training set.

sklearn Integration
-------------------

``CSThresholdClassifier`` is a fully sklearn-compatible meta-estimator: it implements
``predict_proba``, ``predict_log_proba``, and ``decision_function`` by delegating to the
wrapped estimator, and it works inside :class:`~sklearn:sklearn.pipeline.Pipeline` and
:class:`~sklearn:sklearn.model_selection.GridSearchCV`.

Pipeline with cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.models import CSThresholdClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    tp_cost = np.random.default_rng(0).uniform(100, 500, size=len(y))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            CSThresholdClassifier(LogisticRegression()).set_fit_request(tp_cost=True),
        ),
    ])

    scores = cross_val_score(pipeline, X, y, params={'tp_cost': tp_cost})
    print(scores.mean())

Hyperparameter search
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from empulse.metrics import expected_cost_loss
    from empulse.models import CSThresholdClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    fp_cost = np.random.default_rng(0).uniform(1, 10, size=len(y))

    scorer = make_scorer(
        expected_cost_loss,
        response_method='predict_proba',
        greater_is_better=False,
        normalize=True,
        fn_cost=1.0,
    ).set_score_request(fp_cost=True)

    grid = GridSearchCV(
        CSThresholdClassifier(LogisticRegression()).set_fit_request(fp_cost=True),
        param_grid={'estimator__C': np.logspace(-3, 2, 6)},
        scoring=scorer,
    )
    grid.fit(X, y, fp_cost=fp_cost)
    print(f"Best C: {grid.best_params_['estimator__C']:.4f}")


.. _csrate_classifier:

CSRateClassifier
================

:class:`~empulse.models.CSRateClassifier` classifies the *top-k* most-likely-positive
samples as positive, where *k* is chosen to maximise the cost-sensitive metric.  Instead
of a raw score threshold it learns a **positive rate** — the fraction of all samples that
should be labelled positive.

When to use ``CSRateClassifier`` over ``CSThresholdClassifier``
----------------------------------------------------------------

* Your deployment has a fixed capacity constraint (e.g. *"we can only call 10 % of customers"*).  ``CSRateClassifier`` naturally implements a top-k selection rule.
* Your probabilities are ordinal but not well-calibrated and you cannot or do not want to calibrate them.
* You want a decision rule that is invariant to monotone transformations of the predicted scores.

Quick Start
-----------

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.models import CSRateClassifier

    X, y = make_classification(n_samples=1000, random_state=0)

    model = CSRateClassifier(
        estimator=LogisticRegression(),
        tp_cost=300,   # benefit of catching a churner
        fp_cost=10,    # cost of contacting a non-churner
    )
    model.fit(X, y)
    print(f"Optimal positive rate: {model.rate_:.4f}")
    y_pred = model.predict(X)

Custom Metric
-------------

Like ``CSThresholdClassifier``, the rate classifier accepts a custom
:class:`~empulse.metrics.Metric`:

.. code-block:: python

    import sympy
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.metrics import Metric, MaxProfit, CostMatrix
    from empulse.models import CSRateClassifier

    clv, incentive_cost, contact_cost, accept_rate = sympy.symbols(
        'clv incentive_cost contact_cost accept_rate'
    )

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(accept_rate * (clv - incentive_cost - contact_cost))
        .add_tp_benefit((1 - accept_rate) * -contact_cost)
        .add_fp_cost(incentive_cost + contact_cost)
    )
    profit_metric = Metric(cost_matrix, MaxProfit())

    X, y = make_classification(n_samples=1000, random_state=0)

    model = CSRateClassifier(LogisticRegression(), loss=profit_metric)
    model.fit(X, y, clv=200, incentive_cost=10, contact_cost=1, accept_rate=0.3)
    print(f"Optimal positive rate: {model.rate_:.4f}")

Override the Rate at Predict Time
----------------------------------

Exactly like ``CSThresholdClassifier``, you can supply fresh cost parameters at
inference time:

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from empulse.models import CSRateClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    model = CSRateClassifier(LogisticRegression(), tp_cost=300, fp_cost=10).fit(X, y)

    # Double the benefit → expect a higher positive rate
    y_pred_generous = model.predict(X, tp_cost=600, fp_cost=10)
    print(f"Default   positives: {model.predict(X).sum()}")
    print(f"Generous  positives: {y_pred_generous.sum()}")


.. _tuned_threshold_cv:

TunedThresholdClassifierCV with empulse Metrics
================================================

Scikit-learn 1.5+ ships with
:class:`~sklearn:sklearn.model_selection.TunedThresholdClassifierCV`, which scans a grid
of threshold candidates via cross-validation and picks the one that maximises a scorer.
Any callable Empulse metric — whether a standalone score function or a
:class:`~empulse.metrics.Metric` instance — can be wrapped into a scorer with
:func:`~sklearn:sklearn.metrics.make_scorer` and plugged straight in.

.. note::
    Use :class:`~sklearn:sklearn.model_selection.TunedThresholdClassifierCV` when:

    * your classifier's probability estimates are not well-calibrated *and* you cannot
      add calibration, or
    * the cost structure changes frequently between evaluations and you want the
      threshold tuned holistically via cross-validation rather than analytically.

Using a built-in empulse score function
----------------------------------------

All standalone empulse score functions (``mpc_score``, ``empc_score``, ``expected_cost_loss``, etc.)
follow the ``(y_true, y_score, **kwargs) → float`` signature accepted by
:func:`~sklearn:sklearn.metrics.make_scorer`:

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TunedThresholdClassifierCV
    from empulse.metrics import mpc_score

    X, y = make_classification(n_samples=1000, random_state=0)

    scorer = make_scorer(
        mpc_score,
        response_method='predict_proba',
        greater_is_better=True,   # MPC is a profit metric — higher is better
        clv=200,
        incentive_cost=10,
        contact_cost=1,
        accept_rate=0.3,
    )

    model = TunedThresholdClassifierCV(
        estimator=GradientBoostingClassifier(),
        scoring=scorer,
        cv=5,
    )
    model.fit(X, y)
    print(f"Tuned threshold: {model.best_threshold_:.4f}")
    y_pred = model.predict(X)

Using a custom ``Metric`` instance
------------------------------------

A :class:`~empulse.metrics.Metric` object is itself callable with signature
``(y_true, y_score, **parameters) → float``, so it can be passed directly to
:func:`~sklearn:sklearn.metrics.make_scorer`.  Set ``greater_is_better`` to ``True``
when you use :class:`~empulse.metrics.MaxProfit` or :class:`~empulse.metrics.Savings`
(higher is better) and to ``False`` for :class:`~empulse.metrics.Cost` (lower is better):

.. code-block:: python

    import sympy
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TunedThresholdClassifierCV
    from empulse.metrics import Metric, MaxProfit, CostMatrix

    # --- Define the custom profit metric ---
    clv, incentive_cost, contact_cost, accept_rate = sympy.symbols(
        'clv incentive_cost contact_cost accept_rate'
    )

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(accept_rate * (clv - incentive_cost - contact_cost))
        .add_tp_benefit((1 - accept_rate) * -contact_cost)
        .add_fp_cost(incentive_cost + contact_cost)
    )
    profit_metric = Metric(cost_matrix, MaxProfit())

    # --- Create scorer from the Metric ---
    scorer = make_scorer(
        profit_metric,
        response_method='predict_proba',
        greater_is_better=True,   # MaxProfit — higher is better
        clv=200,
        incentive_cost=10,
        contact_cost=1,
        accept_rate=0.3,
    )

    # --- Tune the threshold ---
    X, y = make_classification(n_samples=1000, random_state=0)

    model = TunedThresholdClassifierCV(
        estimator=GradientBoostingClassifier(),
        scoring=scorer,
        cv=5,
    )
    model.fit(X, y)
    print(f"Tuned threshold: {model.best_threshold_:.4f}")
    y_pred = model.predict(X)

Using a cost-minimisation Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the strategy is :class:`~empulse.metrics.Cost`, the metric returns a *loss*
(lower is better).  Set ``greater_is_better=False`` accordingly:

.. code-block:: python

    import sympy
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TunedThresholdClassifierCV
    from empulse.metrics import Metric, Cost, CostMatrix

    fp, fn = sympy.symbols('fp fn')

    cost_matrix = (
        CostMatrix()
        .add_fp_cost(fp)
        .add_fn_cost(fn)
    )
    cost_metric = Metric(cost_matrix, Cost())

    scorer = make_scorer(
        cost_metric,
        response_method='predict_proba',
        greater_is_better=False,   # Cost — lower is better
        fp=5.0,
        fn=1.0,
    )

    X, y = make_classification(n_samples=1000, random_state=0)

    model = TunedThresholdClassifierCV(LogisticRegression(), scoring=scorer, cv=5)
    model.fit(X, y)
    print(f"Tuned threshold: {model.best_threshold_:.4f}")

Instance-dependent costs with metadata routing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instance-dependent costs (per-sample arrays) can be passed to the scorer via
:ref:`metadata routing <sklearn:metadata_routing>`.  Enable routing globally, request
the cost array on the scorer, then pass it to ``fit``:

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TunedThresholdClassifierCV
    from empulse.metrics import expected_cost_loss

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=1000, random_state=0)
    # Per-sample false-positive cost (e.g. individual campaign spend)
    fp_cost = np.random.default_rng(0).uniform(1, 10, size=len(y))

    scorer = (
        make_scorer(
            expected_cost_loss,
            response_method='predict_proba',
            greater_is_better=False,
            normalize=True,
            fn_cost=1.0,
        )
        .set_score_request(fp_cost=True)   # tell the scorer to expect fp_cost
    )

    model = TunedThresholdClassifierCV(
        estimator=GradientBoostingClassifier(),
        scoring=scorer,
        cv=5,
    )
    model.fit(X, y, fp_cost=fp_cost)   # pass the array directly to fit
    print(f"Tuned threshold: {model.best_threshold_:.4f}")

Combining threshold tuning with hyperparameter search
-------------------------------------------------------

:class:`~sklearn:sklearn.model_selection.TunedThresholdClassifierCV` can be nested inside
:class:`~sklearn:sklearn.model_selection.GridSearchCV` to jointly optimise both the base
estimator's hyperparameters and the decision threshold:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
    from empulse.metrics import mpc_score

    X, y = make_classification(n_samples=1000, random_state=0)

    scorer = make_scorer(
        mpc_score,
        response_method='predict_proba',
        greater_is_better=True,
        clv=200,
        incentive_cost=10,
        contact_cost=1,
        accept_rate=0.3,
    )

    tuned_model = TunedThresholdClassifierCV(
        estimator=GradientBoostingClassifier(),
        scoring=scorer,
        cv=3,
    )

    grid_search = GridSearchCV(
        tuned_model,
        param_grid={
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
        },
        scoring=scorer,
        cv=5,
    )
    grid_search.fit(X, y)
    best = grid_search.best_estimator_
    print(f"Best params  : {grid_search.best_params_}")
    print(f"Best threshold: {best.best_threshold_:.4f}")

.. seealso::

    * :ref:`user_defined_value_metric` — how to build a custom :class:`~empulse.metrics.Metric`
    * :ref:`metric_class_in_model` — which Empulse models accept a :class:`~empulse.metrics.Metric` as ``loss``
    * :ref:`cslogit` — linear models that bake the cost-sensitive objective directly into training
    * :class:`~sklearn:sklearn.model_selection.TunedThresholdClassifierCV` — sklearn reference

