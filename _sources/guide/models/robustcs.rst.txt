.. _robustcs:

===============================================
Robust Cost-Sensitive Classification (RobustCS)
===============================================

Instance-dependent cost-sensitive learning relies on accurate per-sample cost estimates.
In practice those estimates often contain noise and outliers — a single erroneous CLV or
exaggerated contact cost can bias the whole cost surface.
:class:`~empulse.models.RobustCSClassifier` wraps **any** cost-sensitive estimator in a
three-step framework [1]_ that detects and corrects outlier costs before training:

1. **Feature-based regression** — A :class:`~sklearn:sklearn.linear_model.HuberRegressor`
   is fitted on the feature matrix ``X`` using the instance-dependent cost as the target.
   A robust regressor is chosen so that the regression itself is not distorted by extreme
   cost values.

2. **Outlier flagging** — Samples whose standardised residual (original cost vs. predicted
   cost) exceeds ``outlier_threshold`` (default 2.5) are marked as outliers.

3. **Cost imputation** — Outlier costs are replaced with the regressor's predicted value,
   which is a smoothed, feature-consistent estimate.

The corrected costs are then passed to the wrapped estimator exactly as if they had been
provided directly, so the rest of the training pipeline is unchanged.

.. note::
    Only arrays with non-zero standard deviation are treated as instance-dependent costs.
    Scalar (class-level) costs are never modified.


Quick Start
===========

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    X, y = make_classification(n_samples=500, random_state=0)

    # Simulate noisy CLV with a few extreme outliers
    rng = np.random.default_rng(0)
    fp_cost = rng.uniform(1, 50, size=len(y))
    fp_cost[[10, 42, 99]] = 10_000   # inject outliers

    model = RobustCSClassifier(CSLogitClassifier())
    model.fit(X, y, fp_cost=fp_cost)
    y_pred = model.predict(X)


Standard API
============

When the wrapped estimator uses the four standard cost terms
(``tp_cost``, ``tn_cost``, ``fn_cost``, ``fp_cost``), outlier detection is controlled by
the ``detect_outliers_for`` parameter.

Correct all instance-dependent costs (default)
----------------------------------------------

By default, every cost term that is passed as an array (i.e. is instance-dependent) is
checked for outliers:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    rng = np.random.default_rng(0)
    fp_cost = rng.uniform(1, 50, size=len(y))
    fn_cost = rng.uniform(1, 10, size=len(y))

    # Both fp_cost and fn_cost will be checked — fp_cost=5 would be skipped (scalar)
    model = RobustCSClassifier(CSLogitClassifier())
    model.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

Correct only selected cost terms
---------------------------------

Pass a cost name or a list of names to limit which costs are checked:

.. code-block:: python

    from empulse.models import CSLogitClassifier, RobustCSClassifier

    # Only false-positive costs are outlier-corrected
    model_fp = RobustCSClassifier(
        CSLogitClassifier(),
        detect_outliers_for='fp_cost',
    )

    # Both false-positive and false-negative costs are outlier-corrected
    model_fp_fn = RobustCSClassifier(
        CSLogitClassifier(),
        detect_outliers_for=['fp_cost', 'fn_cost'],
    )

.. note::
    The standard ``detect_outliers_for`` parameter only supports the four named cost
    terms.  When you need outlier correction for **arbitrary symbolic parameters** of a
    custom profit metric, use the :ref:`custom Metric API <robustcs_metric>` instead.


.. _robustcs_metric:

Custom Metric with ``mark_outlier_sensitive``
=============================================

When the wrapped estimator uses a custom :class:`~empulse.metrics.Metric` as its ``loss``
parameter, ``detect_outliers_for`` is **ignored**.  Instead, the symbols that should be
outlier-corrected are declared directly on the :class:`~empulse.metrics.CostMatrix`
using :meth:`~empulse.metrics.CostMatrix.mark_outlier_sensitive`.

This lets you select *any* symbolic parameter — not just the four standard cost terms —
for robust correction.  It is particularly useful when your cost model mixes instance-
dependent inputs like CLV or incentive cost with class-level parameters like accept rate.

Basic example — marking a single symbol
-----------------------------------------

.. code-block:: python

    import numpy as np
    import sympy as sp
    from sklearn.datasets import make_classification
    from empulse.metrics import Metric, Cost, CostMatrix
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    rng = np.random.default_rng(0)

    a, b = sp.symbols('a b')

    cost_matrix = (
        CostMatrix()
        .add_fp_cost(a)                  # a is instance-dependent
        .add_fn_cost(b)                  # b is class-level (scalar)
        .mark_outlier_sensitive(a)       # only 'a' will be outlier-corrected
    )
    cost_loss = Metric(cost_matrix, Cost())

    # 'b' is a scalar — only 'a' is corrected
    model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
    model.fit(X, y, a=rng.uniform(1, 50, size=len(y)), b=5.0)

Marking multiple symbols
------------------------

You can chain multiple ``mark_outlier_sensitive`` calls to correct several parameters
independently:

.. code-block:: python

    import numpy as np
    import sympy as sp
    from sklearn.datasets import make_classification
    from empulse.metrics import Metric, Cost, CostMatrix
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    rng = np.random.default_rng(0)
    clv = rng.uniform(100, 500, size=len(y))
    d   = rng.uniform(5, 50,  size=len(y))

    clv_sym, d_sym, f_sym, gamma_sym = sp.symbols('clv d f gamma')

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma_sym * (clv_sym - d_sym - f_sym))
        .add_tp_benefit((1 - gamma_sym) * -f_sym)
        .add_fp_cost(d_sym + f_sym)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .mark_outlier_sensitive(clv_sym)    # CLV is instance-dependent → correct
        .mark_outlier_sensitive(d_sym)      # incentive cost is instance-dependent → correct
    )
    cost_loss = Metric(cost_matrix, Cost())

    model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
    model.fit(
        X, y,
        clv=clv,
        incentive_cost=d,        # alias 'd' → d_sym
        contact_cost=1.0,        # scalar — not corrected
        accept_rate=0.3,         # scalar — not corrected
    )

Using string names instead of symbols
--------------------------------------

``mark_outlier_sensitive`` accepts both a :class:`sympy.Symbol` and a plain string
(the symbol name or its alias):

.. code-block:: python

    cost_matrix = (
        CostMatrix()
        .add_fp_cost(sp.Symbol('clv'))
        .add_fn_cost(sp.Symbol('f'))
        .mark_outlier_sensitive('clv')   # equivalent to mark_outlier_sensitive(sp.Symbol('clv'))
    )


Configuring Outlier Detection
==============================

Custom outlier regressor
------------------------

Replace the default :class:`~sklearn:sklearn.linear_model.HuberRegressor` with any
sklearn-compatible regressor that supports ``fit`` and ``predict``:

.. code-block:: python

    from sklearn.linear_model import HuberRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    # Tighter Huber regression (fewer iterations)
    model_huber = RobustCSClassifier(
        CSLogitClassifier(),
        outlier_estimator=HuberRegressor(epsilon=2.0, max_iter=200),
    )

    # Gradient boosting for non-linear cost surfaces
    model_gbr = RobustCSClassifier(
        CSLogitClassifier(),
        outlier_estimator=GradientBoostingRegressor(n_estimators=50),
    )

Outlier threshold
-----------------

``outlier_threshold`` controls how many standard deviations a residual must exceed to be
flagged as an outlier (default: 2.5).  A lower value is more aggressive; a higher value
is more conservative:

.. code-block:: python

    from empulse.models import CSLogitClassifier, RobustCSClassifier

    # More aggressive: flag anything beyond 2 std deviations
    model_aggressive = RobustCSClassifier(CSLogitClassifier(), outlier_threshold=2.0)

    # More conservative: only flag extreme outliers
    model_conservative = RobustCSClassifier(CSLogitClassifier(), outlier_threshold=3.5)


Inspecting Fitted Attributes
=============================

After fitting, two attributes expose the internal state of the outlier correction:

``costs_``
----------

A dictionary of the final (possibly imputed) cost arrays passed to the wrapped estimator:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    X, y = make_classification(n_samples=200, random_state=0)
    fp_cost = np.random.default_rng(0).uniform(1, 50, size=len(y))

    model = RobustCSClassifier(CSLogitClassifier()).fit(X, y, fp_cost=fp_cost)

    print(model.costs_['fp_cost'])  # corrected fp_cost array

``outlier_estimators_``
-----------------------

A dictionary mapping each cost term to its fitted :class:`~sklearn:sklearn.linear_model.HuberRegressor`
(or the custom ``outlier_estimator``).  The value is ``None`` if that term was not
selected for outlier correction or had zero variance:

.. code-block:: python

    print(model.outlier_estimators_)
    # {'tp_cost': None, 'tn_cost': None, 'fn_cost': None, 'fp_cost': HuberRegressor()}

    # Inspect the regression coefficients used for fp_cost imputation
    print(model.outlier_estimators_['fp_cost'].coef_)


sklearn Integration
===================

:class:`~empulse.models.RobustCSClassifier` is a fully sklearn-compatible meta-estimator.
Enable :ref:`metadata routing <sklearn:metadata_routing>` to route instance-dependent
costs through pipelines and cross-validation.

Pipeline with cross-validation
-------------------------------

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.models import CSBoostClassifier, RobustCSClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    fn_cost = np.random.default_rng(0).uniform(1, 5, size=len(y))
    fp_cost = 5.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            RobustCSClassifier(CSBoostClassifier()).set_fit_request(fn_cost=True, fp_cost=True),
        ),
    ])

    scores = cross_val_score(pipeline, X, y, params={'fn_cost': fn_cost, 'fp_cost': fp_cost})
    print(scores.mean())

Hyperparameter search
---------------------

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.metrics import expected_cost_loss
    from empulse.models import CSLogitClassifier, RobustCSClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    fn_cost = np.random.default_rng(0).uniform(1, 5, size=len(y))
    fp_cost = 5.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            RobustCSClassifier(CSLogitClassifier()).set_fit_request(fn_cost=True, fp_cost=True),
        ),
    ])

    scorer = (
        make_scorer(
            expected_cost_loss,
            response_method='predict_proba',
            greater_is_better=False,
            normalize=True,
        )
        .set_score_request(fn_cost=True, fp_cost=True)
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid={'model__estimator__C': np.logspace(-3, 2, 6)},
        scoring=scorer,
    )
    grid_search.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    print(f"Best C: {grid_search.best_params_['model__estimator__C']:.4f}")


References
==========

.. [1] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
       Robust instance-dependent cost-sensitive classification.
       Advances in Data Analysis and Classification, 17(4), 1057-1079.
