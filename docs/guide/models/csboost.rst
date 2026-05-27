.. _csboost:

====================================================
Cost-Sensitive Gradient Boosting (CSBoost & B2Boost)
====================================================

:class:`~empulse.models.CSBoostClassifier` [1]_ is a cost-sensitive gradient boosting model that
injects a cost-sensitive objective directly into the boosting algorithm so that every tree split
minimises a business loss rather than a standard logistic loss.

It wraps three popular gradient boosting backends — XGBoost, LightGBM, and CatBoost — behind a
unified sklearn-compatible interface.  If no estimator is supplied, an
:class:`xgboost:xgboost.XGBClassifier` with default hyperparameters is used.

.. list-table:: Supported backends
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - Class
     - Notes
   * - XGBoost
     - :class:`xgboost:xgboost.XGBClassifier`
     - Default when no estimator is passed
   * - LightGBM
     - :class:`lightgbm:lightgbm.LGBMClassifier`
     - Pass as ``estimator``; install via ``pip install lightgbm``
   * - CatBoost
     - :class:`catboost.CatBoostClassifier`
     - Pass as ``estimator``; install via ``pip install catboost``.
       ``sample_weight`` cannot be used because it is reserved for internal index passing.

.. note::
    You can install all three backends with ``pip install empulse[optional]``.
    If you try to use a backend that isn't installed,
    you'll get an informative error message with installation instructions.

Quick Start
===========

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSBoostClassifier

    X, y = make_classification(n_samples=1000, random_state=0)

    model = CSBoostClassifier(fp_cost=5, fn_cost=1)
    model.fit(X, y)
    y_proba = model.predict_proba(X)

Choosing a Backend
==================

Pass any supported estimator to the ``estimator`` argument:

.. code-block:: python

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from empulse.models import CSBoostClassifier

    # XGBoost (explicit)
    model_xgb = CSBoostClassifier(
        XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05),
        fp_cost=5, fn_cost=1,
    )

    # LightGBM
    model_lgbm = CSBoostClassifier(
        LGBMClassifier(n_estimators=200, num_leaves=31),
        fp_cost=5, fn_cost=1,
    )

    # CatBoost
    model_cat = CSBoostClassifier(
        CatBoostClassifier(iterations=200, depth=4, verbose=0),
        fp_cost=5, fn_cost=1,
    )

.. note::
    When using :class:`~sklearn:sklearn.model_selection.GridSearchCV` to tune
    estimator hyperparameters you **must** supply an explicit estimator instance.
    Without one, sklearn would try to set the parameter on ``None``, which raises
    an error.

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from empulse.models import CSBoostClassifier

    model = CSBoostClassifier(XGBClassifier(), fp_cost=5, fn_cost=1)
    grid_search = GridSearchCV(
        model,
        param_grid={'estimator__max_depth': [3, 5, 7]},
        cv=3,
    )


Cost Matrix
===========

Constant costs
--------------

Pass scalars at construction to use the same cost for every sample:

.. code-block:: python

    from empulse.models import CSBoostClassifier

    model = CSBoostClassifier(
        tp_cost=0,    # no benefit for correct positives
        fp_cost=5,    # cost of contacting a non-churner
        tn_cost=0,
        fn_cost=1,    # cost of missing a churner
    )

Instance-dependent costs
------------------------

Pass per-sample arrays to ``fit`` to give each observation its own cost profile
(e.g. individual Customer Lifetime Values):

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSBoostClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 1000, size=len(y))

    model = CSBoostClassifier(fn_cost=1)
    model.fit(X, y, tp_cost=clv)   # instance-dependent TP benefit

Costs can be mixed: pass scalar class-level costs at construction and override
selected terms with arrays at ``fit`` time.  Costs provided to ``fit`` always
take precedence.


Custom Loss Function
====================

The ``loss`` parameter accepts a :class:`~empulse.metrics.Metric` instance so you can
define your own business objective using the symbolic cost matrix API.  Both
:class:`~empulse.metrics.Cost` (expected cost minimisation) and
:class:`~empulse.metrics.MaxProfit` (profit maximisation) strategies are supported.

.. code-block:: python

    import sympy
    from sklearn.datasets import make_classification
    from empulse.metrics import Metric, MaxProfit, CostMatrix
    from empulse.models import CSBoostClassifier

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
    )
    profit_metric = Metric(cost_matrix, MaxProfit())

    X, y = make_classification(n_samples=1000, random_state=0)
    clvs = 200 + 100 * abs(y - 0.5)   # toy instance-dependent CLV

    model = CSBoostClassifier(loss=profit_metric)
    model.fit(X, y, clv=clvs, incentive_cost=10, contact_cost=1, accept_rate=0.3)

.. note::
    ``MaxProfit`` re-evaluates the gradient and hessian from the current round's
    predicted scores at every boosting iteration, enabling dynamic threshold
    optimisation during training.  ``Cost`` and ``Savings`` are faster because
    they pre-compute a constant gradient vector before training starts.

Read the :ref:`User Guide <user_defined_value_metric>` for how to build custom
:class:`~empulse.metrics.Metric` definitions, and :ref:`metric_class_in_model`
for the full strategy compatibility matrix.


sklearn Integration
===================

``CSBoostClassifier`` is fully sklearn-compatible and works inside
:class:`~sklearn:sklearn.pipeline.Pipeline`,
:class:`~sklearn:sklearn.model_selection.cross_val_score`, and
:class:`~sklearn:sklearn.model_selection.GridSearchCV`.
When instance-dependent costs must flow through cross-validation you need to
enable :ref:`metadata routing <sklearn:metadata_routing>`.

Pipeline with cross-validation
-------------------------------

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.models import CSBoostClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    fn_cost = np.random.default_rng(0).uniform(1, 5, size=len(y))
    fp_cost = 5.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSBoostClassifier().set_fit_request(fn_cost=True, fp_cost=True)),
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
    from xgboost import XGBClassifier
    from empulse.metrics import expected_cost_loss
    from empulse.models import CSBoostClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    fn_cost = np.random.default_rng(0).uniform(1, 5, size=len(y))
    fp_cost = 5.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            CSBoostClassifier(XGBClassifier(n_jobs=2)).set_fit_request(fn_cost=True, fp_cost=True),
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
        param_grid={'model__estimator__learning_rate': np.logspace(-3, 0, 5)},
        scoring=scorer,
    )
    grid_search.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    print(f"Best learning rate: {grid_search.best_params_['model__estimator__learning_rate']:.4f}")


.. _b2boost:

B2Boost
=======

:class:`~empulse.models.B2BoostClassifier` is a use-case-specific subclass of
:class:`~empulse.models.CSBoostClassifier` designed for **B2B customer churn** retention
campaigns [2]_.  Rather than exposing the raw TP/FP/TN/FN cost matrix, it takes four
business-meaningful parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Meaning
   * - ``clv``
     - 200
     - Customer Lifetime Value — can be a per-sample array
   * - ``incentive_fraction``
     - 0.05
     - Fraction of CLV spent on the retention incentive
   * - ``contact_cost``
     - 15
     - Fixed cost of contacting a customer
   * - ``accept_rate``
     - 0.3
     - Probability that a churner accepts the retention offer

The underlying cost model is:

.. math::

    \text{TP benefit} &= \gamma \bigl(\text{CLV} - \delta \cdot \text{CLV} - f\bigr)
                         - (1-\gamma)\,f \\
    \text{FP cost}    &= \delta \cdot \text{CLV} + f

where :math:`\gamma` is the accept rate, :math:`\delta` the incentive fraction, and
:math:`f` the contact cost.  ``CLV`` is instance-dependent (passed as an array to ``fit``);
the remaining parameters are class-level constants.

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier
    from empulse.models import B2BoostClassifier

    X, y = make_classification(n_samples=1000, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 500, size=len(y))

    model = B2BoostClassifier(
        XGBClassifier(n_estimators=100, max_depth=3),
        accept_rate=0.3,
        incentive_fraction=0.05,
        contact_cost=10,
    )
    model.fit(X, y, clv=clv)
    y_proba = model.predict_proba(X)

Constant CLV
------------

When all customers have the same lifetime value, pass a scalar:

.. code-block:: python

    from empulse.models import B2BoostClassifier

    model = B2BoostClassifier(clv=300, accept_rate=0.25, incentive_fraction=0.1, contact_cost=5)
    model.fit(X, y)

Pipeline with cross-validation
-------------------------------

.. code-block:: python

    import numpy as np
    from sklearn import set_config
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from empulse.models import B2BoostClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 500, size=len(y))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            B2BoostClassifier(contact_cost=10).set_fit_request(clv=True),
        ),
    ])

    scores = cross_val_score(pipeline, X, y, params={'clv': clv})
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
    from xgboost import XGBClassifier
    from empulse.metrics import empb_score
    from empulse.models import B2BoostClassifier

    set_config(enable_metadata_routing=True)

    X, y = make_classification(n_samples=500, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 500, size=len(y))
    contact_cost = 10.0

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (
            'model',
            B2BoostClassifier(
                XGBClassifier(),
                contact_cost=contact_cost,
            ).set_fit_request(clv=True),
        ),
    ])

    scorer = (
        make_scorer(
            empb_score,
            response_method='predict_proba',
            greater_is_better=True,
            contact_cost=contact_cost,
        )
        .set_score_request(clv=True)
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid={'model__estimator__learning_rate': np.logspace(-3, 0, 5)},
        scoring=scorer,
    )
    grid_search.fit(X, y, clv=clv)
    print(f"Best learning rate: {grid_search.best_params_['model__estimator__learning_rate']:.4f}")


References
==========

.. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       European Journal of Operational Research, 297(1), 291-300.

.. [2] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
       B2Boost: Instance-dependent profit-driven modelling of B2B churn.
       Annals of Operations Research, 1-27.
