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
     - `CatBoostClassifier <https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier>`__
     - Pass as ``estimator``; install via ``pip install catboost``.
       Supports :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings` losses only.

.. note::
    You can install all three backends with ``pip install empulse[boosting]``.
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

Pass any supported estimator to the ``estimator`` argument. All three are installed by
``pip install empulse[boosting]``; the cost-sensitive objective is identical, so the choice is the
usual one between the libraries themselves.

.. tab-set::

    .. tab-item:: XGBoost
        :sync: xgboost

        The default. Used when ``estimator`` is left as ``None``.

        .. code-block:: python

            from xgboost import XGBClassifier
            from empulse.models import CSBoostClassifier

            model = CSBoostClassifier(
                XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05),
                fp_cost=5,
                fn_cost=1,
            )

    .. tab-item:: LightGBM
        :sync: lightgbm

        Usually the fastest to fit on wide data.

        .. code-block:: python

            from lightgbm import LGBMClassifier
            from empulse.models import CSBoostClassifier

            model = CSBoostClassifier(
                LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1),
                fp_cost=5,
                fn_cost=1,
            )

    .. tab-item:: CatBoost
        :sync: catboost

        Handles categorical features natively. It only supports :class:`~empulse.metrics.Cost` and
        :class:`~empulse.metrics.Savings` losses: CatBoost computes the objective on chunks of the
        training rows, which the :class:`~empulse.metrics.MaxProfit` and
        :class:`~empulse.metrics.LogCost` objectives cannot be evaluated on.

        .. code-block:: python

            from catboost import CatBoostClassifier
            from empulse.models import CSBoostClassifier

            model = CSBoostClassifier(
                CatBoostClassifier(iterations=200, depth=4, verbose=0),
                fp_cost=5,
                fn_cost=1,
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


Specifying costs
================

``CSBoostClassifier`` accepts costs the same two ways as every other cost-sensitive model in
Empulse: as plain ``tp_cost``/``tn_cost``/``fp_cost``/``fn_cost`` values, scalar or per-sample, or
as a :class:`~empulse.metrics.Metric` passed as ``loss``. The rules — where each may be set, how
constructor and ``fit`` values interact, and how the two forms may be mixed — are in
:ref:`specifying_costs`.

One rule is specific to this model: arguments for the underlying booster go in a dedicated
``fit_params`` dict, since anything else would be ambiguous with a cost or metric parameter.

.. code-block:: python

    from sklearn.datasets import make_classification
    from empulse.models import CSBoostClassifier

    X, y = make_classification(n_samples=500, random_state=0)

    model = CSBoostClassifier(fp_cost=5, fn_cost=1)
    model.fit(X, y, fit_params={'verbose': False})

Custom Loss Function
====================

The ``loss`` parameter accepts any :class:`~empulse.metrics.Metric`, so the objective can be
written from business parameters rather than four flat numbers.
:class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.LogCost` and
:class:`~empulse.metrics.Savings` are fully supported;
:class:`~empulse.metrics.MaxProfit` works but is experimental here, and the two ranking-based
strategies are not available on a gradient-boosted model. The CatBoost backend supports only
:class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings`.
:ref:`metric_class_in_model` has the matrix.

.. code-block:: python

    import sympy
    from empulse.metrics import CostMatrix, Metric, Cost
    from empulse.models import CSBoostClassifier

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
    )
    expected_cost = Metric(cost_matrix, Cost())

    clvs = 200 + 100 * abs(y - 0.5)   # toy instance-dependent CLV

    model = CSBoostClassifier(loss=expected_cost)
    model.fit(X, y, clv=clvs, incentive_cost=10, contact_cost=1, accept_rate=0.3)

See :ref:`user_defined_value_metric` for worked cost matrices to use here.


sklearn Integration
===================

``CSBoostClassifier`` is fully sklearn-compatible and drops into
:class:`~sklearn:sklearn.pipeline.Pipeline`,
:func:`~sklearn:sklearn.model_selection.cross_val_score` and
:class:`~sklearn:sklearn.model_selection.GridSearchCV` unchanged. Per-sample costs reach each fold
through metadata routing, covered in :ref:`instance_based_cv`.

One caveat is specific to this model. Searching over the booster's own hyperparameters means
addressing them as ``model__estimator__*``, and :class:`~sklearn.model_selection.GridSearchCV`
cannot set attributes on the default ``estimator=None``. Pass an explicit booster instance:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from empulse.models import CSBoostClassifier

    X, y = make_classification(n_samples=500, random_state=0)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSBoostClassifier(XGBClassifier(n_jobs=2), fp_cost=5, fn_cost=1)),
    ])

    grid_search = GridSearchCV(
        pipeline,
        param_grid={'model__estimator__learning_rate': np.logspace(-3, 0, 3)},
    )
    grid_search.fit(X, y)
    print(grid_search.best_params_['model__estimator__learning_rate'])


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

Scoring a B2Boost model
-----------------------

:func:`~empulse.metrics.empb_score` is the natural companion metric: it is built from the same B2B
churn cost matrix, so tuning against it optimises the quantity the model was trained on. Pass the
shared business parameters to both.

.. code-block:: python

    import numpy as np
    from sklearn import config_context
    from sklearn.datasets import make_classification
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from empulse.metrics import empb_score
    from empulse.models import B2BoostClassifier

    X, y = make_classification(n_samples=500, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 500, size=len(y))
    contact_cost = 10.0

    with config_context(enable_metadata_routing=True):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (
                'model',
                B2BoostClassifier(
                    XGBClassifier(), contact_cost=contact_cost
                ).set_fit_request(clv=True),
            ),
        ])
        scorer = make_scorer(
            empb_score,
            response_method='predict_proba',
            greater_is_better=True,
            contact_cost=contact_cost,
        ).set_score_request(clv=True)

        grid_search = GridSearchCV(
            pipeline,
            param_grid={'model__estimator__learning_rate': np.logspace(-3, 0, 3)},
            scoring=scorer,
        )
        grid_search.fit(X, y, clv=clv)

    print(grid_search.best_params_['model__estimator__learning_rate'])

See :ref:`instance_based_cv` for the routing mechanics, and :ref:`prebuilt_churn_metrics` for the
other churn metrics that pair with this model.


References
==========

.. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       European Journal of Operational Research, 297(1), 291-300.

.. [2] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
       B2Boost: Instance-dependent profit-driven modelling of B2B churn.
       Annals of Operations Research, 1-27.
