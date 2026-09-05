.. _tutorial_pipeline:

================================
Validating the Whole Thing
================================

Everything so far used a single train/test split, which is fine for exposition and not enough to
trust. This page cross-validates the pipeline and tunes it — with one complication worth
understanding.

The complication: costs must follow the folds
=============================================

``clv`` has one value per customer. When cross-validation splits the data, each fold needs *its
own slice* of those values — both to fit the model and to score it. Passing the full array would
silently misalign it with the fold's rows.

scikit-learn solves this with **metadata routing**: estimators and scorers declare which extra
parameters they want, and scikit-learn slices and delivers them per fold. It is opt-in.

.. code-block:: python

    from sklearn import set_config

    set_config(enable_metadata_routing=True)

Cross-validating on savings
===========================

Declare the requests with ``set_fit_request`` on the model and ``set_score_request`` on the scorer,
then hand the full array to ``cross_val_score`` via ``params``.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Cost, Metric, Savings
    from empulse.models import CSBoostClassifier
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']

    expected_cost = Metric(dataset.cost_matrix, Cost())
    savings = Metric(dataset.cost_matrix, Savings())

    scorer = make_scorer(
        savings, response_method='predict_proba', greater_is_better=True
    ).set_score_request(clv=True)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSBoostClassifier(loss=expected_cost).set_fit_request(clv=True)),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scorer, params={'clv': clv})

    print(f'savings per fold: {np.round(scores, 3)}')
    print(f'mean savings: {scores.mean():.4f} (+/- {scores.std():.4f})')

Mean savings of about **1.12**, tight across folds — the single-split result holds up.

Note that ``clv`` is passed once, at the top level, as ``params={'clv': clv}``. Routing takes it
from there: the pipeline forwards it to the model's ``fit`` and the scorer's ``__call__``, each
time sliced to the right rows.

Tuning hyperparameters on business value
========================================

The same scorer drives :class:`~sklearn.model_selection.GridSearchCV`, so model selection optimises
profit rather than accuracy.

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    tuned_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSBoostClassifier(
            estimator=XGBClassifier(), loss=expected_cost
        ).set_fit_request(clv=True)),
    ])

    search = GridSearchCV(
        tuned_pipeline,
        param_grid={'model__estimator__max_depth': [2, 4]},
        cv=3,
        scoring=scorer,
    )
    search.fit(X, y, clv=clv)

    print(f'best params: {search.best_params_}')
    print(f'best savings: {search.best_score_:.4f}')

.. note::
    :class:`~empulse.models.CSBoostClassifier` is given an explicit
    ``estimator=XGBClassifier()`` here. Its default is ``None``, and
    :class:`~sklearn.model_selection.GridSearchCV` cannot call ``set_params`` on ``None`` when
    addressing nested parameters like ``model__estimator__max_depth``. Pass the backend explicitly
    whenever you tune its hyperparameters.

Tuning the business parameters too
==================================

Because the cost matrix keeps its parameters symbolic, they are ordinary metric arguments — so you
can ask what the campaign should look like, not just what the model should look like:

.. code-block:: python

    for accept_rate in (0.2, 0.3, 0.5):
        score = savings(y, pipeline.fit(X, y, clv=clv).predict_proba(X)[:, 1],
                        clv=clv, accept_rate=accept_rate)
        print(f'accept_rate={accept_rate}: savings {score:.4f}')

Treat these as sensitivity analysis rather than something to maximise: ``accept_rate`` is a fact
about the world, not a knob you control. What the sweep tells you is how much your conclusion
depends on an estimate you are not certain of.

Turning routing back off
========================

Metadata routing is a global setting. If other code in your session does not expect it, restore the
default when finished:

.. code-block:: python

    set_config(enable_metadata_routing=False)

What we built
=============

Starting from a logistic regression with a 0.926 AUC that lost 7.20 per customer, we:

1. Wrote the campaign's economics as a cost matrix.
2. Measured the model in money and found it unprofitable.
3. Trained :class:`~empulse.models.CSBoostClassifier` on that cost matrix, reaching a profit of
   2.78 per customer.
4. Chose the operating point — contact the top 26% — from the profit curve instead of defaulting
   to 0.5.
5. Cross-validated the whole pipeline on savings, with per-customer costs routed through each fold.

The same features, the same algorithms; only the objective changed.

Where next
==========

- :ref:`instance_based_cv` — metadata routing in more depth.
- :ref:`threshold_tuning` — including :class:`~sklearn.model_selection.TunedThresholdClassifierCV`
  with Empulse metrics.
- :ref:`robustcs` — when your cost estimates contain outliers.
- :ref:`user_defined_value_metric` — writing a cost matrix for your own domain.
- :doc:`../guide` — the full user guide.
