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

Cross-validating on profit
==========================

Declare the requests with ``set_fit_request`` on the model and ``set_score_request`` on the scorer,
then hand the full array to ``cross_val_score`` via ``params``.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Metric, Profit
    from empulse.models import CSBoostClassifier
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']

    expected_profit = Metric(dataset.cost_matrix, Profit())

    scorer = make_scorer(
        expected_profit, response_method='predict_proba', greater_is_better=True
    ).set_score_request(clv=True)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSBoostClassifier(loss=expected_profit).set_fit_request(clv=True)),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scorer, params={'clv': clv})

    print(f'profit per fold: {np.round(scores, 2)}')
    print(f'mean profit: {scores.mean():.2f} (+/- {scores.std():.2f})')

Mean profit of about **4.47** per customer, with a fold-to-fold spread of **0.79** (folds range
from 3.39 to 5.68). The single-split result of 4.74 sits inside that range, so it was not a lucky
split — though the spread is wide enough that you would not want to report the single number
alone.

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
            estimator=XGBClassifier(), loss=expected_profit
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
    print(f'best profit: {search.best_score_:.2f}')

``max_depth=4`` wins, at about **4.01** per customer over three folds.

.. note::
    :class:`~empulse.models.CSBoostClassifier` is given an explicit
    ``estimator=XGBClassifier()`` here. Its default is ``None``, and
    :class:`~sklearn.model_selection.GridSearchCV` cannot call ``set_params`` on ``None`` when
    addressing nested parameters like ``model__estimator__max_depth``. Pass the backend explicitly
    whenever you tune its hyperparameters.

Sensitivity to the business parameters
======================================

Because the cost matrix keeps its parameters symbolic, they are ordinary metric arguments — so you
can ask how much the conclusion depends on the estimates behind it:

.. code-block:: python

    fitted = pipeline.fit(X, y, clv=clv)
    y_score = fitted.predict_proba(X)[:, 1]

    for accept_rate in (0.2, 0.3, 0.5):
        score = expected_profit(y, y_score, clv=clv, accept_rate=accept_rate)
        print(f'accept_rate={accept_rate}: profit {score:.2f}')

Profit runs from **3.25** to **8.72** as the acceptance rate moves from 0.2 to 0.5. Treat this as
sensitivity analysis rather than something to maximise: ``accept_rate`` is a fact about the world,
not a knob you control. What the sweep tells you is how much your reported figure depends on an
estimate you are not certain of — and here, a lot. If you cannot pin that number down,
:func:`~empulse.metrics.empc_score` from :doc:`03_evaluate` integrates over the uncertainty instead
of guessing.

Turning routing back off
========================

Metadata routing is a global setting. If other code in your session does not expect it, restore the
default when finished:

.. code-block:: python

    set_config(enable_metadata_routing=False)

What we built
=============

Starting from a logistic regression with a 0.926 AUC that earned 2.28 per customer — 42% of what a
perfect model would have earned — we:

1. Wrote the campaign's economics as a cost matrix.
2. Measured the model in money, and found most of the available value was going uncaptured.
3. Trained :class:`~empulse.models.CSBoostClassifier` on that cost matrix, reaching **4.74** per
   customer, or 88% of the ceiling.
4. Chose the operating point from the profit curve instead of defaulting to 0.5, reaching **4.90**
   while contacting fewer customers.
5. Cross-validated the whole pipeline on profit, with per-customer costs routed through each fold.

The same features, the same algorithms; only the objective changed.

Where next
==========

- :ref:`instance_based_cv` — metadata routing in more depth.
- :ref:`threshold_tuning` — including :class:`~sklearn.model_selection.TunedThresholdClassifierCV`
  with Empulse metrics.
- :ref:`robustcs` — when your cost estimates contain outliers.
- :ref:`user_defined_value_metric` — writing a cost matrix for your own domain.
- :doc:`../guide` — the full user guide.
