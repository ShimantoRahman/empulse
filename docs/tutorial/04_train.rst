.. _tutorial_train:

============================
Training on the Cost Matrix
============================

Measuring cost is useful, but the model is still optimising the wrong thing. This page passes the
cost matrix into training, so the model chases business value instead of accuracy.

Setup
=====

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Cost, Metric, Savings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']

    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        X, y, clv, test_size=0.3, random_state=42, stratify=y
    )

    expected_cost = Metric(dataset.cost_matrix, Cost())
    savings = Metric(dataset.cost_matrix, Savings())

    baseline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000)),
    ]).fit(X_train, y_train)
    y_score_baseline = baseline.predict_proba(X_test)[:, 1]

Cost-sensitive logistic regression
==================================

:class:`~empulse.models.CSLogitClassifier` is a logistic regression whose objective is your cost
matrix instead of log loss. Pass the metric as ``loss``, and the per-customer values at fit time.

.. code-block:: python

    from empulse.models import CSLogitClassifier

    cslogit = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier(loss=expected_cost)),
    ])
    cslogit.fit(X_train, y_train, model__clv=clv_train)

    y_score_cslogit = cslogit.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, cslogit.predict(X_test)):.3f}')
    print(f'cost    : {expected_cost(y_test, y_score_cslogit, clv=clv_test):.2f}')
    print(f'savings : {savings(y_test, y_score_cslogit, clv=clv_test):.4f}')

Note the ``model__clv`` prefix: inside a :class:`~sklearn.pipeline.Pipeline`, parameters are
addressed as ``<step name>__<parameter>``, exactly as for any other scikit-learn estimator.

Read the results carefully
==========================

.. list-table::
    :widths: 34 22 22 22
    :header-rows: 1

    * - Model
      - Accuracy
      - Cost
      - Savings
    * - LogisticRegression
      - 0.889
      - 7.20
      - 0.631
    * - CSLogitClassifier
      - 0.750
      - 1.62
      - 0.917

**Accuracy dropped by 14 points, and that is the model working correctly.**

The cost-sensitive model deliberately misclassifies cheap cases in order to get expensive ones
right. It flags more customers than strictly necessary, accepting false positives — which cost a
few percent of a customer's value — to avoid false negatives, which cost the whole thing. Cost fell
from 7.20 to 1.62, and savings rose from 0.63 to 0.92.

If you judge a cost-sensitive model by accuracy, it will always look worse. That is the wrong
yardstick — it is the yardstick we set out to replace.

Cost-sensitive gradient boosting
================================

:class:`~empulse.models.CSBoostClassifier` applies the same idea to gradient boosting, deriving a
custom objective (gradient and hessian) from your cost matrix. It needs one of XGBoost, LightGBM or
CatBoost installed.

.. code-block:: python

    from empulse.models import CSBoostClassifier

    csboost = CSBoostClassifier(loss=expected_cost)
    csboost.fit(X_train, y_train, clv=clv_train)

    y_score_csboost = csboost.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, csboost.predict(X_test)):.3f}')
    print(f'roc auc : {roc_auc_score(y_test, y_score_csboost):.3f}')
    print(f'cost    : {expected_cost(y_test, y_score_csboost, clv=clv_test):.2f}')
    print(f'savings : {savings(y_test, y_score_csboost, clv=clv_test):.4f}')

This gives the best of both: accuracy **0.898** and AUC **0.956** (both above the baseline), with a
cost of **-2.78** — negative, meaning the campaign now turns a profit of 2.78 per customer.

.. list-table::
    :widths: 34 22 22 22
    :header-rows: 1

    * - Model
      - Accuracy
      - Cost
      - Savings
    * - LogisticRegression
      - 0.889
      - 7.20
      - 0.631
    * - CSLogitClassifier
      - 0.750
      - 1.62
      - 0.917
    * - CSBoostClassifier
      - 0.898
      - **-2.78**
      - **1.143**

Across the 945 test customers, the swing from the baseline is roughly 9,400 — from losing money to
making it, on the same data with the same features.

.. note::
    Savings above 1.0 simply means the model beats the naive baseline the savings score is
    normalised against; it is not capped.

Choosing a backend
==================

:class:`~empulse.models.CSBoostClassifier` defaults to XGBoost, but accepts any of the three
backends as its ``estimator``, along with their hyperparameters:

.. code-block:: python

    from xgboost import XGBClassifier

    tuned = CSBoostClassifier(
        estimator=XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1),
        loss=expected_cost,
    )
    tuned.fit(X_train, y_train, clv=clv_train)

Other cost-sensitive models
===========================

The same ``loss=`` argument works across the model family — :ref:`trees and ensembles <cstree>`,
and the gradient-free :ref:`ProfLogitClassifier <proflogit>` and
:ref:`ProfTreeClassifier <proftree>` for non-smooth objectives. See
:ref:`metric_class_in_model` for which strategies each model supports.

Plain costs, without a Metric
=============================

If your costs are just numbers rather than a formula, skip the cost matrix entirely:

.. code-block:: python

    simple = CSBoostClassifier()
    simple.fit(X_train, y_train, fp_cost=5, fn_cost=100)

This is equivalent to building a generic cost matrix with those four terms, and is often all you
need. Reach for a :class:`~empulse.metrics.Metric` when costs are derived from business parameters,
vary per instance, or you want the same definition used for scoring and training.

Next
====

:doc:`05_threshold` decides how many customers to actually contact.
