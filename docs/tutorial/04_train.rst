.. _tutorial_train:

============================
Training on the Cost Matrix
============================

Measuring profit is useful, but the model is still optimising the wrong thing. This page passes the
cost matrix into training, so the model chases business value instead of accuracy.

Setup
=====

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Metric, Profit
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

    expected_profit = Metric(dataset.cost_matrix, Profit())

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
        ('model', CSLogitClassifier(loss=expected_profit)),
    ])
    cslogit.fit(X_train, y_train, model__clv=clv_train)

    y_score_cslogit = cslogit.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, cslogit.predict(X_test)):.3f}')
    print(f'profit  : {expected_profit(y_test, y_score_cslogit, clv=clv_test):.2f}')

Note the ``model__clv`` prefix: inside a :class:`~sklearn.pipeline.Pipeline`, parameters are
addressed as ``<step name>__<parameter>``, exactly as for any other scikit-learn estimator.

.. note::
    Passing :class:`~empulse.metrics.Profit` or :class:`~empulse.metrics.Cost` as ``loss`` trains
    exactly the same model. Models optimise the metric as a loss, which removes the sign
    difference, so the choice is presentation only — the same one you made on the previous page.

Read the results carefully
==========================

.. list-table::
    :widths: 34 22 22 22
    :header-rows: 1

    * - Model
      - Accuracy
      - ROC AUC
      - Profit
    * - LogisticRegression
      - 0.889
      - 0.926
      - 2.28
    * - CSLogitClassifier
      - 0.838
      - 0.918
      - 2.58

**Accuracy dropped by five points, and that is the model working correctly.**

The cost-sensitive model deliberately misclassifies cheap cases in order to get expensive ones
right. It flags more customers than a log-loss model would, accepting false positives — which cost
a few percent of a customer's value — to avoid missing churners, which forgoes the whole retained
value. Profit rose from 2.28 to 2.58.

If you judge a cost-sensitive model by accuracy, it will always look worse. That is the wrong
yardstick — it is the yardstick we set out to replace.

Cost-sensitive gradient boosting
================================

:class:`~empulse.models.CSBoostClassifier` applies the same idea to gradient boosting, deriving a
custom objective (gradient and hessian) from your cost matrix. It needs one of XGBoost, LightGBM or
CatBoost installed.

.. code-block:: python

    from empulse.models import CSBoostClassifier

    csboost = CSBoostClassifier(loss=expected_profit)
    csboost.fit(X_train, y_train, clv=clv_train)

    y_score_csboost = csboost.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, csboost.predict(X_test)):.3f}')
    print(f'roc auc : {roc_auc_score(y_test, y_score_csboost):.3f}')
    print(f'profit  : {expected_profit(y_test, y_score_csboost, clv=clv_test):.2f}')

.. list-table::
    :widths: 34 22 22 22
    :header-rows: 1

    * - Model
      - Accuracy
      - ROC AUC
      - Profit
    * - LogisticRegression
      - 0.889
      - 0.926
      - 2.28
    * - CSLogitClassifier
      - 0.838
      - 0.918
      - 2.58
    * - CSBoostClassifier
      - **0.960**
      - **0.974**
      - **4.74**

Here the extra capacity of a boosted ensemble means there is no trade to make: it beats the
baseline on accuracy, AUC *and* profit at once. Profit more than doubles, from 2.28 to 4.74 —
about 4,481 across the 945 test customers, against the baseline's 2,155.

Recall the ceiling from :doc:`01_problem`: a model that knew exactly who would churn earns 5.40.
The baseline captured 42% of that. This model captures **88%**, on the same features and the same
split. Only the objective changed.

.. note::
    Do not read the accuracy column as a rule. Whether a cost-sensitive model gains or loses
    accuracy depends on the model class and the cost matrix — :class:`~empulse.models.CSLogitClassifier`
    gave up five points here, :class:`~empulse.models.CSBoostClassifier` gained seven. Neither is
    evidence about profit, which is the column that matters.

Choosing a backend
==================

:class:`~empulse.models.CSBoostClassifier` defaults to XGBoost, but accepts any of the three
backends as its ``estimator``, along with their hyperparameters:

.. code-block:: python

    from xgboost import XGBClassifier

    tuned = CSBoostClassifier(
        estimator=XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1),
        loss=expected_profit,
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
