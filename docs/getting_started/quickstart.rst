.. _quickstart:

==========
Quickstart
==========

This page takes about five minutes and ends with a model that makes money instead of one that
merely scores well. We will use a real telecom churn dataset where each customer has their own
lifetime value.

.. note::
    Needs ``pip install empulse[optional] pandas``. The dataset is downloaded once and cached
    under ``~/empulse_data``.

A good model that loses money
=============================

Start with an ordinary logistic regression and judge it the ordinary way.

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']  # each customer's lifetime value

    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        X, y, clv, test_size=0.3, random_state=42, stratify=y
    )

    baseline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000)),
    ]).fit(X_train, y_train)

    y_score_baseline = baseline.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}')
    print(f'roc auc : {roc_auc_score(y_test, y_score_baseline):.3f}')

This reports an accuracy of about **0.889** and an ROC AUC of about **0.926**. By the usual
standards, a good model.

Now ask what it is worth
========================

The dataset ships a cost matrix describing the retention campaign: contacting a customer costs
money, the retention offer costs money, and losing a customer costs their lifetime value. Wrap it
in a :class:`~empulse.metrics.Metric` with the :class:`~empulse.metrics.Cost` strategy to get the
expected cost per customer.

.. code-block:: python

    from empulse.metrics import Cost, Metric

    expected_cost = Metric(dataset.cost_matrix, Cost())

    baseline_cost = expected_cost(y_test, y_score_baseline, clv=clv_test)
    print(f'cost per customer: {baseline_cost:.2f}')

That prints roughly **7.20** — a *loss* of 7.20 per customer. The model ranks customers well, but
acting on its predictions loses money, because it was never told that missing a high-value churner
is far worse than wasting a discount on a loyal one.

Train on the cost matrix instead
================================

Pass the very same metric to a model as its ``loss``, and hand it the per-customer values at fit
time.

.. code-block:: python

    from empulse.models import CSBoostClassifier

    model = CSBoostClassifier(loss=expected_cost)
    model.fit(X_train, y_train, clv=clv_train)

    y_score = model.predict_proba(X_test)[:, 1]
    model_cost = expected_cost(y_test, y_score, clv=clv_test)

    print(f'accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}')
    print(f'cost per customer: {model_cost:.2f}')

Now the cost is about **-2.78**. A negative cost is a profit: the same campaign now *earns* 2.78
per customer instead of losing 7.20. Across the 945 customers in the test set that is a swing of
roughly 9,400 — and accuracy barely moved (0.898 vs 0.889).

That is the whole point. Accuracy could not see the difference; the cost matrix could.

How many customers should you actually target?
==============================================

A cost matrix also tells you where to put the decision threshold. Instead of the default 0.5, ask
the metric for the profit-maximising operating point.

:func:`~empulse.metrics.empc_score` is the Expected Maximum Profit for Customer Churn — it treats
the offer-acceptance rate as uncertain and reports the profit you can expect at the best cut-off,
along with the fraction of the customer base to contact.

.. code-block:: python

    from empulse.metrics import classification_threshold, empc_score

    profit = empc_score(y_test, y_score, clv=clv_test)
    target_fraction = empc_score.optimal_rate(y_test, y_score, clv=clv_test)
    threshold = classification_threshold(y_test, y_score, customer_threshold=target_fraction)

    print(f'expected profit per customer: {profit:.2f}')
    print(f'contact the top {target_fraction:.1%} of customers (score >= {threshold:.3f})')

This says to contact roughly the top **26%** of customers, at an expected profit of about **19.71**
per customer.

Putting the threshold to work
=============================

Apply it to the scores you already have:

.. code-block:: python

    targeted = (y_score >= threshold).astype(int)

    print(f'targeting {targeted.mean():.1%} of customers')

If you would rather have an estimator that does this for you, wrap the model in
:class:`~empulse.models.CSThresholdClassifier`, which learns the cost-optimal threshold itself, or
in scikit-learn's :class:`~sklearn.model_selection.FixedThresholdClassifier` to pin the value
above. See :ref:`threshold_tuning` for both.

Where next
==========

- :doc:`../tutorial` — the full version of this story, including how the cost matrix is built from
  first principles and how to cross-validate it.
- :doc:`overview` — which Empulse component fits your problem.
- :ref:`prebuilt_churn_metrics` — the other ready-made churn metrics.
- :ref:`user_defined_value_metric` — write a cost matrix for your own use case.
