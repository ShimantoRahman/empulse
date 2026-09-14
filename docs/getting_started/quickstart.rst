.. _quickstart:

==========
Quickstart
==========

This page takes about five minutes and ends with a model that captures twice as much business
value as a conventional one. We will use a real telecom churn dataset where each customer has their
own lifetime value.

.. note::
    Needs ``pip install empulse[boosting] pandas``. The dataset is downloaded once and cached
    under ``~/empulse_data``.

A good model, by the usual standards
====================================

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
money, the retention offer costs a fraction of their value, and keeping a churner earns the value
you retain. Wrap it in a :class:`~empulse.metrics.Metric` with the
:class:`~empulse.metrics.Profit` strategy to get the expected profit per customer.

.. code-block:: python

    import numpy as np
    from empulse.metrics import Metric, Profit

    expected_profit = Metric(dataset.cost_matrix, Profit())

    baseline_profit = expected_profit(y_test, y_score_baseline, clv=clv_test)
    oracle = expected_profit(y_test, y_test.to_numpy().astype(float), clv=clv_test)

    print(f'profit per customer: {baseline_profit:.2f}')
    print(f'a perfect model    : {oracle:.2f}')

The campaign earns about **2.28** per customer — but a model that knew exactly who would churn
would earn **5.40**. The good-looking model captures **42%** of what was available, and nothing in
the accuracy or the AUC hints at the rest.

It was never told that missing a high-value churner forgoes far more than wasting a discount on a
loyal customer.

Train on the cost matrix instead
================================

Pass the very same metric to a model as its ``loss``, and hand it the per-customer values at fit
time.

.. code-block:: python

    from empulse.models import CSBoostClassifier

    model = CSBoostClassifier(loss=expected_profit)
    model.fit(X_train, y_train, clv=clv_train)

    y_score = model.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}')
    print(f'profit per customer: {expected_profit(y_test, y_score, clv=clv_test):.2f}')

Profit rises to about **4.74** per customer — more than double the baseline, and **88%** of the
achievable ceiling. Accuracy went *up* too, to 0.960, though that is a side effect rather than the
goal.

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

    mean_clv = float(clv_test.mean())

    target_fraction = empc_score.optimal_rate(
        y_test, y_score, clv=mean_clv, incentive_cost=0.05 * mean_clv
    )
    threshold = classification_threshold(y_test, y_score, customer_threshold=target_fraction)

    print(f'contact the top {target_fraction:.1%} of customers (score >= {threshold:.3f})')

This says to contact roughly the top **19%** of customers.

.. note::
    :func:`~empulse.metrics.empc_score` carries its own cost matrix, whose incentive is a fixed
    amount (``incentive_cost``) rather than a fraction of each customer's value, and whose ``clv``
    is a single global number. Both are therefore passed as scalars here, with the incentive
    written as ``0.05 * mean_clv`` to match the 5% fraction this dataset uses. Left at its own
    defaults it would describe a different campaign.

Putting the threshold to work
=============================

Apply it to the scores you already have, and check what it earns:

.. code-block:: python

    targeted = (y_score >= threshold).astype(int)

    print(f'targeting {targeted.mean():.1%} of customers')
    print(f'profit: {expected_profit(y_test, targeted.astype(float), clv=clv_test):.2f}')

Contacting that 19% earns **4.79** per customer, slightly more than deciding at 0.5 — and the
tutorial shows how a per-customer threshold does better still, while contacting fewer people.

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
