.. _tutorial_problem:

=====================
The Problem with AUC
=====================

The business problem
====================

A telecom company loses customers every month. They can run a retention campaign: call a customer
they believe is about to leave and offer a discount to stay. The campaign is not free — the call
costs money, and so does the discount — so they cannot contact everyone.

The obvious plan is to build a churn classifier, contact whoever it flags, and be done. This page
shows why that plan quietly loses money, even when the classifier is good.

The data
========

Each row is a customer, described by usage and account features, labelled with whether they
churned. Crucially, each customer also has a **Customer Value** — how much they are worth to the
company.

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']

    print(f'customers: {len(y)}, churn rate: {y.mean():.1%}')
    print(f'customer value: mean {clv.mean():.0f}, median {pd.Series(clv).median():.0f}, max {clv.max():.0f}')

About **15.7%** of the 3150 customers churn. Customer value averages 471 but has a median of only
229 — the distribution is heavily skewed, so a minority of customers carry most of the value.

That skew is the whole story. If customers were equally valuable, ranking by churn probability
would be fine. They are not, so a customer with a 40% chance of leaving and a value of 2000 matters
far more than one with an 80% chance and a value of 50.

A conventional baseline
=======================

Train an ordinary logistic regression and evaluate it the ordinary way.

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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

Accuracy about **0.889**, ROC AUC about **0.926**. By conventional standards this is a good model,
and in most projects this is where the modelling stops.

What it is actually worth
=========================

Now measure the same predictions in money. The dataset ships a cost matrix describing the
campaign's economics (the next page builds it from scratch); wrapping it in a
:class:`~empulse.metrics.Metric` gives the expected cost per customer.

.. code-block:: python

    from empulse.metrics import Cost, Metric

    expected_cost = Metric(dataset.cost_matrix, Cost())

    baseline_cost = expected_cost(y_test, y_score_baseline, clv=clv_test)
    print(f'cost per customer: {baseline_cost:.2f}')

The result is about **+7.20**, and positive means cost. Acting on this model's predictions *loses*
7.20 per customer — roughly 6,800 across the test set.

Why it goes wrong
=================

Nothing is broken. The model does exactly what it was asked to do: separate churners from
non-churners, treating every mistake as equally bad. But in this campaign they are not:

- Missing a high-value churner costs their entire lifetime value.
- Contacting a customer who was never going to leave costs only the call and the discount.

Those differ by orders of magnitude, and the model was never told. It spends its capacity getting
cheap cases right, because accuracy rewards that just as much as getting expensive cases right.

Accuracy and AUC cannot detect this failure — both looked fine. Only a metric that knows the price
of each outcome can.

.. note::
    This is not an argument against AUC. It is an argument against using AUC *alone* to decide
    whether a model is fit for a decision that has costs attached.

Next
====

:doc:`02_cost_matrix` writes down the campaign's economics as a cost matrix, which is the object
everything else in this tutorial is built on.
