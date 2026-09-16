.. _quickstart:

==========
Quickstart
==========

This page takes about five minutes and ends with a model that captures noticeably more business
value than a conventional one, trained on the same features. We will use a real telecom churn
dataset where each customer has their own lifetime value.

.. note::
    Needs ``pip install empulse pandas``. The dataset is downloaded once and cached under
    ``~/empulse_data``.

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
    clv = dataset.instance_costs['clv']

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
time. :class:`~empulse.models.CSLogitClassifier` is a logistic regression, exactly like the
baseline above, except its objective is the cost matrix instead of log loss — so any difference in
the numbers below comes purely from what it was trained to optimise, not from a different kind of
model.

.. code-block:: python

    from empulse.models import CSLogitClassifier

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier(loss=expected_profit)),
    ])
    model.fit(X_train, y_train, model__clv=clv_train)

    y_score = model.predict_proba(X_test)[:, 1]

    print(f'accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}')
    print(f'profit per customer: {expected_profit(y_test, y_score, clv=clv_test):.2f}')

Profit rises to about **3.29** per customer — up from 42% to **61%** of the achievable ceiling.
Accuracy drops slightly, to 0.849, and that is fine: the model is deliberately trading cheap
mistakes for expensive ones it now avoids, exactly as :ref:`the tutorial <tutorial_train>` explains
in more detail.

That is the whole point. Accuracy could not see the difference; the cost matrix could.

How many customers should you actually target?
==============================================

A cost matrix also tells you where to put the decision threshold. Instead of the default 0.5, ask
the very same ``expected_profit`` metric for the profit-maximising operating point — no separate
metric or business parameters needed, since ``clv`` already lives in the cost matrix.

.. code-block:: python

    rate = expected_profit.optimal_rate(y_test, y_score, clv=clv_test)

    print(f'contact about {rate:.1%} of customers')

This says to contact roughly **23.5%** of customers.

.. note::
    Running this raises a ``UserWarning`` saying the optimal threshold fell outside ``[0, 1]`` and
    was clipped. Some customers in this dataset have a lifetime value lower than what it costs to
    contact them, so no predicted probability could ever justify targeting them — the cost matrix
    implies an always-negative decision for them. That is inherent to this dataset, not a bug, and
    safe to ignore here.

Putting the threshold to work
=============================

Because ``clv`` is instance-dependent, the break-even point is too: a high-value customer is worth
contacting at a much lower churn probability than a low-value one. Asking for the threshold
therefore returns one per customer, not a single cut-off:

.. code-block:: python

    threshold = expected_profit.optimal_threshold(y_test, y_score, clv=clv_test)
    targeted = (y_score >= threshold).astype(int)

    print(f'targeting {targeted.mean():.1%} of customers')
    print(f'profit: {expected_profit(y_test, targeted.astype(float), clv=clv_test):.2f}')

Contacting that 23.5% earns **3.37** per customer, more than the 3.29 from scoring everyone — while
reaching a quarter of the customer base. :ref:`The full tutorial <tutorial_threshold>` covers this
properly, including cross-validating it.

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
