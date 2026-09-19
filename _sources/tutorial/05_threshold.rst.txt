.. _tutorial_threshold:

=================================
Deciding Who to Actually Contact
=================================

A trained model gives every customer a score. Turning those scores into a campaign needs one more
decision: where to draw the line. The default 0.5 is a convention, not an answer — and with
asymmetric costs it is almost always wrong.

Setup
=====

.. code-block:: python

    import numpy as np
    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import MaxProfit, Metric, Profit
    from empulse.models import CSBoostClassifier
    from sklearn.model_selection import train_test_split

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']

    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        X, y, clv, test_size=0.3, random_state=42, stratify=y
    )

    expected_profit = Metric(dataset.cost_matrix, Profit())
    model = CSBoostClassifier(loss=expected_profit).fit(X_train, y_train, clv=clv_train)
    y_score = model.predict_proba(X_test)[:, 1]

How many customers should we contact?
=====================================

The maximum-profit framework answers this directly. :meth:`~empulse.metrics.Metric.optimal_rate`
returns the *fraction of the customer base* to target — a number a campaign manager can act on
without knowing anything about probabilities.

.. code-block:: python

    max_profit = Metric(dataset.cost_matrix, MaxProfit())

    target_fraction = max_profit.optimal_rate(y_test, y_score, clv=clv_test)
    print(f'contact the top {target_fraction:.1%} of customers')

Contact roughly the top **18%**. Note that this is a decision, not a score: as
:doc:`03_evaluate` warned, :class:`~empulse.metrics.MaxProfit` averages ``clv`` away, so its
*value* is inflated — but the cut-off it picks is still a sensible operating point, and we can
check it with a metric that does not average.

One threshold, or one per customer?
===================================

:class:`~empulse.metrics.Profit` keeps every customer's own value, so its break-even point differs
per row. Asking it for a threshold returns **one threshold per customer**, not a single number:

.. code-block:: python

    thresholds = expected_profit.optimal_threshold(y_test, y_score, clv=clv_test)

    print(f'{np.shape(thresholds)}, mean {np.mean(thresholds):.4f}, '
          f'min {np.min(thresholds):.4f}, max {np.max(thresholds):.4f}')

945 thresholds, averaging **0.21** but ranging from **0.15** to **1.0**. It is worth contacting a
high-value customer at a much lower churn probability than a low-value one. That is precisely the
behaviour a single global threshold cannot express, and one of the strongest reasons to use
instance-dependent costs.

Now compare the three operating points on the metric that honours per-customer values:

.. code-block:: python

    rate_cut = np.sort(y_score)[::-1][int(round(target_fraction * len(y_score))) - 1]

    at_half = expected_profit(y_test, y_score, clv=clv_test)
    at_rate = expected_profit(y_test, (y_score >= rate_cut).astype(float), clv=clv_test)
    per_row = expected_profit(y_test, (y_score >= thresholds).astype(float), clv=clv_test)

    print(f'default 0.5          : {at_half:.2f}')
    print(f'max-profit rate      : {at_rate:.2f}')
    print(f'per-customer cut-off : {per_row:.2f}')
    print(f'per-customer contacts: {(y_score >= thresholds).mean():.1%} of the base')

Deciding at 0.5 earns **4.74**; contacting the top 18% earns **4.81**; giving each customer their
own break-even point earns **4.90**, while contacting only **14%** of the base. The last is both
the most profitable and the cheapest to run — it declines customers the others contact, because
for them the discount was never going to pay for itself.

.. _tutorial_degenerate_costs:

A caveat on this dataset
========================

.. note::
    132 of the 3150 customers have a Customer Value of exactly **0** (35 of them in this test set).
    Contacting them can only lose money: there is no value to retain, and the call still costs
    something. :meth:`~empulse.metrics.Metric.optimal_threshold` handles this correctly by
    assigning them a threshold of ``1.0`` — never contact — rather than failing. That is the
    ``max 1.0`` seen above.

    You will see a ``UserWarning`` here saying the computed threshold fell outside ``[0, 1]`` and
    was clipped. That is this situation: for a customer worth nothing, the break-even point is off
    the end of the scale, and clipping it to ``1.0`` is the right answer.

    Worth knowing generally: a cost matrix that collapses to zero for a row is a modelling signal,
    not just a numerical edge case. It says those customers carry no business consequence either
    way. If **every** term genuinely cancels, there is no break-even point at all, and
    ``optimal_threshold``/``optimal_rate`` raise a ``ValueError`` saying the matrix is degenerate.

Letting a meta-estimator do it
==============================

Rather than thresholding by hand, :class:`~empulse.models.CSThresholdClassifier` wraps any
probabilistic estimator and overrides its ``predict`` to use the cost-optimal threshold. It can
learn the threshold at fit time, or take costs at predict time so one fitted model serves several
cost scenarios.

.. code-block:: python

    from empulse.models import CSThresholdClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    thresholded = CSThresholdClassifier(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000)),
        ]),
    )
    thresholded.fit(X_train, y_train)

    predictions = thresholded.predict(X_test, fp_cost=5, fn_cost=100)
    print(f'targeting {predictions.mean():.1%} of customers')

:class:`~empulse.models.CSRateClassifier` is its sibling for when you must contact a fixed
*fraction* — a call centre with finite capacity, say — rather than everyone above a cut-off. See
:ref:`threshold_tuning` for both.

Thresholds move with the business, not the model
================================================

The optimal cut-off depends on the cost matrix, so changing campaign economics changes who to
contact — with no retraining:

.. code-block:: python

    for incentive in (0.02, 0.05, 0.20):
        rate = max_profit.optimal_rate(
            y_test, y_score, clv=clv_test, incentive_fraction=incentive
        )
        print(f'incentive {incentive:.0%} of value -> target {rate:.1%}')

A 2% discount is worth offering to the top **29%**; at 5% that falls to **18%**, and at 20% only
the top **13%** are worth approaching. The cheaper the offer, the further down the ranking it pays
to go.

This is the practical payoff of keeping business parameters symbolic: you re-answer the question by
passing a different number, not by rebuilding a model.

Next
====

:doc:`06_pipeline` validates all of this properly with cross-validation, and tunes it.
