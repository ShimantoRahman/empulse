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

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from empulse.metrics import Cost, Metric
    from empulse.models import CSBoostClassifier
    from sklearn.model_selection import train_test_split

    dataset = fetch_iranian_churn(backend=pd)
    X, y = dataset.data, dataset.target
    clv = dataset.instance_costs['clv']

    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        X, y, clv, test_size=0.3, random_state=42, stratify=y
    )

    expected_cost = Metric(dataset.cost_matrix, Cost())
    model = CSBoostClassifier(loss=expected_cost).fit(X_train, y_train, clv=clv_train)
    y_score = model.predict_proba(X_test)[:, 1]

How many customers should we contact?
=====================================

The maximum-profit framework answers this directly. :meth:`~empulse.metrics.Metric.optimal_rate`
returns the *fraction of the customer base* to target — a number a campaign manager can act on
without knowing anything about probabilities.

.. code-block:: python

    from empulse.metrics import empc_score

    profit = empc_score(y_test, y_score, clv=clv_test)
    target_fraction = empc_score.optimal_rate(y_test, y_score, clv=clv_test)

    print(f'expected profit per customer: {profit:.2f}')
    print(f'contact the top {target_fraction:.1%} of customers')

Contact roughly the top **26%**, for an expected profit of about **19.71** per customer.

Converting a rate into a threshold
==================================

A fraction is operationally useful, but a classifier needs a score cut-off.
:func:`~empulse.metrics.classification_threshold` converts one to the other:

.. code-block:: python

    from empulse.metrics import classification_threshold

    threshold = classification_threshold(
        y_test, y_score, customer_threshold=target_fraction
    )

    print(f'threshold: {threshold:.4f}')

    targeted = (y_score >= threshold).astype(int)
    print(f'targeting {targeted.mean():.1%} of customers')

.. note::
    The removed ``empc``/``mpc`` functions used to return this rate alongside the score as a tuple.
    It is now ``.optimal_rate(...)``, which takes the same parameters as the metric itself.

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

.. _tutorial_degenerate_costs:

A caveat on this dataset
========================

.. warning::
    132 of the 3150 customers have a Customer Value of exactly **0**. For those rows every term of
    the cost matrix is zero, so no decision is better than any other and the cost-optimal threshold
    is genuinely undefined.

    As a result, :meth:`~empulse.metrics.Metric.optimal_threshold` and
    :meth:`~empulse.metrics.Metric.optimal_rate` raise a ``ValueError`` for the
    :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings` strategies here:

    .. code-block:: text

        ValueError: Cannot compute the optimal threshold/rate: the cost matrix is
        degenerate for at least one sample ...

    This is why the section above uses :func:`~empulse.metrics.empc_score`. The
    :class:`~empulse.metrics.MaxProfit` strategy derives its operating point from the ROC convex
    hull across the whole population, so individual zero-value customers do not break it.

If you need a cost-based threshold on data like this, drop or impute the zero-value rows first:

.. code-block:: python

    import numpy as np

    keep = np.asarray(clv_test) > 0
    cost_threshold = expected_cost.optimal_threshold(
        np.asarray(y_test)[keep], y_score[keep], clv=np.asarray(clv_test)[keep]
    )
    print(f'thresholds: {np.shape(cost_threshold)}, mean {np.mean(cost_threshold):.4f}')

Note that this returns **one threshold per customer**, not a single number. Because ``clv`` differs
per row, so does the break-even point: it is worth contacting a high-value customer at a much lower
churn probability than a low-value one. That is precisely the behaviour a single global threshold
cannot express, and one of the strongest reasons to use instance-dependent costs.

Worth knowing generally: a cost matrix that collapses to zero is a modelling signal, not just a
numerical edge case. It says those customers carry no business consequence either way, so any
prediction for them is equally fine.

Thresholds move with the business, not the model
================================================

The optimal threshold depends on the cost matrix, so changing campaign economics changes who to
contact — with no retraining:

.. code-block:: python

    cheap_campaign = empc_score.optimal_rate(
        y_test, y_score, clv=clv_test, contact_cost=0.1
    )
    print(f'cheaper contact -> target {cheap_campaign:.1%}')

Cheaper outreach makes it worth contacting more customers. This is the practical payoff of keeping
business parameters symbolic: you re-answer the question by passing a different number, not by
rebuilding a model.

Next
====

:doc:`06_pipeline` validates all of this properly with cross-validation, and tunes it.
