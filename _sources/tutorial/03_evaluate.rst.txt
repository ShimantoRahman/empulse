.. _tutorial_evaluate:

==================
Measuring in Money
==================

A cost matrix is not yet a number. A **strategy** decides how to turn it into one, and different
strategies answer different questions. This page walks through all three on the same predictions,
so the differences are concrete.

Setup
=====

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn
    from sklearn.linear_model import LogisticRegression
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

    y_score = baseline.predict_proba(X_test)[:, 1]

Expected cost: what does this model cost me?
============================================

The :class:`~empulse.metrics.Cost` strategy reports the expected cost per customer, in currency.
Lower is better, and negative means profit.

.. code-block:: python

    from empulse.metrics import Cost, Metric

    expected_cost = Metric(dataset.cost_matrix, Cost())

    print(f'{expected_cost(y_test, y_score, clv=clv_test):.2f}')

About **7.20** per customer. This is the most directly interpretable number: multiply by your
customer base and you have the campaign's bottom line.

Because the business parameters are symbolic, you can ask what-if questions without retraining
anything:

.. code-block:: python

    generous = expected_cost(
        y_test, y_score, clv=clv_test, incentive_fraction=0.15, accept_rate=0.6
    )
    print(f'{generous:.2f}')

A more generous offer that more people accept changes the economics — and the optimal customers to
target change with it.

Savings: how good is that, really?
==================================

7.20 is hard to judge in isolation. The :class:`~empulse.metrics.Savings` strategy divides by a
naive baseline, giving a scale-free ratio comparable across datasets and campaigns — much as
:math:`R^2` does for regression.

.. code-block:: python

    from empulse.metrics import Savings

    savings = Metric(dataset.cost_matrix, Savings())

    print(f'{savings(y_test, y_score, clv=clv_test):.4f}')

About **0.63**: the model captures 63% of the achievable saving relative to the naive rule of
treating everyone the same way. Higher is better, 1.0 is perfect, and 0 means no better than the
naive baseline.

.. warning::
    Both :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings` assume ``y_score``
    holds **calibrated probabilities**, because they weight each outcome by its predicted
    likelihood. If your model outputs uncalibrated scores, calibrate it first — no error is raised,
    the number is just wrong.

Maximum profit: what if the threshold is not fixed?
===================================================

Both metrics above ask "how much does this model cost, given how it decides?" But the decision
threshold is a choice, and 0.5 is rarely the right one.

The :class:`~empulse.metrics.MaxProfit` strategy asks a different question: **at the best possible
cut-off, how much can this model make?** It needs only a ranking, not calibrated probabilities.

.. code-block:: python

    from empulse.metrics import MaxProfit

    max_profit = Metric(dataset.cost_matrix, MaxProfit())

    print(f'{max_profit(y_test, y_score, clv=clv_test):.2f}')

Handling uncertainty in the business parameters
===============================================

So far ``accept_rate`` has been exactly 0.3. Nobody knows it that precisely — it is an estimate,
and the profit you report inherits its uncertainty.

The maximum-profit framework handles this by letting a parameter be a **random variable** rather
than a number. Instead of committing to 0.3, model the acceptance rate as a Beta distribution and
integrate over it. That is the Expected Maximum Profit for Customer Churn, and Empulse ships it
prebuilt as :func:`~empulse.metrics.empc_score`.

.. code-block:: python

    from empulse.metrics import empc_score

    print(f'{empc_score(y_test, y_score, clv=clv_test):.2f}')

About **18.00** expected profit per customer, averaged over plausible acceptance rates rather than
betting on a single guess. Its defaults are ``alpha=6, beta=14``, a Beta distribution with mean
:math:`6/(6+14) = 0.3` — the same 0.3 as before, but now with honest uncertainty around it.

.. note::
    The profit numbers (18.00) and the cost numbers (7.20) are not comparable: they answer
    different questions. Cost measures the model *as it decides today*; maximum profit measures it
    *at its best possible cut-off*. Comparing models is only meaningful within one strategy.

Which should you use?
=====================

.. list-table::
    :widths: 22 78
    :header-rows: 1

    * - Strategy
      - Use when
    * - :class:`~empulse.metrics.Cost`
      - You want the bottom line in currency, and your model is calibrated.
    * - :class:`~empulse.metrics.Savings`
      - You want to compare models or datasets on a common 0-1 scale.
    * - :class:`~empulse.metrics.MaxProfit`
      - The threshold is still open, and/or your business parameters are uncertain.

Using them as scikit-learn scorers
==================================

Any metric can be wrapped with :func:`~sklearn.metrics.make_scorer` and used anywhere scikit-learn
accepts a scorer:

.. code-block:: python

    from sklearn.metrics import make_scorer

    savings_scorer = make_scorer(
        savings, response_method='predict_proba', greater_is_better=True
    )

:doc:`06_pipeline` shows how to route the per-customer ``clv`` through cross-validation so the
scorer sees the right values for each fold.

Next
====

:doc:`04_train` stops merely measuring the cost and starts optimising it.
