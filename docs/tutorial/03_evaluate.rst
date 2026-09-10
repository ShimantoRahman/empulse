.. _tutorial_evaluate:

==================
Measuring in Money
==================

A cost matrix is not yet a number. A **strategy** decides how to turn it into one, and different
strategies answer different questions. This page walks through the ones that matter for this
campaign, on the same predictions, so the differences are concrete.

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

Expected profit: what is this model worth?
==========================================

The :class:`~empulse.metrics.Profit` strategy reports the expected profit per customer, in
currency. Higher is better.

.. code-block:: python

    from empulse.metrics import Metric, Profit

    expected_profit = Metric(dataset.cost_matrix, Profit())

    print(f'{expected_profit(y_test, y_score, clv=clv_test):.2f}')

About **2.28** per customer. This is the most directly interpretable number: multiply by your
customer base and you have the campaign's bottom line.

If you would rather read the same quantity as a cost to minimise, use
:class:`~empulse.metrics.Cost` instead. It is the same computation reported with the opposite sign:

.. code-block:: python

    from empulse.metrics import Cost

    expected_cost = Metric(dataset.cost_matrix, Cost())

    print(f'{expected_cost(y_test, y_score, clv=clv_test):.2f}')

**-2.28** — a negative cost is a profit. Which one you pick is presentation only: they rank models
identically, and a model trained on one is the same model trained on the other. This tutorial uses
:class:`~empulse.metrics.Profit` because a retention campaign is naturally discussed in terms of
what it earns.

Asking what-if questions
========================

Because the business parameters are symbolic, you can vary them without retraining anything:

.. code-block:: python

    generous = expected_profit(
        y_test, y_score, clv=clv_test, incentive_fraction=0.15, accept_rate=0.6
    )
    stingy = expected_profit(
        y_test, y_score, clv=clv_test, incentive_fraction=0.02, accept_rate=0.2
    )

    print(f'generous offer: {generous:.2f}')
    print(f'stingy offer  : {stingy:.2f}')

A bigger discount that more people accept earns **3.68**; a token discount that few accept earns
**1.64**. The model never changed — only the campaign around it did. Which customers are worth
contacting moves with those numbers too, as :doc:`05_threshold` shows.

.. warning::
    :class:`~empulse.metrics.Profit` and :class:`~empulse.metrics.Cost` assume ``y_score`` holds
    **calibrated probabilities**, because they weight each outcome by its predicted likelihood. If
    your model outputs uncalibrated scores, calibrate it first — no error is raised, the number is
    just biased.

Maximum profit: what if the threshold is not fixed?
===================================================

The metric above asks "what is this model worth, given how it decides?" But the decision threshold
is a choice, and 0.5 is rarely the right one.

The :class:`~empulse.metrics.MaxProfit` strategy asks a different question: **at the best possible
cut-off, how much can this model make?** It needs only a ranking, not calibrated probabilities.

.. code-block:: python

    from empulse.metrics import MaxProfit

    max_profit = Metric(dataset.cost_matrix, MaxProfit())

    print(f'{max_profit(y_test, y_score, clv=clv_test):.2f}')
    print(f'{max_profit.optimal_rate(y_test, y_score, clv=clv_test):.1%}')

About **15.40**, by contacting the top **21.5%**.

.. warning::
    That 15.40 is **not** comparable to the 2.28 above, for two reasons.

    The first is the question: 2.28 is what the model earns deciding at 0.5, while 15.40 is what it
    could earn at its best cut-off.

    The second is subtler and specific to :class:`~empulse.metrics.MaxProfit`. It works from
    population-level true- and false-positive rates, which cannot express per-customer values, so
    it **averages ``clv`` away first** — every customer is treated as worth the mean, 475. Since
    the value distribution is heavily skewed (median 228), that flatters the result. Use
    :class:`~empulse.metrics.Profit` for the bottom line, and
    :class:`~empulse.metrics.MaxProfit` for the cut-off. :doc:`05_threshold` compares them
    directly. If you need a maximum-profit measure that honours per-customer values, that is
    :class:`~empulse.metrics.EmpiricalMaxProfit`.

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

    mean_clv = float(clv_test.mean())

    print(f'{empc_score(y_test, y_score, clv=mean_clv, incentive_cost=0.05 * mean_clv):.2f}')
    print(f'{empc_score.optimal_rate(y_test, y_score, clv=mean_clv, incentive_cost=0.05 * mean_clv):.1%}')

About **15.50** expected profit per customer at a target rate of **23.7%**, averaged over plausible
acceptance rates rather than betting on a single guess. Its defaults are ``alpha=6, beta=14``, a
Beta distribution with mean :math:`6/(6+14) = 0.3` — the same 0.3 as before, but now with honest
uncertainty around it.

.. note::
    :func:`~empulse.metrics.empc_score` is built on its own cost matrix, which differs from this
    dataset's in one respect worth knowing. Its incentive is ``incentive_cost``, a **fixed amount**
    per customer, whereas the dataset's is ``incentive_fraction``, a fraction of that customer's
    value. Since :class:`~empulse.metrics.MaxProfit` averages instance-dependent values away
    anyway, ``clv`` is a single global number there too.

    That is why both are passed as scalars above, with the incentive expressed as
    ``0.05 * mean_clv`` to line up with the dataset's 5% fraction. Left at its own defaults
    (``clv=200``, ``incentive_cost=10``, ``contact_cost=1``) it would describe a different campaign
    from this one and return a different number.

.. note::
    :class:`~empulse.metrics.Savings` is the other strategy you will meet in the user guide: it
    divides expected cost by a naive baseline to give a scale-free ratio. It is not used in this
    tutorial because it is not meaningful on this cost matrix — "contact nobody" costs exactly
    zero here, so there is no baseline cost to take a fraction of. See :ref:`choosing_metric` for
    when it does apply.

Which should you use?
=====================

.. list-table::
    :widths: 26 74
    :header-rows: 1

    * - Strategy
      - Use when
    * - :class:`~empulse.metrics.Profit` / :class:`~empulse.metrics.Cost`
      - You want the bottom line in currency, per-customer values honoured, and your model is
        calibrated.
    * - :class:`~empulse.metrics.MaxProfit` / :class:`~empulse.metrics.MinCost`
      - The threshold is still open, and/or your business parameters are uncertain.
    * - :class:`~empulse.metrics.EmpiricalMaxProfit`
      - Both at once: the best cut-off, without averaging per-customer values away.

Using them as scikit-learn scorers
==================================

Any metric can be wrapped with :func:`~sklearn.metrics.make_scorer` and used anywhere scikit-learn
accepts a scorer:

.. code-block:: python

    from sklearn.metrics import make_scorer

    profit_scorer = make_scorer(
        expected_profit, response_method='predict_proba', greater_is_better=True
    )

Set ``greater_is_better`` to match the strategy — ``True`` for
:class:`~empulse.metrics.Profit`, ``False`` for :class:`~empulse.metrics.Cost`. Every metric
reports which it is as ``metric.direction``, so you need not hard-code it.

:doc:`06_pipeline` shows how to route the per-customer ``clv`` through cross-validation so the
scorer sees the right values for each fold.

Next
====

:doc:`04_train` stops merely measuring the profit and starts optimising it.
