.. _choosing_metric:

=========================
Choosing a strategy
=========================

A cost matrix says what the outcomes are worth. It does not say what number to report. A
**strategy** decides that, and the same matrix paired with different strategies answers genuinely
different questions: what will this model cost me, how much better is it than doing nothing, and
how much could it earn at the best possible cut-off.

Empulse ships six. This page covers what each one computes, what it needs from you, and which
models can train on it.

.. code-block:: python

    from empulse.metrics import Cost, CostMatrix, Metric

    matrix = (
        CostMatrix()
        .add_fp_cost('c_fp')
        .add_fn_cost('c_fn')
        .set_default(c_fp=1.0, c_fn=5.0)
    )

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    print(Metric(matrix, Cost())(y_true, y_score))

At a glance
===========

.. list-table::
    :widths: 20 12 20 16 16 16
    :header-rows: 1

    * - Strategy
      - Direction
      - ``y_score`` must be
      - Per-row costs
      - Uncertain parameters
      - Trainable
    * - :class:`~empulse.metrics.Cost`
      - lower is better
      - calibrated probabilities
      - used per row
      - replaced by the mean
      - yes
    * - :class:`~empulse.metrics.LogCost`
      - lower is better
      - calibrated probabilities
      - used per row
      - replaced by the mean
      - yes
    * - :class:`~empulse.metrics.Savings`
      - higher is better
      - calibrated probabilities
      - used per row
      - replaced by the mean
      - yes
    * - :class:`~empulse.metrics.MaxProfit`
      - higher is better
      - any ranking score
      - averaged first
      - integrated over
      - yes
    * - :class:`~empulse.metrics.EmpiricalMaxProfit`
      - higher is better
      - any ranking score
      - used per row
      - replaced by the mean
      - tree and evolutionary models only
    * - :class:`~empulse.metrics.AUEPC`
      - higher is better
      - any ranking score
      - used per row
      - replaced by the mean
      - tree and evolutionary models only

.. themed-figure:: three_strategies
    :alt: The same predictions scored three ways: an expected cost in currency, a savings ratio on
        a nought-to-one gauge, and the peak of a profit curve.

    One set of predictions, three questions. Only the question changes between the panels.

.. warning::
    Nothing checks the "``y_score`` must be" column at runtime. Passing an uncalibrated
    ``predict_proba`` to :class:`~empulse.metrics.Cost` does not raise; it returns a number that is
    quietly wrong. :ref:`calibration` covers how much this matters and what to do about it.

Cost
====

The cost loss answers the most direct question: what would this classifier cost if it were
deployed? It sums the cost of each prediction under the cost matrix.

.. math::

    \text{Cost}(X, \theta) = \sum_{i=1}^{n} \text{C}_i(\hat{y}_i(X_i, \theta)|y_i)

where :math:`\hat{y}_i`, :math:`y_i` and :math:`X_i` are the predicted class, true class and feature
vector of the :math:`i`-th instance.

:class:`~empulse.metrics.Cost` computes the *expected* version, weighting each outcome by the
predicted class probability rather than thresholding first:

.. math::

    \mathbb{E}(\text{Cost}(X, \theta)) = \sum_{i=1}^n \big[ \text{P}(y_i = 1 | X_i, \theta) \cdot \text{C}_i(1|y_i) + \text{P}(y_i = 0 | X_i, \theta) \cdot \text{C}_i(0|y_i) \big]

This punishes a model for being unconfident, and it is differentiable, which is what makes it
usable as a training objective. Empulse returns the **mean** per instance, so the number is
directly readable as "cost per customer".

Reach for it when you want an interpretable number in currency, and when your model's probabilities
mean what they say.

The hard-label variant, which thresholds scores first, is available as
:func:`~empulse.metrics.cost_loss`. It is a plain function rather than a strategy, because there is
nothing differentiable to train on.

LogCost
=======

:class:`~empulse.metrics.LogCost` is the same idea through a logarithm: it weights the cost of each
outcome by the log of the predicted probability rather than the probability itself.

The effect is a much steeper penalty for confident mistakes. Where :class:`~empulse.metrics.Cost`
charges a fixed amount for a wrong prediction regardless of how sure the model was,
:class:`~empulse.metrics.LogCost` charges more the more certain the model was about the wrong class.
It is the cost-weighted generalisation of cross-entropy, and reduces exactly to log loss when
``tp_cost = tn_cost = -1`` and ``fp_cost = fn_cost = 0``.

.. code-block:: python

    from empulse.metrics import LogCost

    print(Metric(matrix, LogCost())(y_true, y_score))

Use it when you care about the quality of the probabilities themselves, not only the decisions they
lead to — for instance when the scores feed a downstream system that will threshold them
differently.

Savings
=======

A cost is hard to judge in isolation: is 7.20 per customer good? :class:`~empulse.metrics.Savings`
answers that by expressing the cost relative to a baseline, the way :math:`R^2` expresses error
relative to predicting the mean.

.. math::

    \text{Savings}(X, \theta) = 1 - \frac{\text{Cost}(X, \theta)}{\text{Cost}(X,\theta^\prime)}

with :math:`\theta^\prime` the parameters of the baseline. One is a perfect model, zero is no better
than the baseline, and negative is worse than doing the obvious thing. Because it is a ratio, it is
comparable across datasets and across cost scales in a way a raw cost is not.

The default baseline is the naive model: predict all ones or all zeros, whichever is cheaper.

.. math::
    \text{Cost}_{\text{naive}}(X) = \min\left(\sum_{i=1}^n \text{C}_i(0 | y_i), \sum_{i=1}^n \text{C}_i(1 | y_i)\right)

Choosing a baseline
-------------------

The baseline is a parameter, not a fixed choice, and it changes what the score means:

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - ``baseline``
      - The model being compared against
    * - ``'zero_one'``
      - All-zeros or all-ones, whichever is cheaper. The default.
    * - ``'zero'``
      - Always predict the negative class — "do nothing".
    * - ``'one'``
      - Always predict the positive class — "treat everyone".
    * - ``'prior'``
      - Predict the class prior probability for every instance.
    * - an array
      - Your own baseline scores, one per instance — for example a model already in production.

.. code-block:: python

    from empulse.metrics import Savings

    savings = Metric(matrix, Savings())

    print(savings(y_true, y_score))
    print(savings(y_true, y_score, baseline='zero'))
    print(savings(y_true, y_score, baseline='prior'))

``'zero'`` and ``'one'`` are worth reaching for when the naive baseline is not the alternative you
actually face: if the campaign either runs for everyone or not at all, compare against that.

.. note::
    ``'prior'`` and array baselines are available on the :class:`~empulse.metrics.Savings` strategy
    only. The standalone :func:`~empulse.metrics.savings_score` function accepts ``'zero_one'``,
    ``'zero'``, ``'one'`` and arrays, but not ``'prior'``.

MaxProfit
=========

The three strategies above evaluate a model at the operating point implied by its scores.
:class:`~empulse.metrics.MaxProfit` asks a different question: if the threshold is not yet fixed,
what is the most this model could earn at the best possible cut-off?

Profit as a function of the threshold :math:`t` is

.. math::

    \text{Profit}(t) = b_0 \pi_0 F_0(t) + b_1 \pi_1 (1 - F_1(t)) - c_0 \pi_0 (1 - F_0(t)) - c_1 F_1(t)

where :math:`F_0(t)` and :math:`F_1(t)` are the cumulative distributions of the scores in each
class and :math:`\pi_0`, :math:`\pi_1` the class priors. The measure is the maximum over all
thresholds:

.. math::

    \text{MP} = \max_{\forall t} \text{Profit}(t) = \text{Profit}(T)

.. note::
    In the value-driven literature the positive class is written as class 0, so :math:`\pi_0` above
    is the prior of the positive class. Empulse's ``tp``/``fp``/``fn``/``tn`` naming follows the
    usual machine-learning convention instead; the translation is handled internally.

Because it works from the ranking rather than from calibrated probabilities,
:class:`~empulse.metrics.MaxProfit` is the strategy to use when your model produces scores you
trust the *order* of but not the *values* of. It is also the only strategy that yields an operating
point directly, through ``optimal_rate`` and ``optimal_threshold`` — see :ref:`threshold_tuning`.

Uncertain parameters
--------------------

Its second distinguishing feature is that it integrates over uncertainty rather than collapsing it.
When a cost-matrix parameter is a :mod:`sympy.stats` random variable, the measure becomes an
expectation over that parameter's distribution:

.. math::

    \mathbb{E}(\text{MP}) = \int_{b_0} \int_{c_0} \int_{b_1} \int_{c_1} \text{Profit}(T;b_0, c_0, b_1, c_1) \cdot w(b_0, c_0, b_1, c_1) \, db_0 dc_0 db_1 dc_1

with :math:`w` the joint density of the cost-benefit distribution. This is what separates the
"expected maximum profit" family of measures from the plain maximum profit ones. In practice only
one parameter is usually treated as uncertain.

How the integral is computed
----------------------------

``integration_method`` controls that, and the default ``'auto'`` picks a ladder from exact to
approximate:

.. list-table::
    :widths: 24 76
    :header-rows: 1

    * - ``integration_method``
      - Behaviour
    * - ``'auto'``
      - One random variable with a polynomial profit function: exact closed form. Two: numerical
        quadrature. More: quasi-Monte Carlo.
    * - ``'quad'``
      - Numerical quadrature via :mod:`scipy.integrate`.
    * - ``'monte-carlo'``
      - Plain Monte Carlo sampling.
    * - ``'quasi-monte-carlo'``
      - Low-discrepancy sampling; converges faster than plain Monte Carlo for the same budget.

The exact path covers the Uniform, Beta, Normal, Log-Normal, Gamma, Pareto, Triangular,
Exponential, Chi-squared and Weibull distributions. It is both faster and free of sampling noise,
so leaving ``integration_method='auto'`` is almost always right; override it only to check a result
or when a distribution falls outside the supported set.

``n_mc_samples_exp`` sets the sampling budget as :math:`2^{n}` (default 16, so 65,536 samples), and
``random_state`` makes the sampled methods reproducible.

.. code-block:: python

    import sympy, sympy.stats
    from empulse.metrics import MaxProfit

    gamma = sympy.stats.Beta('gamma', 6, 14)
    clv = sympy.symbols('clv')

    stochastic_matrix = CostMatrix().add_tp_benefit(gamma * clv).add_fp_cost(10)

    exact = Metric(stochastic_matrix, MaxProfit())
    sampled = Metric(
        stochastic_matrix,
        MaxProfit(integration_method='quasi-monte-carlo', n_mc_samples_exp=14, random_state=0),
    )

    print(exact(y_true, y_score, clv=200))
    print(sampled(y_true, y_score, clv=200))

.. warning::
    :class:`~empulse.metrics.MaxProfit` is a **population-level** measure: it is defined over the
    score distributions of the two classes, not over individual rows. Array-valued cost parameters
    are therefore reduced to their mean before the measure is computed. This is mathematically
    equivalent to averaging the measure over instances, but it does mean an array and its mean give
    the same answer. If per-row costs must genuinely drive the result, use
    :class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.Savings` or
    :class:`~empulse.metrics.EmpiricalMaxProfit`.

EmpiricalMaxProfit
==================

:class:`~empulse.metrics.EmpiricalMaxProfit` answers the same question as
:class:`~empulse.metrics.MaxProfit`, but from the data rather than from a model of it. Instead of
working with the theoretical score distributions, it ranks the samples by score, walks the ROC
convex hull, and reports the profit at the best point actually achievable on this dataset.

.. code-block:: python

    from empulse.metrics import EmpiricalMaxProfit

    print(Metric(matrix, EmpiricalMaxProfit())(y_true, y_score))

The practical difference is that it honours **per-row costs**. Where
:class:`~empulse.metrics.MaxProfit` must average a per-customer lifetime value away,
:class:`~empulse.metrics.EmpiricalMaxProfit` accumulates each customer's own value as the ranking
descends. That makes it the right choice when the whole point is that customers differ — which is
why :func:`~empulse.metrics.empb_score` is built on it.

The cost is that it is an empirical quantity: it can be optimistic on small samples, since it picks
the best cut-off on the same data it is measured on.

AUEPC
=====

Maximum profit measures report the peak of the profit curve. :class:`~empulse.metrics.AUEPC`
reports the **area under it**: not "how much can I make at the best cut-off?" but "how good is this
ranking across all the cut-offs I might end up using?".

.. code-block:: python

    from empulse.metrics import AUEPC

    print(Metric(matrix, AUEPC())(y_true, y_score))

With ``normalize=True`` (the default) the curve is divided by that of a perfect ranking, giving a
0–1 score in the spirit of ROC AUC but weighted by money rather than counting every swap equally.
The curve is truncated where even the oracle's cumulative profit turns negative, since targeting
beyond that point is never worthwhile.

Use it to compare rankings when the operating point is genuinely unknown or expected to move, and a
single peak would be a misleading summary.

Three metrics, one curve
------------------------

:class:`~empulse.metrics.MaxProfit`, :class:`~empulse.metrics.EmpiricalMaxProfit` and
:class:`~empulse.metrics.AUEPC` all read the same profit curve. They differ in what they take from
it.

.. themed-figure:: profit_curve
    :alt: A profit curve against the fraction of the population targeted, with its peak marked,
        the area beneath it shaded, the convex hull drawn above it, and a perfect ranking for
        comparison.

    The peak is the maximum profit, the shaded area is the AUEPC, and the convex hull is the
    frontier the empirical measure walks along.

Which models can train on which
===============================

A metric can always be *evaluated*. Training on one is a stronger requirement: the model has to be
able to optimise it. Gradient-based models need an objective with usable derivatives, which the two
ranking-based strategies do not provide; evolutionary and tree-based models only need a scalar
fitness, so they accept anything.

.. _metric_class_in_model:

.. list-table:: Models by supported strategy
    :widths: 26 12 12 12 12 14 12
    :header-rows: 1

    * - Model
      - :class:`~empulse.metrics.Cost`
      - :class:`~empulse.metrics.LogCost`
      - :class:`~empulse.metrics.Savings`
      - :class:`~empulse.metrics.MaxProfit`
      - :class:`~empulse.metrics.EmpiricalMaxProfit`
      - :class:`~empulse.metrics.AUEPC`
    * - :class:`~empulse.models.CSLogitClassifier`
      - ✅
      - ✅
      - ✅
      - ⚠️
      - ❌
      - ❌
    * - :class:`~empulse.models.CSBoostClassifier`
      - ✅
      - ✅
      - ✅
      - ⚠️
      - ❌
      - ❌
    * - :class:`~empulse.models.CSTreeClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
    * - :class:`~empulse.models.CSForestClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
    * - :class:`~empulse.models.CSBaggingClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
    * - :class:`~empulse.models.CSThresholdClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ❌
    * - :class:`~empulse.models.CSRateClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ❌
    * - :class:`~empulse.models.ProfLogitClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ❌
      - ❌
    * - :class:`~empulse.models.ProfTreeClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
    * - :class:`~empulse.models.ProfSRClassifier`
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
      - ✅
    * - :class:`~empulse.models.ProfMPMClassifier`
      - ❌
      - ❌
      - ❌
      - ✅
      - ❌
      - ❌
    * - :class:`~empulse.models.ProfMEMPMClassifier`
      - ❌
      - ❌
      - ❌
      - ✅
      - ❌
      - ❌

⚠️ marks :class:`~empulse.metrics.MaxProfit` on the two gradient-based models. It works, by
approximating the piecewise-constant true/false positive rates with a smooth sigmoid, but it is
**experimental** and not recommended for production. The sigmoid's temperature is the ``alpha``
parameter, and annealing it during training is what ``alpha_schedule`` on the gradient optimizers
does — see :ref:`cslogit`.

The two minimax models are the mirror image: they optimise a worst-case expected profit and accept
:class:`~empulse.metrics.MaxProfit` only. Anything else raises a ``ValueError`` naming the
restriction. :class:`~empulse.models.RobustCSClassifier` and
:class:`~empulse.models.B2BoostClassifier` inherit the row of the model they wrap or subclass.

A note on training cost
-----------------------

:class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.Savings` produce a **static** objective:
the gradient and Hessian depend only on the labels and the cost matrix, so they are computed once
and reused every boosting round. :class:`~empulse.metrics.MaxProfit` and
:class:`~empulse.metrics.LogCost` produce a **dynamic** one, re-derived each round because the
optimal threshold (or the log weighting) moves as the model changes. Expect the dynamic strategies
to train noticeably slower.

Choosing between them
=====================

.. list-table::
    :widths: 30 70
    :header-rows: 1

    * - If you want...
      - Use
    * - A number in currency you can put in a business case
      - :class:`~empulse.metrics.Cost`
    * - To compare models across datasets or cost scales
      - :class:`~empulse.metrics.Savings`
    * - Well-calibrated probabilities, not only good decisions
      - :class:`~empulse.metrics.LogCost`
    * - The best achievable profit when the threshold is not fixed
      - :class:`~empulse.metrics.MaxProfit`
    * - The same, but with costs that genuinely differ per row
      - :class:`~empulse.metrics.EmpiricalMaxProfit`
    * - To compare rankings when the operating point may move
      - :class:`~empulse.metrics.AUEPC`

Where next
==========

- :ref:`metric_objects` — what the resulting metric object can do.
- :ref:`calibration` — what your scores have to mean for these numbers to be trustworthy.
- :ref:`user_defined_value_metric` — worked cost matrices to pair with these strategies.
- :ref:`prebuilt_churn_metrics` and its siblings — strategies already paired with a domain matrix.
