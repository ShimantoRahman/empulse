.. _metric_objects:

==============================
Working with metric objects
==============================

Pairing a :class:`~empulse.metrics.CostMatrix` with a :class:`~empulse.metrics.MetricStrategy`
produces a :class:`~empulse.metrics.Metric`. It behaves like an ordinary scoring function, but it
carries the cost matrix with it, which lets it do several things a plain function cannot: derive an
operating point, serve as a training objective, and tell you what it needs.

.. code-block:: python

    import numpy as np
    from empulse.metrics import Cost, CostMatrix, Metric

    matrix = (
        CostMatrix()
        .add_fp_cost('c_fp')
        .add_fn_cost('c_fn')
        .set_default(c_fp=1.0, c_fn=5.0)
    )
    expected_cost = Metric(matrix, Cost())

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9])

    print(expected_cost(y_true, y_score))

Supplying parameters
====================

Any symbol in the cost matrix becomes a keyword argument. Empulse resolves each one in order:
an explicit keyword argument wins, otherwise the value from
:meth:`~empulse.metrics.CostMatrix.set_default` is used, and if neither exists the call fails
naming the missing parameter.

A symbol can be supplied under its own name or under any alias registered for it — but not both.
Passing both raises a ``ValueError`` rather than letting whichever came last silently win.

Scalars apply to every row; arrays supply one value per row, and must match the length of
``y_true``. A mismatch raises a ``ValueError`` naming the parameter, rather than surfacing later as
an unrelated broadcasting error.

A keyword the metric does not recognise raises a warning and is ignored, so a typo in a parameter
name does not silently fall back to a default.

Finding the operating point
===========================

Because the metric knows the cost matrix, it can say where the break-even point between acting and
not acting lies.

.. code-block:: python

    print(expected_cost.optimal_threshold(y_true, y_score))
    print(expected_cost.optimal_rate(y_true, y_score))

``optimal_threshold`` returns a score cut-off; ``optimal_rate`` returns the fraction of the
population to act on. They are two views of the same decision, and
:func:`~empulse.metrics.classification_threshold` converts between them.
:ref:`threshold_tuning` covers when each is the more useful form.

With instance-dependent costs, ``optimal_threshold`` returns an **array**: each row has its own
break-even point, because each row has its own costs.

.. code-block:: python

    per_row_costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    print(expected_cost.optimal_threshold(y_true, y_score, c_fn=per_row_costs))

Using a metric as a scikit-learn scorer
=======================================

A metric is callable with the ``(y_true, y_score, **kwargs)`` signature that
:func:`~sklearn.metrics.make_scorer` expects, so it wraps directly:

.. code-block:: python

    from sklearn.metrics import make_scorer

    scorer = make_scorer(
        expected_cost,
        response_method='predict_proba',
        greater_is_better=False,
    )

Set ``greater_is_better`` to match the strategy: ``False`` for :class:`~empulse.metrics.Cost` and
:class:`~empulse.metrics.LogCost`, ``True`` for the rest. Rather than hard-coding it, read it off
the metric:

.. code-block:: python

    greater_is_better = expected_cost.direction.name == 'MAXIMIZE'
    print(greater_is_better)

Every metric also has a settable ``__name__``, which is what appears in
:class:`~sklearn.model_selection.GridSearchCV` results and scorer error messages:

.. code-block:: python

    expected_cost.__name__ = 'campaign_cost'
    print(expected_cost.__name__)

Metrics are picklable, including ones built from symbolic and stochastic cost matrices, so
``n_jobs > 1`` works without rebuilding them per worker.

The prebuilt metrics are metric objects
=======================================

:func:`~empulse.metrics.empc_score`, :func:`~empulse.metrics.mpc_score`,
:func:`~empulse.metrics.expected_cost_loss` and the rest are not functions — they are
:class:`~empulse.metrics.Metric` (or :class:`~empulse.metrics.MixtureMetric`) instances built from
the standard cost matrix of their domain. Everything on this page therefore applies to them too:

.. code-block:: python

    from empulse.metrics import empc_score

    print(empc_score(y_true, y_score, clv=200))
    print(empc_score.optimal_rate(y_true, y_score, clv=200))

They can also be passed straight to a model as ``loss``, so the metric you report is the metric you
train on.

Mixtures of metrics
===================

Some domain measures assume a parameter whose distribution mixes point masses with a continuous
piece — the recovery fraction on a defaulted loan is 0 with some probability, 1 with some
probability, and spread over the interval otherwise. :mod:`sympy.stats` cannot express that as a
single random variable.

:class:`~empulse.metrics.MixtureMetric` sidesteps it. Each
:class:`~empulse.metrics.MixtureComponent` pairs a weight with an ordinary metric and any parameter
values fixed for that piece, and the mixture is their weighted sum.
:ref:`user_defined_value_metric` has a worked example.

This is **exact, not an approximation**. The score, its gradients, and the optimal predicted-positive
rate are all linear functionals of the assumed density, and integration and differentiation are
linear, so a mixture's value is exactly the weighted sum of its components' values.

``optimal_threshold`` is the one exception, because a threshold is a non-linear function of a rate:
combining the components' thresholds would be wrong. :class:`~empulse.metrics.MixtureMetric`
therefore combines the *rates* first and converts the result to a threshold once.

A mixture behaves like a plain metric everywhere else. It satisfies the same
:class:`~empulse.metrics.BaseMetric` interface, so it can be a model's ``loss``, and its
``direction`` and ``strategy`` are read from its components — which must agree, or accessing them
raises.

Where next
==========

- :ref:`choosing_metric` — which strategy to build the metric with.
- :ref:`threshold_tuning` — turning ``optimal_rate`` into a deployed decision rule.
- :ref:`instance_based_cv` — scoring with a metric inside cross-validation.
