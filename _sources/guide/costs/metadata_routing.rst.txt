.. _instance_based_cv:

======================================
Costs that differ per row
======================================

When every row carries its own cost — a customer's lifetime value, an applicant's credit line — a
simple train/test split is easy: pass the array to ``fit``, ``fit_resample`` or the metric and you
are done. Cross-validation is not, because each fold needs its own slice of that array, aligned
with the rows in the fold. Passing the full array to a folded estimator silently misaligns costs
and rows, and produces a plausible number that is wrong.

scikit-learn solves this with **metadata routing**: you declare which methods want which extra
arrays, and the cross-validator slices and delivers them.

:doc:`../../tutorial/06_pipeline` walks through the whole mechanism on a real dataset. This page is
the reference: what to declare, on what, and the Empulse-specific parts.

Enabling routing
================

Metadata routing is an experimental scikit-learn feature and is off by default. Turn it on globally:

.. code-block:: python

    from sklearn import set_config

    set_config(enable_metadata_routing=True)

or scope it to the block that needs it, so you do not change global state for the rest of the
process:

.. code-block:: python

    from sklearn import config_context

    with config_context(enable_metadata_routing=True):
        ...

Not every cross-validation utility supports routing; scikit-learn keeps a
`list of the ones that do <https://scikit-learn.org/stable/metadata_routing.html#metadata-routing-models>`_.

The vocabulary
==============

A full explanation is in scikit-learn's
`user guide <https://scikit-learn.org/stable/metadata_routing.html>`_, but the short version is
enough to use Empulse:

- A **requester** is anything with a ``fit``, ``fit_resample``, ``score`` or ``predict`` method that
  can ask for extra arrays. It asks by calling a ``set_*_request`` method.
- A **router** is something like :func:`~sklearn.model_selection.cross_val_score` or
  :class:`~sklearn.model_selection.GridSearchCV` that receives arrays at the top level, splits them
  per fold, and hands each requester what it asked for.

You declare requests once, then pass the arrays to the router's ``params`` argument.

What can be requested
=====================

The four cost names are always available:

.. code-block:: python

    from empulse.models import CSLogitClassifier

    model = CSLogitClassifier().set_fit_request(fp_cost=True, fn_cost=True)

**Business symbols are available too.** This is the part that is easy to miss. When a model is
built with a :class:`~empulse.metrics.Metric` as its ``loss``, Empulse extends the model's request
signature at construction time with every symbol and alias that the metric's cost matrix uses. A
metric parameterised by ``clv`` and ``accept_rate`` therefore yields a model that can request
exactly those:

.. code-block:: python

    from empulse.metrics import Cost, CostMatrix, Metric

    cost_matrix = (
        CostMatrix()
        .add_fn_cost('clv')
        .add_fp_cost('contact_cost')
        .set_default(contact_cost=1)
    )
    expected_cost = Metric(cost_matrix, Cost())

    model = CSLogitClassifier(loss=expected_cost).set_fit_request(clv=True)

This is what makes it possible to cross-validate a model whose costs are derived from business
parameters, rather than only one whose costs are four flat arrays.

The three kinds of requester
============================

**Models** request on ``fit``:

.. code-block:: python

    model = CSLogitClassifier(loss=expected_cost).set_fit_request(clv=True)

**Samplers** request on ``fit_resample``:

.. code-block:: python

    from empulse.samplers import CostSensitiveSampler

    sampler = CostSensitiveSampler(random_state=42).set_fit_resample_request(
        fp_cost=True, fn_cost=True
    )

.. note::
    A sampler inside a pipeline needs imbalanced-learn's
    :class:`~imblearn.pipeline.Pipeline`, not scikit-learn's — the latter has no sampler step and
    will not pass the parameters through.

**Scorers** request on ``score``. A :class:`~empulse.metrics.Metric` is itself callable, so it can
be wrapped directly with :func:`~sklearn.metrics.make_scorer`:

.. code-block:: python

    from sklearn.metrics import make_scorer
    from empulse.metrics import Savings

    savings = Metric(cost_matrix, Savings())

    scorer = make_scorer(
        savings,
        response_method='predict_proba',
        greater_is_better=True,
    ).set_score_request(clv=True)

Set ``greater_is_better`` to match the strategy's direction: ``True`` for
:class:`~empulse.metrics.Savings` and the :class:`~empulse.metrics.MaxProfit` family, ``False`` for
:class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.LogCost`. Every metric exposes this as
``metric.direction`` if you would rather not hard-code it.

Putting it together
===================

With the requests declared, hand the arrays to the router once and it does the slicing:

.. code-block:: python

    import numpy as np
    from sklearn import config_context
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=300, random_state=42)
    clv = np.random.default_rng(0).uniform(50, 500, size=len(y))

    with config_context(enable_metadata_routing=True):
        model = CSLogitClassifier(loss=expected_cost).set_fit_request(clv=True)
        scorer = make_scorer(
            savings, response_method='predict_proba', greater_is_better=True
        ).set_score_request(clv=True)

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        scores = cross_val_score(pipeline, X, y, scoring=scorer, params={'clv': clv}, cv=3)

    print(scores)

``clv`` now reaches both the model's ``fit`` and the scorer's ``score``, sliced to each fold.

Hyperparameter search works the same way:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    with config_context(enable_metadata_routing=True):
        model = CSLogitClassifier(loss=expected_cost).set_fit_request(clv=True)
        scorer = make_scorer(
            savings, response_method='predict_proba', greater_is_better=True
        ).set_score_request(clv=True)

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        search = GridSearchCV(
            pipeline,
            {'model__C': [0.1, 1.0]},
            scoring=scorer,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        )
        search.fit(X, y, clv=clv)

    print(search.best_params_)

Because the business parameters are ordinary routed metadata, they can be swept like any other
value — see :doc:`../../tutorial/06_pipeline` for why that is a sensitivity analysis rather than an
optimisation.

Aliased metrics
===============

If the cost matrix registers aliases, request the **alias**, not the underlying symbol — that is
the name the metric will look for:

.. code-block:: python

    aliased_matrix = (
        CostMatrix()
        .add_fn_cost('v')
        .add_fp_cost('c')
        .alias({'lifetime_value': 'v', 'contact_cost': 'c'})
        .set_default(contact_cost=1)
    )
    aliased_metric = Metric(aliased_matrix, Cost())

    with config_context(enable_metadata_routing=True):
        model = CSLogitClassifier(loss=aliased_metric).set_fit_request(lifetime_value=True)

Requesting both a symbol and one of its aliases raises a ``ValueError`` rather than letting one
silently win. Meta-estimators that wrap another estimator keep the two apart, so a parameter
requested by the outer metric does not leak into the inner estimator's ``fit``.

Costs at predict time
=====================

:class:`~empulse.models.CSThresholdClassifier` and :class:`~empulse.models.CSRateClassifier` are
unusual in that they can take costs at ``predict`` as well as at ``fit``. The threshold is then
recomputed from the new costs without refitting, which lets one fitted model serve several cost
scenarios:

.. code-block:: python

    from empulse.models import CSThresholdClassifier

    inner = CSLogitClassifier().set_fit_request(
        tp_cost=True, tn_cost=True, fp_cost=True, fn_cost=True
    )
    decider = CSThresholdClassifier(estimator=inner).fit(X, y, fp_cost=1, fn_cost=5)

    balanced = decider.predict(X)
    cautious = decider.predict(X, fp_cost=20, fn_cost=1)

Note that all four cost names are declared on the inner estimator, not just the two actually
passed. Requests are per-name, and a name that is neither requested nor explicitly refused raises
``UnsetMetadataPassedError`` as soon as a router tries to deliver it.

Inside a router this is ``set_predict_request``, mirroring the ``fit`` case. See
:ref:`threshold_tuning` for what the two classifiers do with the costs.

Parallelism
===========

Metric objects — including ones built from symbolic cost matrices and stochastic parameters — are
picklable, so ``n_jobs > 1`` on a pipeline, a grid search or an ensemble works normally. There is
no need to rebuild the metric inside each worker.

Turning routing off again
=========================

If you enabled routing globally rather than with :func:`~sklearn.config_context`, turn it off when
you are done. It is process-wide state, and leaving it on changes how unrelated estimators
interpret extra ``fit`` arguments:

.. code-block:: python

    set_config(enable_metadata_routing=False)

Where next
==========

- :doc:`../../tutorial/06_pipeline` — the same mechanism worked through on a real dataset.
- :ref:`specifying_costs` — the rules for costs outside a cross-validation loop.
- :ref:`cost_sampling` — the sampler side in more detail.
