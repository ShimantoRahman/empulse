.. _specifying_costs:

=============================
Handing costs to an estimator
=============================

Every cost-sensitive estimator in Empulse accepts costs in one of two forms. The rules described
here are implemented once, on the shared base class, so they apply identically to
:class:`~empulse.models.CSLogitClassifier`, :class:`~empulse.models.CSBoostClassifier`, the tree and
ensemble models, the threshold meta-estimators and :class:`~empulse.models.RobustCSClassifier`.

The two forms
=============

.. tab-set::

    .. tab-item:: Plain costs
        :sync: plain

        The quickest route when your costs really are four numbers, or four arrays.

        .. code-block:: python

            from sklearn.datasets import make_classification
            from empulse.models import CSBoostClassifier

            X, y = make_classification(n_samples=200, random_state=42)

            model = CSBoostClassifier()
            model.fit(X, y, fp_cost=5, fn_cost=100)

    .. tab-item:: A metric
        :sync: metric

        The route when the costs are computed from business parameters, when you want the same
        definition used for scoring and training, or when a parameter varies per instance.

        .. code-block:: python

            from empulse.metrics import Cost, CostMatrix, Metric

            cost_matrix = (
                CostMatrix()
                .add_fp_cost('discount')
                .add_fn_cost('lost_value')
                .set_default(discount=5, lost_value=100)
            )
            expected_cost = Metric(cost_matrix, Cost())

            model = CSBoostClassifier(loss=expected_cost)
            model.fit(X, y)

The ``loss`` parameter accepts any :class:`~empulse.metrics.BaseMetric`, which means a
:class:`~empulse.metrics.Metric`, a :class:`~empulse.metrics.MixtureMetric`, or one of the prebuilt
metrics such as :func:`~empulse.metrics.empc_score`.

Prefer the metric route when a parameter varies per row, when you want to tune a business parameter
by cross-validation, when the cost formula is more than one number per outcome, or simply when you
want the number you train on and the number you report to be the same object.

.. note::
    Not every model supports every strategy as a training objective. The compatibility table lives
    in :ref:`metric_class_in_model`.

Where costs can be set
======================

Plain costs can be supplied at construction, at ``fit``, or both:

.. code-block:: python

    from empulse.models import CSLogitClassifier

    # at construction
    CSLogitClassifier(fp_cost=5, fn_cost=100).fit(X, y)

    # at fit
    CSLogitClassifier().fit(X, y, fp_cost=5, fn_cost=100)

    # both: fit wins for the terms it names, the constructor supplies the rest
    CSLogitClassifier(fp_cost=5, fn_cost=100).fit(X, y, fn_cost=1)

The cost arguments of ``fit`` default to a sentinel, ``Parameter.UNCHANGED``, rather than to zero —
which is why the API reference renders their default as ``$UNCHANGED$``. It means "leave whatever
the constructor set", and it is what makes the third form above behave as it does. Passing ``0``
explicitly is a real value and does override the constructor.

Prefer the constructor for costs that are part of the model's configuration and the same across
folds, so that :func:`~sklearn.base.clone` carries them. Prefer ``fit`` for anything
instance-dependent: cloning does not carry per-sample arrays, and cross-validation needs them
sliced per fold, which is what :ref:`instance_based_cv` is for.

.. warning::
    **Constructor costs are never forwarded to a metric** ``loss``. They default to ``0.0``, so
    forwarding them would silently zero out any cost-matrix term of the same name. A metric's
    parameters must be passed to ``fit``.

So this is correct:

.. code-block:: python

    matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')
    model = CSBoostClassifier(loss=Metric(matrix, Cost()))

    model.fit(X, y, fp_cost=5, fn_cost=100)

while setting the same values on the constructor instead raises ``Metric expected a value for
fn_cost, did not receive it``.

If every cost ends up zero
--------------------------

Fitting with no costs at all is almost never intended, so Empulse does not silently train a
cost-blind model. It emits a ``UserWarning`` and substitutes ``fp_cost = fn_cost = 1``, which
reduces to ordinary misclassification error:

.. code-block:: python

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        CSLogitClassifier().fit(X, y)

Set the costs explicitly to make the warning go away.

Cost names inside a cost matrix
===============================

A cost matrix may legitimately name one of its own symbols ``tp_cost``, ``tn_cost``, ``fp_cost`` or
``fn_cost`` — three of the bundled datasets do, because their costs were precomputed rather than
derived from business parameters. Those names collide with ``fit``'s dedicated keyword arguments.

Empulse resolves the collision in favour of the metric: when a ``loss`` is set, a cost argument
whose name matches one of the metric's symbols is routed to the metric.

.. code-block:: python

    matrix = CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')
    model = CSBoostClassifier(loss=Metric(matrix, Cost()))
    model.fit(X, y, fp_cost=5.0, fn_cost=100.0)   # both reach the metric

A cost argument that the metric does *not* use is a mistake — usually a leftover from the plain-cost
route — and raises a warning naming the parameters the metric actually expects, rather than being
dropped silently.

Passing arguments to a wrapped estimator
========================================

:class:`~empulse.models.CSBoostClassifier` and :class:`~empulse.models.B2BoostClassifier` wrap a
third-party booster, so ``fit`` has to serve two audiences: the metric, and the booster underneath.
Anything not recognised as a cost or a metric parameter would be ambiguous, so booster arguments go
in a dedicated ``fit_params`` dict:

.. code-block:: python

    model = CSBoostClassifier()
    model.fit(X, y, fp_cost=5, fn_cost=100, fit_params={'verbose': False})

``sample_weight`` is the one exception: it is recognised and forwarded into the booster's own
``fit``.

Which models honour per-row costs
=================================

Supplying an array does not guarantee the model uses it per row. Models whose objective is defined
over the population rather than the individual — anything using the
:class:`~empulse.metrics.MaxProfit` family, and the minimax models in :doc:`../training/minimax_models`
— reduce array costs to a single class-dependent value before fitting. This is not an
approximation to work around, it is what those measures mean; but it does mean an array and its
mean produce the same model.

:class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.LogCost` and
:class:`~empulse.metrics.Savings` use per-row costs directly. :ref:`choosing_metric` has the full
table.

Where next
==========

- :ref:`instance_based_cv` — getting per-row costs through pipelines and cross-validation.
- :ref:`choosing_metric` — which strategy to pair with your cost matrix.
- :doc:`../training` — the models themselves.
