.. _cslogit:
.. _proflogit:

==============================================
Linear Cost-Sensitive Models
==============================================

:class:`~empulse.models.CSLogitClassifier` [1]_ and :class:`~empulse.models.ProfLogitClassifier` [2]_
are two flavours of the same underlying model — a logistic regression classifier that
directly optimizes a cost-sensitive objective function during training.
They share every parameter and every feature; the **only difference is their default optimizer**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - :class:`~empulse.models.CSLogitClassifier`
     - :class:`~empulse.models.ProfLogitClassifier`
   * - Default optimizer
     - :class:`~empulse.optimizers.LBFGSBOptimizer` (L-BFGS-B)
     - :class:`~empulse.optimizers.GeneticAlgorithmOptimizer` (RGA)
   * - Default loss
     - Expected cost (:class:`~empulse.metrics.Cost` strategy)
     - Maximum profit (:class:`~empulse.metrics.MaxProfit` strategy)
   * - Best suited for
     - Smooth, differentiable objectives
     - Non-smooth or non-convex objectives


Quick Start
===========

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSLogitClassifier, ProfLogitClassifier
    from empulse.optimizers import GeneticAlgorithmOptimizer

    X, y = make_classification(n_samples=200, random_state=0)

    # CSLogit: minimize expected cost — fast, gradient-based
    cslogit = CSLogitClassifier(fp_cost=5, fn_cost=1)
    cslogit.fit(X, y)

    # ProfLogit: maximize expected profit — derivative-free GA
    proflogit = ProfLogitClassifier(
        tp_cost=300, fp_cost=10,
        optimizer=GeneticAlgorithmOptimizer(max_iter=50, population_size=10, random_state=42),
    )
    proflogit.fit(X, y)

    y_proba = cslogit.predict_proba(X)[:, 1]


Specifying costs
================

Both models accept costs the same two ways as every other cost-sensitive model in Empulse: as plain
``tp_cost``/``tn_cost``/``fp_cost``/``fn_cost`` values, scalar or per-sample, or as a
:class:`~empulse.metrics.Metric` passed as ``loss``. :ref:`specifying_costs` covers where each may
be set and how the two interact; :ref:`instance_based_cv` covers getting per-sample arrays through
cross-validation.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import CSLogitClassifier

    X, y = make_classification(n_samples=200, random_state=0)
    clv = np.random.default_rng(0).uniform(100, 1000, size=len(y))
    contact_cost = 10

    model = CSLogitClassifier(fn_cost=1)
    model.fit(X, y, tp_cost=clv - contact_cost, fp_cost=contact_cost)


Regularization
==============

Both models use elastic-net regularization, controlled by two parameters:

* ``C`` — inverse regularization strength (like sklearn's :class:`~sklearn:sklearn.linear_models.LogisticRegression`).
  Smaller values → stronger regularization.
* ``l1_ratio`` — mixing coefficient between L1 and L2 penalties.
  ``1.0`` (default) is pure L1; ``0.0`` is pure L2.

.. code-block:: python

    from empulse.models import CSLogitClassifier

    # Strong L2 regularization
    model = CSLogitClassifier(C=0.01, l1_ratio=0.0)

    # Elastic-net mix
    model = CSLogitClassifier(C=1.0, l1_ratio=0.5)

Soft thresholding
-----------------

Setting ``soft_threshold=True`` applies a proximal soft-threshold operator to the
coefficients at each gradient step, promoting sparsity without changing the
optimization landscape (useful when using gradient-based optimizers):

.. code-block:: python

    model = CSLogitClassifier(C=0.1, soft_threshold=True)


Custom Loss Functions
=====================

The default losses (:class:`~empulse.metrics.Cost` for :class:`~empulse.models.CSLogitClassifier` and
:class:`~empulse.metrics.MaxProfit` for :class:`~empulse.models.ProfLogitClassifier`) cover the most
common use cases, but any :class:`~empulse.metrics.Metric` from
:mod:`empulse.metrics` can be plugged in directly:

.. code-block:: python

    from empulse.metrics import Metric, Savings, CostMatrix
    from empulse.models import CSLogitClassifier

    # Optimize expected savings score instead of expected cost
    savings_metric = Metric(
        cost_matrix=CostMatrix().add_fp_cost('fp').add_fn_cost('fn'),
        strategy=Savings(),
    )
    model = CSLogitClassifier(loss=savings_metric)
    model.fit(X, y, fp=5, fn=1)

When a custom ``loss`` is provided, its symbolic parameters (e.g. ``fp``,
``fn``) are passed as keyword arguments to ``fit``.


Optimization
============

Both models accept an ``optimizer`` parameter that accepts any
:class:`~empulse.optimizers.Optimizer` instance.  When ``optimizer=None``
(the default), each model uses its built-in default.

.. code-block:: python

    from empulse.models import CSLogitClassifier
    from empulse.optimizers import Adam

    model = CSLogitClassifier(optimizer=Adam(lr=0.01, max_iter=200))

The table below summarises all available optimizers:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Class
     - Type
     - Best used when …
   * - :class:`~empulse.optimizers.LBFGSBOptimizer`
     - Quasi-Newton
     - Objective is smooth and differentiable; default for :class:`~empulse.models.CSLogitClassifier`
   * - :class:`~empulse.optimizers.ScipyOptimizer`
     - Wraps :func:`scipy.optimize.minimize`
     - You need a specific scipy method (CG, Newton-CG, TNC, …)
   * - :class:`~empulse.optimizers.SGD`
     - First-order gradient
     - Large datasets; controllable learning-rate schedules
   * - :class:`~empulse.optimizers.RMSProp`
     - Adaptive gradient
     - Noisy or non-stationary gradients
   * - :class:`~empulse.optimizers.Adam`
     - Adaptive moment estimation
     - General-purpose; usually the best first-order choice
   * - :class:`~empulse.optimizers.GeneticAlgorithmOptimizer`
     - Evolutionary
     - Non-smooth, non-convex objectives; default for :class:`~empulse.models.ProfLogitClassifier`
   * - :class:`~empulse.optimizers.MemeticOptimizer`
     - Evolutionary + gradient hybrid
     - When GA diversity and gradient precision are both needed

L-BFGS-B Optimizer
-------------------

The **default for** ``CSLogitClassifier``.  It is a limited-memory quasi-Newton
method, well-suited for smooth objectives and scales to thousands of features:

.. code-block:: python

    from empulse.models import CSLogitClassifier
    from empulse.optimizers import LBFGSBOptimizer

    model = CSLogitClassifier(
        optimizer=LBFGSBOptimizer(max_iter=200, tolerance=1e-5)
    )

Scipy Optimizer
---------------

:class:`~empulse.optimizers.ScipyOptimizer` is a thin wrapper around
:func:`scipy.optimize.minimize` that supports every method scipy provides:

.. code-block:: python

    from empulse.optimizers import ScipyOptimizer
    from empulse.models import CSLogitClassifier

    # BFGS (unbounded)
    model = CSLogitClassifier(optimizer=ScipyOptimizer(method='BFGS'))

    # Nelder–Mead (derivative-free — no gradient required)
    model = CSLogitClassifier(
        optimizer=ScipyOptimizer(method='Nelder-Mead', use_jacobian=False, max_iter=200)
    )

Gradient Optimizers (SGD, RMSProp, Adam)
-----------------------------------------

The three first-order gradient optimizers share a common set of features:
learning-rate schedules, alpha (smoothing) schedules, and mini-batch training.

Basic usage
~~~~~~~~~~~

.. code-block:: python

    from empulse.models import CSLogitClassifier
    from empulse.optimizers import SGD, RMSProp, Adam

    # Plain SGD with momentum
    model = CSLogitClassifier(optimizer=SGD(lr=0.05, momentum=0.9, max_iter=200))

    # RMSProp
    model = CSLogitClassifier(optimizer=RMSProp(lr=0.01, max_iter=200))

    # Adam
    model = CSLogitClassifier(optimizer=Adam(lr=0.001, max_iter=200))

Learning-rate schedules
~~~~~~~~~~~~~~~~~~~~~~~

Any :class:`~empulse.optimizers.Schedule` can be passed as ``lr_schedule``
to vary the learning rate over training:

.. code-block:: python

    from empulse.optimizers import Adam, CosineAnnealingSchedule, WarmupSchedule

    cosine = CosineAnnealingSchedule(max_value=1e-2, min_value=1e-5, t_max=500)
    schedule = WarmupSchedule(warmup_steps=50, after_schedule=cosine)

    model = CSLogitClassifier(
        optimizer=Adam(lr=1e-2, lr_schedule=schedule, max_iter=200)
    )

Available schedules:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Schedule
     - Description
   * - :class:`~empulse.optimizers.ConstantSchedule`
     - Always returns the same value (equivalent to a fixed LR)
   * - :class:`~empulse.optimizers.LinearSchedule`
     - Linear interpolation from ``start_value`` to ``end_value`` over ``n_steps`` epochs
   * - :class:`~empulse.optimizers.ExponentialSchedule`
     - ``value = start_value × gamma^epoch``, clipped at ``min_value`` and, optionally, ``max_value``
   * - :class:`~empulse.optimizers.StepSchedule`
     - Multiplies by ``gamma`` every ``step_size`` epochs
   * - :class:`~empulse.optimizers.CosineAnnealingSchedule`
     - Cosine annealing between ``max_value`` and ``min_value`` over ``t_max`` epochs; optionally cycles
   * - :class:`~empulse.optimizers.WarmupSchedule`
     - Linear warm-up for ``warmup_steps`` epochs, then delegates to another schedule

Alpha schedules
~~~~~~~~~~~~~~~

Some metrics (notably those using :class:`~empulse.metrics.MaxProfit`) use a
smoothing parameter *alpha* to soften a piecewise-constant gradient during
training.  Starting with a low alpha (smooth landscape) and increasing it as
training progresses is a form of curriculum learning:

.. code-block:: python

    from empulse.optimizers import Adam, LinearSchedule, CosineAnnealingSchedule

    # Linearly grow alpha from 0.5 to 20 over the first 200 epochs
    alpha_schedule = LinearSchedule(start_value=0.5, end_value=20.0, n_steps=200)
    lr_schedule = CosineAnnealingSchedule(max_value=1e-2, min_value=1e-5, t_max=500)

    model = ProfLogitClassifier(
        optimizer=Adam(
            lr=1e-2,
            lr_schedule=lr_schedule,
            alpha_schedule=alpha_schedule,
            max_iter=1000,
        )
    )

The ``alpha_schedule`` is silently ignored on objectives that do not expose a
``set_alpha`` method, so it is always safe to set.

:class:`~empulse.metrics.MaxProfit` itself only takes a single, constant ``alpha`` - it does not
anneal on its own. To reproduce a growing-then-capped temperature (e.g. starting at ``1.0`` and
annealing up to a ceiling of ``100.0`` at a rate of ``1.1`` per epoch), use an ``alpha_schedule``
with a ``max_value``:

.. code-block:: python

    from empulse.optimizers import Adam, ExponentialSchedule

    alpha_schedule = ExponentialSchedule(start_value=1.0, gamma=1.1, max_value=100.0)
    model = ProfLogitClassifier(optimizer=Adam(lr=1e-2, alpha_schedule=alpha_schedule))

Mini-batch training
~~~~~~~~~~~~~~~~~~~

Set ``batch_size`` to process a random subset of samples per gradient step,
which can speed up training on large datasets and acts as a regularizer:

.. code-block:: python

    from empulse.optimizers import Adam

    model = CSLogitClassifier(
        optimizer=Adam(lr=0.005, max_iter=2000, batch_size=256, random_state=42)
    )

Pass ``random_state`` as an integer for reproducibility, or as a
``numpy.random.Generator`` instance if you need fine-grained control.

Genetic Algorithm Optimizer
----------------------------

The **default for** ``ProfLogitClassifier``.  It is a Real-coded Genetic
Algorithm (RGA) that does not require gradients, making it suitable for
non-smooth objectives like the Expected Maximum Profit metric:

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from empulse.optimizers import GeneticAlgorithmOptimizer

    rga = GeneticAlgorithmOptimizer(
        max_iter=50,
        patience=10,
        bounds=(-10, 10),
        population_size=10,
        random_state=42,
    )
    model = ProfLogitClassifier(optimizer=rga)
    model.fit(X, y, tp_cost=clv - contact_cost, fp_cost=contact_cost)

Memetic Optimizer
-----------------

:class:`~empulse.optimizers.MemeticOptimizer` combines the population diversity
of a genetic algorithm with a per-individual gradient local search
(Lamarckian learning).  It is more expensive per generation but typically
converges in far fewer generations than a plain RGA:

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from empulse.optimizers import MemeticOptimizer

    model = ProfLogitClassifier(
        optimizer=MemeticOptimizer(
            max_iter=10,
            population_size=10,
            local_steps=2,
            lr=0.05,
        )
    )


sklearn Integration
===================

Both models are ordinary scikit-learn estimators and drop into
:class:`~sklearn:sklearn.pipeline.Pipeline`,
:func:`~sklearn:sklearn.model_selection.cross_val_score` and
:class:`~sklearn:sklearn.model_selection.GridSearchCV` unchanged. Per-sample costs reach each fold
through metadata routing — see :ref:`instance_based_cv`.

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CSLogitClassifier(fp_cost=5, fn_cost=1)),
    ])

    grid_search = GridSearchCV(pipeline, {'model__C': np.logspace(-2, 1, 4)}, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_params_['model__C'])


Inspecting the Optimization Result
===================================

After fitting, the full result of the optimization is stored in ``result_``:

.. code-block:: python

    from empulse.models import CSLogitClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, random_state=0)
    model = CSLogitClassifier(fp_cost=5).fit(X, y)

    print(model.result_.success)   # True / False
    print(model.result_.message)   # Human-readable status
    print(model.result_.nit)       # Number of iterations
    print(model.coef_)             # Fitted coefficients
    print(model.intercept_)        # Fitted intercept

If the optimizer did not converge, ``result_.success`` is ``False`` and the
message will explain why.  Increasing ``max_iter`` or scaling the input
features (e.g. with :class:`sklearn.preprocessing.StandardScaler`) usually
resolves convergence issues for gradient-based optimizers.


References
==========

.. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       *European Journal of Operational Research*, 297(1), 291–300.

.. [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
       Snoeck, M. (2017). Profit Maximizing Logistic Model for
       Customer Churn Prediction Using Genetic Algorithms.
       *Swarm and Evolutionary Computation*.

