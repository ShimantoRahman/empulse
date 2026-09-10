.. _profmpm:
.. _profmempm:
.. _profsr:

===========================
Minimax and symbolic models
===========================

Three models sit apart from the rest of :doc:`../training`. They optimise profit, but neither by
descending a gradient nor by growing a tree, and each answers a question the others cannot.

- :class:`~empulse.models.ProfMPMClassifier` and :class:`~empulse.models.ProfMEMPMClassifier`
  maximise the profit you are guaranteed **in the worst case**, using nothing but the mean and
  covariance of each class. No distributional assumption is made at all.
- :class:`~empulse.models.ProfSRClassifier` searches over mathematical *expressions* rather than
  coefficients, producing a decision function you can read as a formula.

All three default to the :class:`~empulse.metrics.MaxProfit` strategy, and the two minimax models
accept nothing else.

Profit-driven minimax probability machines
==========================================

An ordinary classifier asks what happens on average. A minimax probability machine asks what
happens in the worst case: across *every* distribution with the observed class means and
covariances, what is the largest expected profit that can be guaranteed?

The bound comes from the multivariate Chebyshev–Cantelli inequality [1]_, and the answer is a
linear decision boundary. Because only first and second moments enter, nothing has to be assumed
about the shape of the class distributions — hence *distribution-free*. The price is that only those two
moments are used, so a strongly non-Gaussian signal in higher moments is invisible to the model.

Both models take the four standard cost terms:

.. code-block:: python

    from sklearn.datasets import make_classification
    from empulse.models import ProfMPMClassifier

    X, y = make_classification(n_samples=500, n_features=5, random_state=0)

    model = ProfMPMClassifier(tp_cost=-200, fp_cost=10)
    model.fit(X, y)

    print(model.coef_.round(3))
    print(model.result_['success'])

Like the linear models in :ref:`cslogit`, they expose ``coef_``, ``intercept_`` and a ``result_``
holding the optimizer's own report.

Shared or per-class worst case
------------------------------

The two models differ in a single modelling choice, and it is the reason to pick one over the
other.

:class:`~empulse.models.ProfMPMClassifier` constrains **both classes to share one worst-case
accuracy bound**, set by whichever of the two is tighter. It is the more conservative and the more
stable of the pair: one class cannot be sacrificed to flatter the other.

:class:`~empulse.models.ProfMEMPMClassifier` lets the two bounds **differ**. Where the costs are
strongly asymmetric this is what you want — guaranteeing a lot about the expensive class and less
about the cheap one is exactly the trade the cost matrix is asking for — but the extra freedom
makes the fit more sensitive to the covariance estimates.

.. code-block:: python

    from empulse.models import ProfMEMPMClassifier

    mempm = ProfMEMPMClassifier(tp_cost=-200, fp_cost=10).fit(X, y)
    print(mempm.coef_.round(3))

Start with :class:`~empulse.models.ProfMPMClassifier`, and move to
:class:`~empulse.models.ProfMEMPMClassifier` when the classes genuinely warrant different
guarantees.

Regularisation
--------------

``lambda_reg`` selects which formulation is solved, and the default of ``0`` is not merely "no
penalty" — it changes the constraint set.

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - ``lambda_reg``
      - Formulation
    * - ``0`` (default)
      - The original published models. :class:`~empulse.models.ProfMEMPMClassifier` constrains the
        weight vector to unit norm; :class:`~empulse.models.ProfMPMClassifier` leaves it
        unconstrained. No penalty term.
    * - ``> 0``
      - The Lp-regularised variants. The unit-norm constraint is dropped and an L1 or L2 penalty on
        the weights is added to the objective instead, controlling the scale of ``w``.

``penalty`` chooses between ``'l1'`` and ``'l2'`` for the regularised form, and ``ridge_penalty``
adds a small amount to the diagonal of the covariance estimates. Raise it when the covariance
matrix is near-singular — many correlated features, or few samples relative to features — and the
solver struggles to converge.

.. code-block:: python

    import numpy as np

    plain = ProfMPMClassifier(tp_cost=-200, fp_cost=10).fit(X, y)
    regularised = ProfMPMClassifier(
        tp_cost=-200, fp_cost=10, penalty='l1', lambda_reg=1.0, ridge_penalty=1e-4
    ).fit(X, y)

    print(np.linalg.norm(plain.coef_).round(4))
    print(np.linalg.norm(regularised.coef_).round(4))

The penalty shrinks the **scale** of the weight vector rather than driving individual coefficients
to exactly zero: the constrained problem is solved continuously, so ``'l1'`` here is a magnitude
penalty, not a feature selector. Judge its effect by the norm of ``coef_``, and use it to stabilise
a fit rather than to prune features.

.. warning::
    Both models are **class-dependent only**. Array-valued costs are averaged before fitting,
    because a worst-case bound derived from class moments has no notion of an individual row. If
    per-customer costs must drive the decision boundary, use one of the models in :ref:`cslogit`,
    :ref:`csboost` or :ref:`cstree` instead.

Only :class:`~empulse.metrics.MaxProfit` is accepted as a ``loss``; any other strategy raises a
``ValueError`` naming the restriction. This is not a limitation to work around — the whole
derivation is in terms of a profit function of the true and false positive rates.

.. code-block:: python

    import sympy
    from empulse.metrics import CostMatrix, MaxProfit, Metric

    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    churn_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1, accept_rate=0.3)
    )

    model = ProfMPMClassifier(loss=Metric(churn_matrix, MaxProfit()))
    model.fit(X, y, clv=200)

Profit-driven symbolic regression
=================================

:class:`~empulse.models.ProfSRClassifier` searches a space of *expressions* — arithmetic
combinations of the input features — using genetic programming [2]_, and scores each candidate
directly on the profit metric. The winning expression's output is squashed through a logistic
function to give a probability.

The result is a decision function you can read:

.. code-block:: python

    from empulse.models import ProfSRClassifier

    model = ProfSRClassifier(generations=5, population_size=50, random_state=42)
    model.fit(X, y, tp_cost=-200, fp_cost=10)

    print(model.model_)

That readability is the point. A linear model tells you the weight on each feature; a symbolic model
can tell you that what matters is a *ratio* or a *product* of two features, and say so in a form a
domain expert can argue with.

.. note::
    :class:`~empulse.models.ProfSRClassifier` needs the optional
    `gplearn <https://gplearn.readthedocs.io/>`_ dependency. Install it with
    ``pip install empulse[symbolic]``.

Controlling the search
----------------------

.. list-table::
    :widths: 26 74
    :header-rows: 1

    * - Parameter
      - Effect
    * - ``generations``
      - How many rounds of evolution to run. More generations find better expressions and take
        proportionally longer.
    * - ``population_size``
      - How many candidate programs are kept per generation. Larger populations explore more of the
        space but cost memory and time per generation.
    * - ``parsimony_coefficient``
      - Penalty on program length. This is the knob that keeps expressions readable: raise it when
        the fitted program has grown into something nobody can interpret, lower it if the model is
        underfitting.
    * - ``random_state``
      - The search is stochastic. Fix this or successive fits will differ.

``parsimony_coefficient`` deserves particular attention. Genetic programming suffers from *bloat*:
without pressure against it, programs grow ever longer while barely improving fitness, and the
interpretability that motivated the model evaporates.

Unlike the minimax models, :class:`~empulse.models.ProfSRClassifier` accepts any strategy, since
evaluating a candidate program only requires a scalar fitness — including the two ranking-based
strategies that gradient methods cannot use.

.. code-block:: python

    from empulse.metrics import AUEPC, CostMatrix, Metric

    matrix = CostMatrix().add_fp_cost('c_fp').add_fn_cost('c_fn').set_default(c_fp=1.0, c_fn=5.0)

    model = ProfSRClassifier(
        loss=Metric(matrix, AUEPC()), generations=5, population_size=50, random_state=42
    )
    model.fit(X, y)

    print(model.n_iter_)

When to reach for these
=======================

.. list-table::
    :widths: 34 66
    :header-rows: 1

    * - Model
      - Best when
    * - :class:`~empulse.models.ProfMPMClassifier`
      - You need a guarantee rather than an average, and cannot justify a distributional
        assumption. The conservative default of the pair.
    * - :class:`~empulse.models.ProfMEMPMClassifier`
      - The same, but the costs are asymmetric enough that the two classes deserve different
        worst-case guarantees.
    * - :class:`~empulse.models.ProfSRClassifier`
      - You want an interpretable *formula*, not coefficients, and can afford an evolutionary
        search.

All three are ordinary scikit-learn estimators and work in
:class:`~sklearn.pipeline.Pipeline`, :func:`~sklearn.model_selection.cross_val_score` and
:class:`~sklearn.model_selection.GridSearchCV` — see :ref:`instance_based_cv` for routing, keeping
in mind that the minimax models will average per-row costs regardless.

References
==========

.. [1] Bravo, C., & Vanderschueren, T. (2023). Profit maximizing distribution-free classifiers:
    a study on the minimax probability machine. In *Joint European Conference on Machine Learning
    and Knowledge Discovery in Databases*.

.. [2] Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of
    Natural Selection*. MIT Press.
