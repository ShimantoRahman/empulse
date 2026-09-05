.. _tutorial_cost_matrix:

=======================
Writing the Cost Matrix
=======================

A cost matrix answers one question for each of the four possible outcomes: *what does this cost
us?* Get that right and everything else in Empulse follows from it.

The four outcomes
=================

In our campaign the model predicts who will churn, and we contact whoever it flags. The positive
class is "will churn", so "predicted positive" means "contact this customer".

Write :math:`CLV_i` for a customer's lifetime value, and express the two campaign costs as
fractions of it:

- :math:`f` — the cost of contacting a customer (``contact_fraction``), incurred whether or not
  they accept.
- :math:`d` — the retention incentive (``incentive_fraction``), only paid if they accept.
- :math:`\gamma` — the probability a contacted customer accepts the offer (``accept_rate``).

Now work through each cell:

**True positive** — a real churner we contact. With probability :math:`\gamma` they accept: we keep
their value, minus the incentive and the contact cost. With probability :math:`1 - \gamma` they
leave anyway and we are out the contact cost.

.. math::
    \text{tp benefit} = \gamma\,(CLV_i - d \cdot CLV_i - f \cdot CLV_i) - (1 - \gamma)\, f \cdot CLV_i

**False positive** — a loyal customer we contact needlessly. They were never leaving, so there is
no value to save; we simply pay to contact them and, if they take it, the discount.

.. math::
    \text{fp cost} = d \cdot CLV_i + f \cdot CLV_i

**False negative** — a churner we failed to flag. They leave, and we lose their entire value.

.. math::
    \text{fn cost} = CLV_i

**True negative** — a loyal customer we correctly leave alone. Nothing happens, nothing is spent.

.. math::
    \text{tn benefit} = 0

Note the asymmetry that the baseline model could not see: a false negative costs a full
:math:`CLV_i`, while a false positive costs only a few percent of it. With the defaults below, a
false negative is roughly **17 times** more expensive.

Building it in Empulse
======================

:class:`~empulse.metrics.CostMatrix` builds this with a fluent API. Terms are symbolic, so the
business parameters stay named rather than being baked into numbers.

.. code-block:: python

    import sympy
    from empulse.metrics import CostMatrix

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma * (clv - d * clv - f * clv))
        .add_tp_benefit(-(1 - gamma) * f * clv)
        .add_fp_cost(d * clv + f * clv)
        .add_fn_cost(clv)
        .alias('accept_rate', gamma)
        .alias('incentive_fraction', d)
        .alias('contact_fraction', f)
        .set_default(accept_rate=0.3, incentive_fraction=0.05, contact_fraction=0.01)
    )

Three things are worth noting:

- **Costs and benefits are opposites.** ``add_tp_benefit`` and ``add_tp_cost`` are the same
  quantity with opposite signs; use whichever reads more naturally. Calling ``add_tp_benefit``
  twice, as above, adds both terms together.
- **Aliases** let callers write ``accept_rate=0.5`` instead of remembering that the symbol is
  ``gamma``.
- **Order matters**: call ``.alias()`` before ``.set_default()``, because defaults are resolved
  against the names in use at the time.

.. note::
    This is exactly the matrix :func:`~empulse.datasets.fetch_iranian_churn` already ships as
    ``dataset.cost_matrix``, so the rest of the tutorial uses that instead of rebuilding it. It is
    also the same matrix behind the prebuilt :ref:`churn metrics <prebuilt_churn_metrics>`.

Which parameters vary per customer?
===================================

``clv`` is instance-dependent: every customer has their own. The other three are single numbers
that apply to the whole campaign — which is why they have defaults and ``clv`` does not.

.. code-block:: python

    import pandas as pd
    from empulse.datasets import fetch_iranian_churn

    dataset = fetch_iranian_churn(backend=pd)

    print(dataset.instance_costs.keys())

Anything instance-dependent is passed at call time as an array; anything class-dependent can be
left to its default or overridden with a scalar. Empulse does not care which is which — you simply
pass values and it broadcasts appropriately.

Sanity-check your matrix
========================

A cost matrix is a business claim, so it is worth inspecting before trusting it. The derived
properties show the assembled expressions:

.. code-block:: python

    print(f'tp_benefit: {dataset.cost_matrix.tp_benefit}')
    print(f'fp_cost   : {dataset.cost_matrix.fp_cost}')
    print(f'fn_cost   : {dataset.cost_matrix.fn_cost}')
    print(f'tn_benefit: {dataset.cost_matrix.tn_benefit}')

If a term looks wrong here, every number downstream will be wrong too, in a way no test will catch.

Next
====

:doc:`03_evaluate` turns this cost matrix into metrics, and shows how the choice of strategy
changes the question being asked.
