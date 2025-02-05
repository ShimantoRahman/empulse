.. _user_defined_value_metric:

==============================================
Define your own cost-sensitive or value metric
==============================================

Empulse allows users to define their own cost-senstive/value metric using the :class:`empulse.metrics.Metric` class.
This is useful for when the default metrics are not appropriate for the your application.

As an example, we will redefine the Maximum Profit measure for customer churn (MPC) and
Expected Maximum Profit measure for customer churn (EMPC).

Implementing the MPC measure
----------------------------

In the case of customer churn, the costs and benefits are defined as follows:

- **True positive**: The company contacts the churner with cost :math:`f` and
  sends an incentive offer with cost :math:`d`.
  A proportion :math:`\gamma` of the churners accept the offer and stay with the company,
  retaining their Customer Lifetime Value :math:`CLV`.
  The remaining proportion :math:`1 - \gamma` of the churners leave the company and do not accept the offer.
  This results in a true positive benefit of :math:`\gamma (CLV-d-f) - (1-\gamma) f`.

- **False positive**: The company contacts the non-churner with cost :math:`f` and
  sends an incentive offer with cost :math:`d`.
  The non-churner accepts the offer and stays with the company.
  This results in a false positive cost of :math:`d+f`.

- **True negative**: The company does not send an incentive offer to the non-churner.
  Since the company does not take action, this has no cost or benefit.

- **False negative**: The company does not contact the churner and the customer leaves the company.
  Since the company does not take action, this has no cost or benefit.


To define out cost-benefits matrix, we can use Sympy to define the variables and equations.
Sympy is a Python library for symbolic mathematics and the :class:`empulse.metrics.Metric` class
uses Sympy under the hood to compute the metric.

.. code-block:: python

    import sympy

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

:class:`empulse.metrics.Metric` class uses a builder design pattern to step-by-step define the metric.
First, you need to define which type of metric you want to implement.
In this case, we want to implement a maximum profit metric, so will pass the ``kind`` as ``"max_profit"``.


.. code-block:: python

    from empulse.metrics import Metric

    mpc_score = Metric(kind='max profit')

Afterwards we can start assembling all parts of the cost-benefit matrix.

1. We can add the true positive benefit when a churner accepts the offer and stays with the company;
2. the true possitive benefit when a churner does not accept the offer and leaves the company;
3. the false positive cost when a non-churner accepts the offer and stays with the company.

.. code-block:: python

    mpc_score.add_tp_benefit(gamma * (clv - d - f))
    mpc_score.add_tp_benefit((1 - gamma) * -f)
    mpc_score.add_fp_cost(d + f)

Now that we have established the cost-benefit matrix, we just need to build the metric before we can use it!

.. code-block:: python

    mpc_score.build()

Now that the metric is built, you can use it like any other metric in scikit-learn.

.. code-block:: python

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    mpc_score(y_true, y_proba, clv=100, d=10, f=1, gamma=0.3)

One issue with the current implementation is that the arguments ``d``, ``f`` and ``gamma`` not very descriptive.
We can easily change this by using the ``alias`` method before building the metric.

.. code-block:: python

    mpc_score = (
        Metric(kind='max profit')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .build()
    )

    mpc_score(y_true, y_proba, clv=100, incentive_cost=10, contact_cost=1, accept_rate=0.3)

One final improvement we can make is set the default values for the cost-benefit matrix,
through the ``set_default`` method.

.. code-block:: python

    mpc_score = (
        Metric(kind='max profit')
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1, accept_rate=0.3)
        .build()
    )

    mpc_score(y_true, y_proba, clv=100)

Implementing the EMPC measure
-----------------------------

The biggest difference between the Maximum Profit function and the Expected Maximum Profit function
is that the costs and benefits can be stochastic.

In the case of customer churn, there is only one stochastic variable,
the proportion of churners who accept the offer :math:`\gamma`.
:math:`\gamma` follows a Beta distribution with parameters :math:`\alpha` and :math:`\beta`.

The only thing that you need to change from the MPC example above, is to define ``gamma`` as a stochastic variable.

.. code-block:: python

    clv, d, f, alpha, beta = sympy.symbols('clv d f alpha beta')
    gamma = sympy.stats.Beta('gamma', alpha, beta)

    empc_score = (
        Metric(kind="max profit")
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1, alpha=6, beta=14)
        .build()
    )

    empc_score(y_true, y_proba, clv=100)

You can also define :math:`\gamma` to follow a Uniform distribution with from 0 to 1.

.. code-block:: python

    clv, d, f = sympy.symbols('clv d f')
    gamma = sympy.stats.Uniform('gamma', 0, 1)

    empc_score = (
        Metric(kind="max profit")
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1)
        .build()
    )

    empc_score(y_true, y_proba, clv=100)

Or instead of making :math:`\gamma` a stochastic variable, you can make :math:`\clv` a stochastic variable.
We'll define :math:`\clv` to follow a Gamma distribution with parameters :math:`\alpha` and :math:`\beta`.

.. code-block:: python

    d, f, gamma, alpha, beta = sympy.symbols('d f gamma alpha beta')
    clv = sympy.stats.Gamma('clv', alpha, beta)

    empc_score = (
        Metric(kind="max profit")
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1, accept_rate=0.3)
        .build()
    )

    empc_score(y_true, y_proba, alpha=6, beta=10)

Implementing expected cost and savings
--------------------------------------

Now that we have defined the cost-benefit matrix,
we can also create expected cost and savings metrics by just changing the ``kind`` of metric.

Expected Cost
~~~~~~~~~~~~~

.. code-block:: python

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    expected_cost_loss = (
        Metric(kind='cost')  # change the kind to 'savings'
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1, accept_rate=0.3)
        .build()
    )

    expected_cost_loss(y_true, y_proba, clv=100)

Expected Savings
~~~~~~~~~~~~~~~~

.. code-block:: python

    clv, d, f, gamma = sympy.symbols('clv d f gamma')

    expected_savings_score = (
        Metric(kind='savings')  # change the kind to 'savings'
        .add_tp_benefit(gamma * (clv - d - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(d + f)
        .alias({'incentive_cost': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(incentive_cost=10, contact_cost=1, accept_rate=0.3)
        .build()
    )

    expected_savings_score(y_true, y_proba, clv=100)