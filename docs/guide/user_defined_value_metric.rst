=========================
User-Defined Value Metric
=========================

Empulse allows users to define their own value metric, following the Expected Maximum Profit (EMP) specification.
This is useful for when the default metrics are not appropriate for the user's application.
The user-defined value metric can be either deterministic or stochastic,
using the :func:`empulse.metrics.max_profit` and :func:`empulse.metrics.emp` functions, respectively.

As an example, we will redefine the Maximum Profit measure for customer churn (MPC) and
Expected Maximum Profit measure for customer churn (EMPC).

Implementing the MPC measure
----------------------------

The Maximum Profit function :func:`empulse.metrics.max_profit`
requires the user to define the costs and benefits of each possible model prediction
(true positive, false positive, true negative and false negative).
In the case of customer churn, the costs and benefits are defined as follows:

    - **True positive**: The company contacts the churner with cost :math:`f` and
      sends an incentive offer with cost :math:`d`.
      A proportion :math:`\gamma` of the churners accept the offer and stay with the company,
      retaining their Customer Lifetime Value :math:`CLV`.
      The remaining proportion :math:`1 - \gamma` of the churners leave the company and do not accept the offer.
      This results in a true positive benefit of :math:`\gamma (CLV-d-f) - (1-\gamma)(d+f)`.

    - **False positive**: The company contacts the non-churner with cost :math:`f` and
      sends an incentive offer with cost :math:`d`.
      The non-churner accepts the offer and stays with the company.
      This results in a false positive cost of :math:`d+f`.

    - **True negative**: The company does not send an incentive offer to the non-churner.
      Since the company does not take action, this has no cost or benefit.

    - **False negative**: The company does not contact the churner and the customer leaves the company.
      Since the company does not take action, this has no cost or benefit.

.. code-block:: python

    from empulse.metrics import max_profit_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    clv = 200  # customer lifetime value
    d = 10  # cost of incentive offer
    f = 1  # cost of contacting customer
    gamma = 0.3  # proportion of churners who accept the offer
    tp_benefit = gamma * (clv - d - f) - (1 - gamma) * (d + f)
    fp_cost = d + f

    max_profit_score(y_true, y_pred, tp_benefit=tp_benefit, fp_cost=fp_cost)

Note that when computing the Maximum Profit for customer churn,
the :func:`empulse.metrics.mpc` should be preferred over the :func:`empulse.metrics.max_profit` function.

Implementing the EMPC measure
-----------------------------

The biggest difference between the Maximum Profit function :func:`empulse.metrics.max_profit`
and the Expected Maximum Profit function :func:`empulse.metrics.emp`
is that the latter requires the user to define a weighted probability density function (PDF)
of the joint distribution of the stochastic benefits and costs.
The weighted PDF is defined as the product of the PDF and the step size of the benefits and costs.

In the case of customer churn, there is only one stochastic variable,
the proportion of churners who accept the offer :math:`\gamma`.
:math:`\gamma` follows a Beta distribution with parameters :math:`\alpha` and :math:`\beta`.
Here the weighted PDF function needs to determine the product of :math:`h(\gamma)` and :math:`\Delta \gamma`
of the following equation:

:math:`EMPC \approx \sum_\gamma [[\gamma (clv - d - f) - (1 - \gamma) f] \pi_0 F_0(T) - (d+f) \pi_1 F_1(T) ] \cdot \underbrace{h(\gamma) \Delta \gamma}_{\text{weighted pdf}}`

Since the weighted PDF function only received the benefits, costs and step sizes as input,
gamma will need to be derived from the benefits and costs.

.. math::

    b_0 &= \gamma (CLV - d - f) - (1 - \gamma) f \\
    b_0 &= \gamma (CLV - d) - f  \\
    \gamma &= \frac{b_0 + f}{(CLV - d)} \\

To compute :math:`h(\gamma)`, we need to compute the PDF of :math:`\gamma`,
which can be done through the ``pdf()`` method of :data:`scipy:scipy.stats.beta`.

To compute :math:`\Delta \gamma`, we need to compute the step size of :math:`\gamma`.
Assume two consecutive values of :math:`\gamma` are :math:`\gamma_0` and :math:`\gamma_1`.
We can take the difference between the two values of the profit to compute the step size of :math:`\gamma`:

.. math::

    \Delta \gamma &= \gamma_1 - \gamma_0 \\
    \Delta \gamma &= \frac{b_1 + f}{(CLV - d)} - \frac{b_0 + f}{(CLV - d)} \\
    \Delta \gamma &= \frac{b_1 - b_0}{(CLV - d)} \\

The weighted PDF function can now be implemented as follows:

.. code-block:: python

    from scipy.stats import beta

    def weighted_pdf(b0, b1, c0, c1, b0_step, b1_step, c0_step, c1_step):
        gamma = (b0 + f) / (clv - d)
        gamma_step = b0_step / (clv - d)
        return beta.pdf(gamma, a=6, b=14) * gamma_step

Since the true positive is stochastic since it depends on :math:`\gamma`,
the value for ``tp_benefit`` should be set to a range of values, with a minimum and maximum value.
The minimum value is the benefit when :math:`\gamma = 0` and the maximum value is the benefit when :math:`\gamma = 1`.

For :math:`\gamma = 0`:

.. math::

    b_0 &= \gamma (CLV - d - f) - (1 - \gamma) f \\
    b_0 &= 0 (CLV - d - f) - (1 - 0) f \\
    b_0 &= -f \\

For :math:`\gamma = 1`:

.. math::

    b_1 &= \gamma (CLV - d - f) - (1 - \gamma) f \\
    b_1 &= 1 (CLV - d - f) - (1 - 1) f \\
    b_1 &= CLV - d - f \\

When all combined the EMPC measure can be implemented as follows:

.. code-block:: python

    from empulse.metrics import emp
    from scipy.stats import beta

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

    clv = 200  # customer lifetime value
    d = 10  # cost of incentive offer
    f = 1  # cost of contacting customer
    tp_benefit = (-f, clv - d - f)  # range of values for the stochastic true positive benefit
    fp_cost = d + f  # deterministic cost of false positive

    def weighted_pdf(b0, b1, c0, c1, b0_step, b1_step, c0_step, c1_step):
        gamma = (b0 + f) / (clv - d)
        gamma_step = b0_step / (clv - d)
        return beta.pdf(gamma, a=6, b=14) * gamma_step

    emp(
        y_true,
        y_pred,
        weighted_pdf=weighted_pdf,
        tp_benefit=tp_benefit,
        fp_cost=fp_cost,
        n_buckets=1000  # number of buckets to use for the approximation
    )
