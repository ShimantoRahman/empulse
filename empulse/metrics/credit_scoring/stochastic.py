from ..metric.prebuilt_metrics import make_credit_scoring_empirical_max_profit_metric

empcs_score = make_credit_scoring_empirical_max_profit_metric()
empcs_score.__name__ = 'empcs_score'
empcs_score.__doc__ = r"""
Expected Maximum Profit measure for Credit Scoring (EMPCS).

EMPCS presumes a situation where a company is considering whether to grant a loan to a
customer. Correctly identifying defaulters results in receiving a return on investment (ROI),
while incorrectly identifying non-defaulters as defaulters results in a loss of the loan
amount. The degree to which the loan is lost is determined by the probability that the entire
loan is lost (``default_rate``), the probability that the entire loan is paid back
(``success_rate``), and a uniform distribution of partial loan losses
(``1 - default_rate - success_rate``). For detailed information, consult the paper [1]_.

This is a :class:`~empulse.metrics.MixtureMetric` of three
:class:`~empulse.metrics.Metric` components (one for each point mass, one for the continuous
uniform piece) since :mod:`sympy.stats` cannot express this mixed distribution directly.
See :func:`~empulse.metrics.mpcs_score` for a deterministic version of this metric.

The EMP measure for Credit Scoring is defined as [1]_:

.. math:: \int_0^1 \lambda \pi_0 F_0(T) - ROI \pi_1 F_1(T) \cdot h(\lambda) d\lambda

The EMP measure for Credit Scoring requires that the default class is encoded as 0, and it is
NOT interchangeable. However, this implementation assumes the standard notation
('default': 1, 'no default': 0).

.. rubric:: Methods

``__call__(y_true, y_score, *, success_rate=0.55, default_rate=0.1, roi=0.2644)``
    Compute the expected maximum profit that can be achieved by a classifier at its optimal
    decision threshold.

``optimal_threshold(y_true, y_score, *, success_rate=0.55, default_rate=0.1, roi=0.2644)``
    Compute the classification threshold that maximizes the expected profit.

``optimal_rate(y_true, y_score, *, success_rate=0.55, default_rate=0.1, roi=0.2644)``
    Compute the predicted positive rate (fraction of loan applications that should be
    accepted) at which the maximum expected profit is achieved.

Examples
--------
.. code-block:: python

    from empulse.metrics import empcs_score

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    empcs_score(y_true, y_score, success_rate=0.55, default_rate=0.1, roi=0.2644)

References
----------
.. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
    Development and application of consumer credit scoring models using profit-based
    classification measures. European Journal of Operational Research, 238(2), 505-513.
"""
