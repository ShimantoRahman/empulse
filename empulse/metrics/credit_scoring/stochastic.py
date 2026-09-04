from ..metric.prebuilt_metrics import make_credit_scoring_empirical_max_profit_metric

empcs_score = make_credit_scoring_empirical_max_profit_metric()
#: Expected Maximum Profit measure for Credit Scoring (EMPCS).
#:
#: EMPCS presumes a situation where a company is considering whether to grant a loan to a
#: customer. Correctly identifying defaulters results in receiving a return on investment (ROI),
#: while incorrectly identifying non-defaulters as defaulters results in a loss of the loan
#: amount. The degree to which the loan is lost is determined by the probability that the entire
#: loan is lost (``default_rate``), the probability that the entire loan is paid back
#: (``success_rate``), and a uniform distribution of partial loan losses
#: (``1 - default_rate - success_rate``). For detailed information, consult the paper [1]_.
#:
#: This is a :class:`~empulse.metrics.MixtureMetric` of three
#: :class:`~empulse.metrics.Metric` components (one for each point mass, one for the continuous
#: uniform piece) since :mod:`sympy.stats` cannot express this mixed distribution directly. Call
#: as ``empcs_score(y_true, y_score, success_rate=0.55, default_rate=0.1, roi=0.2644)``. Use
#: :meth:`~empulse.metrics.MixtureMetric.optimal_rate` on this metric to get the fraction of loan
#: applications that should be accepted to maximize profit, and
#: :func:`~empulse.metrics.mpcs_score` for a deterministic version of this metric.
#:
#: The EMP measure for Credit Scoring is defined as [1]_:
#:
#: .. math:: \int_0^1 \lambda \pi_0 F_0(T) - ROI \pi_1 F_1(T) \cdot h(\lambda) d\lambda
#:
#: The EMP measure for Credit Scoring requires that the default class is encoded as 0, and it is
#: NOT interchangeable. However, this implementation assumes the standard notation
#: ('default': 1, 'no default': 0).
#:
#: References
#: ----------
#: .. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
#:     Development and application of consumer credit scoring models using profit-based
#:     classification measures. European Journal of Operational Research, 238(2), 505-513.
