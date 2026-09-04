from ..metric.prebuilt_metrics import make_credit_scoring_max_profit_metric

mpcs_score = make_credit_scoring_max_profit_metric()
#: Maximum Profit measure for Credit Scoring (MPCS).
#:
#: MPCS presumes a situation where a company is considering whether to grant a loan to a
#: customer. Correctly identifying defaulters results in receiving a return on investment (ROI),
#: while incorrectly identifying non-defaulters as defaulters results in a fraction of the loan
#: amount being lost. For detailed information, consult the paper [1]_.
#:
#: Call as ``mpcs_score(y_true, y_score, loan_lost_rate=0.275, roi=0.2644)``. Use
#: :meth:`~empulse.metrics.Metric.optimal_rate` on this metric to get the fraction of loan
#: applications that should be accepted to maximize profit, and
#: :func:`~empulse.metrics.empcs_score` for a stochastic version of this metric.
#:
#: The MP measure for Credit Scoring is defined as [1]_:
#:
#: .. math:: \max_t \lambda \pi_0 F_0(t) - ROI \pi_1 F_1(t)
#:
#: The MP measure for Credit Scoring requires that the default class is encoded as 0, and it is
#: NOT interchangeable. However, this implementation assumes the standard notation
#: ('default': 1, 'no default': 0).
#:
#: References
#: ----------
#: .. [1] Verbraken, T., Bravo, C., Weber, R., & Baesens, B. (2014).
#:     Development and application of consumer credit scoring models using profit-based
#:     classification measures. European Journal of Operational Research, 238(2), 505-513.
