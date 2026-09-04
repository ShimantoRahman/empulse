from ..metric.prebuilt_metrics import make_churn_max_profit_metric

mpc_score = make_churn_max_profit_metric(stochastic=False)
#: Maximum Profit Measure for Customer Churn (MPC).
#:
#: MPC presumes a situation where identified churners are contacted and offered an incentive to
#: remain customers. Only a fraction of churners accepts the incentive offer, described by a
#: constant ``accept_rate``. For detailed information, consult the paper [1]_.
#:
#: Call as ``mpc_score(y_true, y_score, accept_rate=0.3, clv=200, incentive_cost=10,
#: contact_cost=1)``. Use :meth:`~empulse.metrics.Metric.optimal_rate` on this metric to get the
#: fraction of the customer base that should be targeted to maximize profit,
#: :meth:`~empulse.metrics.Metric.optimal_threshold` for the corresponding classification
#: threshold, and :func:`~empulse.metrics.empc_score` for a stochastic version of this metric.
#:
#: The MPC is defined as [1]_:
#:
#: .. math::  CLV (\gamma (1 - \delta) - \phi) \pi_0 F_0(T) - CLV (\delta + \phi) \pi_1 F_1(T)
#:
#: The MPC requires that the churn class is encoded as 0, and it is NOT interchangeable.
#: However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).
#:
#: Examples
#: --------
#: .. code-block:: python
#:
#:     from empulse.metrics import mpc_score
#:
#:     y_true = [0, 1, 0, 1, 0, 1, 0, 1]
#:     y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
#:     mpc_score(y_true, y_score)
#:
#: References
#: ----------
#: .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
#:     A Novel Profit Maximizing Metric for Measuring Classification
#:     Performance of Customer Churn Prediction Models. IEEE Transactions on
#:     Knowledge and Data Engineering, 25(5), 961-973.
