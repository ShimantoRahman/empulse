from ..metric.prebuilt_metrics import make_churn_cost_metric

expected_cost_loss_churn = make_churn_cost_metric()
#: Expected cost of a classifier for customer churn.
#:
#: The cost function presumes a situation where identified churners are contacted and offered an
#: incentive to remain customers. Only a fraction of churners accepts the incentive offer. For
#: detailed information, consult the paper [1]_.
#:
#: Call as ``expected_cost_loss_churn(y_true, y_proba, accept_rate=0.3, clv=200,
#: incentive_fraction=0.05, contact_cost=1)``. ``y_proba`` should be (calibrated) probabilities.
#: This metric always returns the average cost per sample.
#:
#: .. seealso::
#:     :class:`~empulse.models.B2BoostClassifier` : Uses this metric as the training objective.
#:
#: References
#: ----------
#: .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
#:     B2Boost: Instance-dependent profit-driven modelling of B2B churn.
#:     Annals of Operations Research, 1-27.
