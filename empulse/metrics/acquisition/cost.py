from ..metric.prebuilt_metrics import make_acquisition_cost_metric

expected_cost_loss_acquisition = make_acquisition_cost_metric()
#: Expected cost of a classifier for customer acquisition.
#:
#: The cost function presumes a situation where leads are targeted either directly or
#: indirectly. Directly targeted leads are contacted and handled by the internal sales team.
#: Indirectly targeted leads are contacted and then referred to intermediaries, which receive a
#: commission. The company gains a contribution from a successful acquisition.
#:
#: Call as ``expected_cost_loss_acquisition(y_true, y_proba, contribution=7000, contact_cost=50,
#: sales_cost=500, direct_selling=1, commission=0.1)``. ``y_proba`` should be (calibrated)
#: probabilities. This metric always returns the average cost per sample.
