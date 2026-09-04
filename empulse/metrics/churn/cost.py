from ..metric.prebuilt_metrics import make_churn_cost_metric

expected_cost_loss_churn = make_churn_cost_metric()
expected_cost_loss_churn.__name__ = 'expected_cost_loss_churn'
expected_cost_loss_churn.__doc__ = r"""
Expected cost of a classifier for customer churn.

The cost function presumes a situation where identified churners are contacted and offered an
incentive to remain customers. Only a fraction of churners accepts the incentive offer. For
detailed information, consult the paper [1]_. ``y_proba`` should be (calibrated) probabilities.
This metric always returns the average cost per sample.

.. seealso::
    :class:`~empulse.models.B2BoostClassifier` : Uses this metric as the training objective.

.. rubric:: Methods

``__call__(y_true, y_proba, *, accept_rate=0.3, clv=200, incentive_fraction=0.05, contact_cost=1)``
    Compute the expected cost of a classifier.

``optimal_threshold(y_true, y_proba, *, accept_rate=0.3, clv=200, incentive_fraction=0.05, contact_cost=1)``
    Compute the classification threshold(s) that minimize(s) the expected cost.

``optimal_rate(y_true, y_proba, *, accept_rate=0.3, clv=200, incentive_fraction=0.05, contact_cost=1)``
    Compute the predicted positive rate that minimizes the expected cost.

Examples
--------
.. code-block:: python

    from empulse.metrics import expected_cost_loss_churn

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    expected_cost_loss_churn(
        y_true, y_proba, accept_rate=0.3, clv=200, incentive_fraction=0.05, contact_cost=1
    )

References
----------
.. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
    B2Boost: Instance-dependent profit-driven modelling of B2B churn.
    Annals of Operations Research, 1-27.
"""
