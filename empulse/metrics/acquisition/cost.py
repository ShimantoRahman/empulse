from ..metric.prebuilt_metrics import make_acquisition_cost_metric

expected_cost_loss_acquisition = make_acquisition_cost_metric()
expected_cost_loss_acquisition.__name__ = 'expected_cost_loss_acquisition'
expected_cost_loss_acquisition.__doc__ = r"""
Expected cost of a classifier for customer acquisition.

The cost function presumes a situation where leads are targeted either directly or
indirectly. Directly targeted leads are contacted and handled by the internal sales team.
Indirectly targeted leads are contacted and then referred to intermediaries, which receive a
commission. The company gains a contribution from a successful acquisition. ``y_proba`` should
be (calibrated) probabilities. This metric always returns the average cost per sample.

.. rubric:: Methods

All three methods take ``y_true`` and ``y_proba``, followed by the keyword-only parameters
``contribution=7000``, ``contact_cost=50``, ``sales_cost=500``, ``direct_selling=1`` and
``commission=0.1``.

``__call__(y_true, y_proba, **parameters)``
    Compute the expected cost of a classifier.

``optimal_threshold(y_true, y_proba, **parameters)``
    Compute the classification threshold(s) that minimize(s) the expected cost.

``optimal_rate(y_true, y_proba, **parameters)``
    Compute the predicted positive rate that minimizes the expected cost.

Examples
--------
.. code-block:: python

    from empulse.metrics import expected_cost_loss_acquisition

    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    expected_cost_loss_acquisition(y_true, y_proba, direct_selling=1)
"""
