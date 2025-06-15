.. _cost_functions:

Boosting Algorithms Custom Cost Functions
==========================================

Empulse provides a number of cost functions for use with the :class:`xgboost:xgboost.XGBClassifier`.
These cost function wrap around various implementation of the :func:`~empulse.metrics.expected_cost_loss`.
They compute the gradient and hessian of the expected cost loss,
which is then used by the boosting algorithm to optimize the cost function.

The :mod:`empulse.models` module defines the :class:`empulse.models.B2BoostClassifier` and
:class:`empulse.models.CSBoostClassifier` as a convenience,
but you easily define your own XGBoost model and use the cost functions directly.

For example, to use the :func:`empulse.metrics.make_objective_aec` with xgboost, you can do the following:

.. code-block::  python

        import xgboost as xgb
        from sklearn.datasets import make_classification
        from empulse.metrics import make_objective_aec

        X, y = make_classification()
        cost_function = make_objective_aec(fp_cost=5, fn_cost=1, tp_cost=1, tn_cost=1)
        clf = xgb.XGBClassifier(objective=cost_function, n_estimators=100, max_depth=3)
        clf.fit(X, y)


Empulse also contains :func:`empulse.metrics.make_objective_churn` and
:func:`empulse.metrics.make_objective_acquisition` functions for optimizing case-specific implementation of the
:func:`~empulse.metrics.expected_cost_loss`.
