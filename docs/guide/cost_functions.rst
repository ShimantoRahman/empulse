=======================================
Boosting Algorithms With Cost Functions
=======================================

Empulse provides a number of cost functions for use with XGBoost.
The :mod:`empulse.models` module defines the :class:`empulse.models.B2BoostClassifier` as convenience,
but you can use any of the cost functions.

For example, to use the :func:`empulse.metrics.create_objective_acquisition` with xgboost, you can do the following:

.. code-block::  python

        import xgboost as xgb
        from empulse.metrics import create_objective_acquisition

        # create the cost function
        cost_function = create_objective_acquisition()

        # create the xgboost classifier
        clf = xgb.XGBClassifier(objective=cost_function, n_estimators=100, max_depth=3)

        # fit the model
        clf.fit(X_train, y_train)

        # predict
        y_pred = clf.predict(X_test)

        # predict probabilities
        y_pred_proba = clf.predict_proba(X_test)

