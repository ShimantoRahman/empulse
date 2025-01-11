.. _csboost:

====================================================
Cost-Sensitive Gradient Boosting (CSBoost & B2Boost)
====================================================

CSBoost
=======

CSBoost is a cost-sensitive gradient boosting model that optimizes the
:func:`~empulse.metrics.expected_cost_loss` with the Extreme gradient boosting algorithm [1]_.

The CSBoost model is a wrapper around the :class:`xgboost:xgboost.XGBClassifier` model.
By default a XGBoost model with default parameters is used, if the user does not pass an instance of the XGBoost model.
However, the user can pass an instance of the XGBoost model to the CSBoost model to customize the hyperparameters.

.. code-block:: python

    from empulse.models import CSBoostClassifier
    from xgboost import XGBClassifier

    csboost = CSBoostClassifier(XGBClassifier(n_estimators=100, max_depth=3))

Note that if CSBoost is used in a context where the hyperparameters of the XGBoost model are set dynamically,
like for example when training a :class:`~sklearn:sklearn.model_selection.GridSearchCV`,
the user should define a XGBoost model.
Otherwise sklearn will try to set the hyperparameters to a ``None`` value.

For example:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV

    csboost = CSBoostClassifier(XGBClassifier())
    param_grid = {'estimator__max_depth': [3, 5]}
    grid_search = GridSearchCV(csboost, param_grid, cv=2)

Cost Matrix
===========

CSBoost allows constant class-dependent costs to be passed during instantiation.

.. code-block:: python

    csboost = CSBoostClassifier(fp_cost=5, fn_cost=1, tp_cost=1, tn_cost=1)

Instance-dependent costs can be passed during training in the `fit` method.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification()
    fp_cost = np.random.rand(X.shape[0])  # instance-dependent costs
    csboost = CSBoostClassifier(fn_cost=1, tp_cost=1, tn_cost=1)  # class-dependent costs

    csboost.fit(X, y, fp_cost=fp_cost)

Note that class-dependent costs can also still be passed during training.
If costs are both passed during instantiation and training, the costs passed during training will be used.

B2Boost
=======

B2Boost can be seen as a use-case specific implementation of the CSBoost model specialized for B2B churn prediction.
Instead of taking the abstract true positive, false positive, true negative, and false negative costs as arguments,
B2Boost takes the customer lifetime value (CLV),
the cost of the incentive offered to potential churners (as a fraction of the CLV of the customer),
the contact cost, and the probability that a churn accepts the incentive as arguments.
B2Boost sees the CLV of a customer and corresponding incentive cost as instance-dependent costs.
All other costs are seen as class-dependent costs.

All aspects mentioned about the CSBoost model apply to the B2Boost model as well.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from empulse.models import B2BoostClassifier

    X, y = make_classification()
    clv = np.random.rand(X.shape[0]) * 100
    b2boost = B2BoostClassifier(
        estimator=XGBClassifier(n_estimators=100, max_depth=3),
        accept_rate=0.2,
        incentive_fraction = 0.05,
        contact_cost = 10,
    )  # class-dependent costs

    b2boost.fit(X, y, clv=clv)  # instance-dependent costs

References
==========

.. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       European Journal of Operational Research, 297(1), 291-300.