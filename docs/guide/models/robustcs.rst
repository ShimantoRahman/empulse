.. _robustcs:

===============================================
Robust Cost-Sensitive Classification (RobustCS)
===============================================

Instance-dependent cost-sensitive (IDCS) learning methods have proven
useful for binary classification tasks where individual instances are associated
with variable misclassification costs.
However, IDCS methods are sensitive to noise and outliers in relation to instance-dependent misclassification
costs and their performance strongly depends on the cost distribution of the data sample.
The robust cost-sensitive classifier (:class:`~empulse.models.RobustCSClassifier`) makes IDCS methods more robust by
applying a three-step framework:

1. **Outlier detection**: Outliers are detected by training a :class:`~sklearn:sklearn.linear_model.HuberRegressor`
   on the instance-dependent costs.
2. **Outlier correction**: Outlier costs are corrected using the predictions of the Huber regressor.
3. **Robust cost-sensitive classification**: The corrected costs are used to train a cost-sensitive classifier.

For an in-depth explanation of the robust cost-sensitive framework, please refer to the paper [1]_.

Usage
=====

To make any cost-sensitive classifier robust, you can use the :class:`~empulse.models.RobustCSClassifier` class.
Simply pass the cost-sensitive classifier you want to make robust as the ``estimator`` parameter.

.. code-block:: python

    from empulse.models import RobustCSClassifier
    from empulse.models import CSLogitClassifier

    robust_cslogit = RobustCSClassifier(estimator=CSLogitClassifier())

By default, the robust cost-sensitive classifier uses the :class:`~sklearn:sklearn.linear_model.HuberRegressor`
with default parameters.
You can customize the outlier detection step by passing a custom outlier detector to the ``outlier_estimator`` parameter.

.. code-block:: python

    from sklearn.linear_model import HuberRegressor

    robust_cslogit = RobustCSClassifier(
        CSLogitClassifier(),
        outlier_estimator=HuberRegressor(C=0.1)
    )

:class:`~empulse.models.RobustCSClassifier` considers a cost an outlier if it the predicted value of the Huber regressor
is larger than 2.5 times the standardized residuals.
You can change this threshold by setting the ``outlier_threshold`` parameter.

.. code-block:: python

    robust_cslogit = RobustCSClassifier(CSLogitClassifier(), outlier_threshold=3)

By default, all instance-dependent costs are corrected (class-dependent costs are ignored).
If you wish to only correct particular costs, you can change the ``detect_outliers_for`` parameter.
For instance, to only correct false positive costs, you can set ``detect_outliers_for='fp_cost'``.

.. code-block:: python

    robust_cslogit = RobustCSClassifier(CSLogitClassifier(), detect_outliers_for='fp_cost')

Or if if you want to correct multiple costs, you can pass a list of cost names.

.. code-block:: python

    robust_cslogit = RobustCSClassifier(
        CSLogitClassifier(),
        detect_outliers_for=['fp_cost', 'fn_cost']
    )

To fit the robust cost-sensitive classifier,
you can use the ``fit`` method with the instance-dependent costs as you would with any other cost-sensitive model.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification()
    fp_cost = np.random.rand(X.shape[0])  # instance-dependent costs

    robust_cslogit = RobustCSClassifier(CSLogitClassifier())
    robust_cslogit.fit(X, y, fp_cost=fp_cost)

After fitting you can inspect the corrected costs using the ``costs_`` attribute.

.. code-block:: python

    print(robust_cslogit.costs_)

References
==========

.. [1] De Vos, S., Vanderschueren, T., Verdonck, T., & Verbeke, W. (2023).
       Robust instance-dependent cost-sensitive classification.
       Advances in Data Analysis and Classification, 17(4), 1057-1079.
