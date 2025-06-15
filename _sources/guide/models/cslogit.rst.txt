.. _cslogit:

============================================
Cost-Sensitive Logistic Regression (CSLogit)
============================================

CSLogit is a cost-sensitive logistic regression model that optimizes the
:func:`~empulse.metrics.expected_cost_loss` during training with elastic net regularization [1]_.
Due to being based on a logistic model, the model is interpretable and trains relatively quickly.

Regularization
==============

The strength of regularization can be controlled by the ``C`` parameter.
The ``l1_ratio`` parameter controls the ratio of L1 regularization to L2 regularization.
By default ``l1_ratio`` is set to 1, which means L1 regularization is used
and a sparse solution if found for the coefficients.

.. code-block:: python

    from empulse.models import CSLogitClassifier

    cslogit = CSLogitClassifier(C=100, l1_ratio=0.2)

Cost Matrix
===========

The CSLogit allows constant class-dependent costs to be passed during instantiation.

.. code-block:: python

    cslogit = CSLogitClassifier(fp_cost=5, fn_cost=1, tp_cost=1, tn_cost=1)

Instance-dependent costs can be passed during training in the ``fit`` method.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification()
    fp_cost = np.random.rand(X.shape[0])  # instance-dependent costs
    cslogit = CSLogitClassifier(fn_cost=1, tp_cost=1, tn_cost=1)  # class-dependent costs

    cslogit.fit(X, y, fp_cost=fp_cost)

Note that class-dependent costs can also still be passed during training.
If costs are both passed during instantiation and training, the costs passed during training will be used.

Optimization
============

CSLogit, by default, optimizes the average expected cost through
`L-BFGS-B optimization <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_.
However, CSLogit offers flexibility in terms of customization.
You can use different loss functions and change the optimization algorithms.

Custom Loss Functions
---------------------

CSLogit allows the use of any cost-sensitive metric as the loss function.
The metric must take ``tp_cost``, ``fp_cost``, ``tn_cost``, ``fn_cost``, as keyword arguments.
To use a different metric,
simply pass the metric to the :class:`~empulse.models.CSLogitClassifier` initializer.

.. code-block:: python

    from empulse.metrics import expected_savings_score

    cslogit = CSLogitClassifier(loss=expected_savings_score)

Note that the this will not have the desired effect since by default CSLogit minimizes the loss function.
Instead, you can pass the inverse of the metric to the ``loss`` argument.

.. code-block:: python

    def inverse_expected_savings_score(
        y_true,
        y_pred,
        tp_cost=0.0,
        fp_cost=0.0,
        tn_cost=0.0,
        fn_cost=0.0
    ):
        return -expected_savings_score(
            y_true,
            y_pred,
            tp_cost=tp_cost,
            fp_cost=fp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost
        )

    cslogit = CSLogitClassifier(loss=inverse_expected_savings_score)

Custom Optimization Algorithms
------------------------------

CSLogit also supports the use of other optimization algorithms.
If you can fit them in an optimize function, you can use them to optimize the loss function.
For instance, if you want to use the L-BFGS-B algorithm from :mod:`scipy:scipy.optimize`
with the coefficients being bounded between -5 and 5, you can do the following:

.. code-block:: python

    from scipy.optimize import minimize, OptimizeResult

    def optimize(objective, X, max_iter=10000, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        bounds = [(-5, 5)] * X.shape[1]
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,
            },
            **kwargs
        )
        return result

    cslogit = CSLogitClassifier(optimize_fn=optimize)

Any arguments passed to ``optimizer_params`` will be passed to the ``optimize_fn`` during training.
So in this case we can dynamically change the maximum number of iterations for the optimizer.


.. code-block:: python

    def optimize(objective, X, max_iter=10000, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        bounds = [(-5, 5)] * X.shape[1]
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,
            },
            **kwargs
        )
        return result

    cslogit = CSLogitClassifier(optimize_fn=optimize, optimizer_params={'max_iter': 1000})

References
==========

.. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
       Instance-dependent cost-sensitive learning for detecting transfer fraud.
       European Journal of Operational Research, 297(1), 291-300.