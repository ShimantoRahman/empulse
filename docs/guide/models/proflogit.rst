.. _proflogit:

=============================================
Profit-Driven Logistic Regression (ProfLogit)
=============================================

ProfLogit is a profit-driven logistic regression model that optimizes the expected maximum profit (EMP)
with elastic net regularization [1]_.

Regularization
==============

The strength of regularization can be controlled by the ``C`` parameter.
The ``l1_ratio`` parameter controls the ratio of L1 regularization to L2 regularization.
By default ``l1_ratio`` is set to 1, which means L1 regularization is used
and a sparse solution if found for the coefficients.

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from empulse.metrics import empc_score

    proflogit = ProfLogitClassifier(loss=empc_score, C=100, l1_ratio=0.2)


Here, ProfLogit, utilizes the EMPC metric (:func:`~empulse.metrics.empc_score`)
as its loss function, but any EMP metric can be used.

The model is optimized by a real-coded genetic algorithm (RGA).
The RGA runs for 1000 iterations, but it can stop early if the loss converges.
However, ProfLogit offers flexibility in terms of customization.
You can modify the stopping conditions, use different loss functions, and even change the optimization algorithms.

Optimization
============

Custom Stopping Conditions
--------------------------
The number of iterations, relative tolerance level,
or the number of iterations without improvement can be easily adjusted.
You just need to pass the desired values to the :class:`~empulse.models.ProfLogitClassifier` initializer.

.. code-block:: python

    from empulse.models import ProfLogitClassifier

    proflogit = ProfLogitClassifier(
        empc_score, optimizer_params={'max_iter': 10000, 'tolerance': 1e-3, 'patience': 100}
    )

For more advanced customization of the stopping conditions,
you can pass an optimize function to the :class:`~empulse.models.ProfLogitClassifier` initializer.
For example, if you want to use the RGA for a set amount of time, you can do the following:

.. code-block:: python

    from empulse.optimizers import Generation
    from scipy.optimize import OptimizeResult
    from time import perf_counter

    def optimize(objective, X, max_time=5, **kwargs) -> OptimizeResult:
        generation = Generation(**kwargs)
        bounds = [(-5, 5)] * X.shape[1]

        start = perf_counter()
        for _ in generation.optimize(objective, bounds):
            if perf_counter() - start > max_time:
                generation.result.message = "Maximum time reached."
                generation.result.success = True
                break
        return generation.result

    proflogit = ProfLogitClassifier(empc_score, optimize_fn=optimize, optimizer_params={'max_time': 10})

Or you can stop the RGA after a set number of fitness evaluations:

.. code-block:: python

    def optimize(objective, X, max_evals=10_000, **kwargs) -> OptimizeResult:
        generation = Generation(**kwargs)
        bounds = [(-5, 5)] * X.shape[1]

        for _ in rga.optimize(objective, bounds):
            if generation.result.nfev > max_evals:
                generation.result.message = "Maximum number of evaluations reached."
                generation.result.success = True
                break
        return generation.result

    proflogit = ProfLogitClassifier(empc_score, optimize_fn=optimize, optimizer_params={'max_evals': 10_000})

Custom Loss Functions
---------------------

ProfLogit allows the use of any metrics defined in the :mod:`empulse.metrics` module as the loss function.
To use a different metric,
simply pass the metric function to the :class:`~empulse.models.ProfLogitClassifier` initializer.

.. code-block:: python

    from empulse.metrics import empa_score

    proflogit = ProfLogitClassifier(loss=empa_score)

Custom Optimization Algorithms
------------------------------

ProfLogit also supports the use of other optimization algorithms.
If you can fit them in an optimize function, you can use them to optimize the loss function.
For instance, if you want to use the L-BFGS-B algorithm from scipy.optimize, you can do the following:

.. code-block:: python

    import numpy as np

    def optimize(objective, X, max_iter=10000, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        bounds = [(-5, 5)] * X.shape[1]
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
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

    proflogit = ProfLogitClassifier(empc_score, optimize_fn=optimize)

Note that EMPC is a maximization problem, so we need to pass the inverse objective function to the optimizer.

You can also use unbounded optimization algorithms like BFGS:

.. code-block:: python

    def optimize(objective, X, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='BFGS',
            **kwargs
        )
        return result

    proflogit = ProfLogitClassifier(empc_score, optimize_fn=optimize)

References
==========

.. [1] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
    Snoeck, M. (2017). Profit Maximizing Logistic Model for
    Customer Churn Prediction Using Genetic Algorithms.
    Swarm and Evolutionary Computation.