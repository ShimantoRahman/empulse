.. _proflogit:

=====================
Customizing ProfLogit
=====================

ProfLogit, by default,
utilizes the EMPC metric as its loss function and is optimized by a real-coded genetic algorithm (RGA).
The RGA runs for 1000 iterations, but it can stop early if the loss converges.
However, ProfLogit offers flexibility in terms of customization.
You can modify the stopping conditions, use different loss functions, and even change the optimization algorithms.


Custom Stopping Conditions
--------------------------
The number of iterations, relative tolerance level,
or the number of iterations without improvement can be easily adjusted.
You just need to pass the desired values to the :class:`~empulse.models.ProfLogitClassifier` initializer.

.. code-block:: python

    from empulse.models import ProfLogitClassifier

    proflogit = ProfLogitClassifier(optimizer_params={'max_iter': 10000, 'tolerance': 1e-3, 'patience': 100})

For more advanced customization of the stopping conditions,
you can pass an optimize function to the :class:`~empulse.models.ProfLogitClassifier` initializer.
For example, if you want to use the RGA for a set amount of time, you can do the following:

.. code-block:: python

    from empulse.models import ProfLogitClassifier
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

    proflogit = ProfLogitClassifier(optimize_fn=optimize, optimizer_params={'max_time': 10})

Or you can stop the RGA after a set number of fitness evaluations:

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from empulse.optimizers import Generation
    from scipy.optimize import OptimizeResult

    def optimize(objective, X, max_evals=10_000, **kwargs) -> OptimizeResult:
        generation = Generation(**kwargs)
        bounds = [(-5, 5)] * X.shape[1]

        for _ in rga.optimize(objective, bounds):
            if generation.result.nfev > max_evals:
                generation.result.message = "Maximum number of evaluations reached."
                generation.result.success = True
                break
        return generation.result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, optimizer_params={'max_evals': 10_000})

Custom Loss Functions
---------------------

ProfLogit allows the use of any metrics defined in the :mod:`empulse.metrics` module as the loss function.
To use a different metric,
simply pass the metric function to the :class:`~empulse.models.ProfLogitClassifier` initializer.

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from empulse.metrics import empa_score

    proflogit = ProfLogitClassifier(loss=empa_score)

Custom Optimization Algorithms
------------------------------

ProfLogit also supports the use of other optimization algorithms.
If you can fit them in an optimize function, you can use them to optimize the loss function.
For instance, if you want to use the L-BFGS-B algorithm from scipy.optimize, you can do the following:

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from scipy.optimize import minimize, OptimizeResult
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

    proflogit = ProfLogitClassifier(optimize_fn=optimize)

Note that EMPC is a maximization problem, so we need to pass the inverse objective function to the optimizer.

You can also use unbounded optimization algorithms like BFGS:

.. code-block:: python

    from empulse.models import ProfLogitClassifier
    from scipy.optimize import minimize, OptimizeResult
    import numpy as np

    def optimize(objective, X, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='BFGS',
            **kwargs
        )
        return result

    proflogit = ProfLogitClassifier(optimize_fn=optimize)

