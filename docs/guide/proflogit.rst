=====================
Customizing Proflogit
=====================

By default Proflogit will uses the EMPC metric as its loss function and
optimized by a real-coded genetic algorithm (RGA).
The RGA goes for 1000 iterations, stopping early if the loss converges.
However, Proflogit can be customized to have different stopping conditions, use other loss functions and
optimization algorithms quite easily.

Custom Stopping Conditions
--------------------------

To change the number of iterations, relative tolerance level or number of iterations without improvement,
simply pass the desired values to the :class:`proflogit.ProflogitClassifier` initializer.
:class:`proflogit.ProflogitClassifier` initializer.

.. code-block:: python

    from empulse.models import ProflogitClassifier

    proflogit = ProflogitClassifier(max_iter=10000, tolerance=1e-3, patience=100)

To customize the stopping conditions even further, you can pass a optimize function to the
:class:`proflogit.ProflogitClassifier` initializer.

For instance, if you want to use the RGA for a set number of time, you can do the following:

.. code-block:: python

    from empulse.models import ProflogitClassifier
    from scipy.optimize import OptimizeResult
    from time import perf_counter

    def optimize(objective, bounds, max_time=5, **kwargs) -> OptimizeResult:
                rga = RGA(**kwargs)

                start = perf_counter()
                for _ in rga.optimize(objective, bounds):
                    if perf_counter() - start > max_time:
                        rga.result.message = "Maximum time reached."
                        rga.result.success = True
                        break
                return rga.result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, max_time=10)

Or you can stop the RGA after a set number of fitness evaluations:

.. code-block:: python

    from empulse.models import ProflogitClassifier
    from scipy.optimize import OptimizeResult

    def optimize(objective, bounds, max_evals=10000, **kwargs) -> OptimizeResult:
                rga = RGA(**kwargs)

                for _ in rga.optimize(objective, bounds):
                    if rga.result.nfev > max_evals:
                        rga.result.message = "Maximum number of evaluations reached."
                        rga.result.success = True
                        break
                return rga.result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, max_evals=10000)

Custom Loss Functions
---------------------
Any of the metrics defined in the :mod:`proflogit.metrics` module can be used.
To use a different metric, simply pass the metric function to the
:class:`proflogit.ProflogitClassifier` initializer.

.. code-block:: python

    from empulse.models import ProflogitClassifier
    from empulse.metrics import empa_score

    proflogit = Proflogit(loss_fn=empa_score)

Custom Optimization Algorithms
------------------------------
Other algorithms can be used to optimize the loss function if you can fit them in an optimize function.
For instance if you want to use the L-BFGS-B algorithm from scipy.optimize, you can do the following:

.. code-block:: python

    from scipy.optimize import minimize, OptimizeResult
    import numpy as np

    def optimize(objective, bounds, max_iter=10000, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(len(bounds))
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

    from scipy.optimize import minimize, OptimizeResult
    import numpy as np

    def optimize(objective, bounds, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(len(bounds))
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='BFGS',
            **kwargs
        )
        return result

    proflogit = ProfLogitClassifier(optimize_fn=optimize)

