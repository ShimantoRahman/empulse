from collections.abc import Callable
from itertools import islice
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from .._types import FloatNDArray
from ..metrics import LogitObjective
from ._base import Optimizer, objective_scale
from .generation import Generation, LamarckianGeneration


def _as_generation_fitness(logit_loss: Callable[[FloatNDArray], float]) -> Callable[[FloatNDArray], float]:
    """
    Adapt a minimization loss into the fitness :meth:`Generation.optimize` expects.

    ``LogitObjective.logit_loss`` is always a loss for *minimization*, while
    :class:`~empulse.optimizers.Generation` maximizes by construction (``direction =
    Direction.MAXIMIZE``): it selects and reports the *highest*-fitness individual. Negate here so
    the two conventions line up, rather than handing a minimization loss straight to a maximizer.
    """
    return lambda weights: -logit_loss(weights)


class GeneticAlgorithmOptimizer(Optimizer):
    """
    Real-coded Genetic Algorithm (RGA) optimizer for logit models.

    Uses :class:`~empulse.optimizers.Generation` under the hood with
    patience-based early stopping.  This is the default optimizer for
    :class:`~empulse.models.ProfLogitClassifier`.

    The objective function is evaluated as a *scalar* (via
    :meth:`~empulse.metrics.LogitObjective.logit_loss`) so that the GA can
    rank individuals without computing their gradients.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of GA generations.
    tolerance : float, default=1e-4
        Improvement of the loss below which the counter towards *patience* is incremented, relative to
        the scale of the cost matrix (the average cost of a wrong decision), so that it does not depend
        on the units of the costs.
    patience : int, default=250
        Number of consecutive generations with improvement < *tolerance* before stopping.
    bounds : tuple of (float, float), default=(-5, 5)
        Symmetric lower and upper bounds applied to every coefficient.
    population_size : int or None, default=None
        Number of individuals.  ``None`` uses :class:`~empulse.optimizers.Generation`'s
        default of ``max(10, 10 * n_features)``.
    crossover_rate : float, default=0.8
        Crossover probability.
    mutation_rate : float, default=0.1
        Mutation probability.
    elitism : float, default=0.05
        Fraction of best individuals carried over unchanged each generation.
        At least one individual is always carried over.
    random_state : int or None, default=None
        Seed for reproducibility.
    n_jobs : int, default=1
        Number of threads evaluating the fitness of the population in parallel.
    verbose : bool, default=False
        Print generation-level progress.

    Examples
    --------
    .. code-block:: python

        from empulse.models import ProfLogitClassifier
        from empulse.optimizers import GeneticAlgorithmOptimizer

        rga = GeneticAlgorithmOptimizer(max_iter=10, bounds=(-10, 10), population_size=30)
        model = ProfLogitClassifier(optimizer=rga)
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        patience: int = 250,
        bounds: tuple[float | int, float | int] = (-5, 5),
        population_size: int | None = None,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: float = 0.05,
        random_state: int | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.patience = patience
        self.bounds = bounds
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @property
    def requires_gradient(self) -> bool:
        """``False``: the genetic algorithm only ranks individuals by their loss."""
        return False

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Run the genetic algorithm and return the optimization result.

        Parameters
        ----------
        objective : :class:`~empulse.metrics.LogitObjective`
            Prepared objective exposing the loss and gradient of the logit model.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix, used only to size the coefficient vector.
        **kwargs : Any
            Forwarded to the underlying solver.

        Returns
        -------
        result : :class:`scipy.optimize.OptimizeResult`
            The optimization result; ``fun`` is reported as a loss to be minimized.
        """
        generation_kwargs: dict[str, Any] = {
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'elitism': self.elitism,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
        }
        if self.population_size is not None:
            generation_kwargs['population_size'] = self.population_size
        if self.random_state is not None:
            generation_kwargs['random_state'] = self.random_state

        rga = Generation(**generation_kwargs)
        bounds_per_feature = [self.bounds] * X.shape[1]

        # Generation.optimize() maximizes, but objective.logit_loss is a loss to minimize.
        fitness = _as_generation_fitness(objective.logit_loss)

        previous_loss: float | None = None
        iter_stagnant = 0
        # Improvements are measured against the cost matrix's scale rather than the loss itself, which
        # constants added to the costs would change without changing any decision.
        scale = objective_scale(objective)

        for _ in islice(rga.optimize(fitness, bounds_per_feature), self.max_iter):
            fitness_value = rga.result.fun  # type: ignore[attr-defined]
            loss = -fitness_value
            improvement = np.inf if previous_loss is None else (previous_loss - loss) / scale
            previous_loss = loss

            if improvement < self.tolerance:
                iter_stagnant += 1
                if iter_stagnant >= self.patience:
                    rga.result.message = 'Converged.'  # type: ignore[attr-defined]
                    rga.result.success = True  # type: ignore[attr-defined]
                    break
            else:
                iter_stagnant = 0
        else:
            rga.result.message = 'Maximum number of iterations reached.'  # type: ignore[attr-defined]
            rga.result.success = False  # type: ignore[attr-defined]

        result = rga.result
        # `fun` is reported as a loss, per the Optimizer contract.
        result.fun = -result.fun  # type: ignore[attr-defined]
        return result  # type: ignore[return-value]


class MemeticOptimizer(Optimizer):
    """
    Real-coded Lamarckian Memetic Algorithm optimizer for logit models.

    Combines a real-coded genetic algorithm (population-level diversity) with
    a gradient local search applied Lamarckian-style to every individual before
    its fitness is evaluated.  The gradient-refined weights overwrite the
    original genome so evolution always acts on already-locally-optimised solutions.

    The per-individual local search reuses the ROC convex hull across all
    ``local_steps`` gradient evaluations (see :class:`LamarckianGeneration`).
    Final fitness is always evaluated on a fresh hull via
    :meth:`~empulse.metrics.metric.strategies.max_profit_strategy.gradient_piecewise.MaxProfitLogitGradientPiecewise.score`.

    Parameters
    ----------
    bounds : tuple of (float, float), default=(-10.0, 10.0)
        Symmetric lower and upper bounds applied to every coefficient.
    population_size : int, default=50
        Number of individuals in the population.
    max_iter : int, default=100
        Maximum number of GA generations.
    patience : int, default=20
        Stop early when the best fitness has not improved by more than ``tol``
        over the last ``patience`` generations.
    tol : float, default=1e-6
        Convergence tolerance for the patience criterion, relative to the scale of the cost matrix
        (the average cost of a wrong decision).
    crossover_rate : float, default=0.8
        Crossover probability (passed to :class:`LamarckianGeneration`).
    mutation_rate : float, default=0.1
        Mutation probability (passed to :class:`LamarckianGeneration`).
    elitism : float, default=0.05
        Elite fraction (passed to :class:`LamarckianGeneration`).
        At least one individual is always carried over.
    local_steps : int, default=5
        Number of gradient steps per individual per generation.
    lr : float, default=0.05
        Learning rate for the local search.
    optimizer : {"adam", "sgd"}, default="adam"
        Local-search update rule passed to :class:`LamarckianGeneration`.
    beta1 : float, default=0.9
        Adam: exponential decay rate for the first moment estimate. Ignored when ``optimizer="sgd"``.
    beta2 : float, default=0.999
        Adam: exponential decay rate for the second moment estimate. Ignored when ``optimizer="sgd"``.
    eps : float, default=1e-8
        Adam: small term added to the denominator for numerical stability. Ignored when ``optimizer="sgd"``.
    grad_clip : float, default=5.0
        Gradient clipping threshold applied element-wise before the update.
    random_state : int, default=42
        Seed for the GA random-number generator.
    """

    def __init__(
        self,
        bounds: tuple[float | int, float | int] = (-10.0, 10.0),
        population_size: int = 50,
        max_iter: int = 100,
        patience: int = 20,
        tol: float = 1e-6,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: float = 0.05,
        local_steps: int = 5,
        lr: float = 0.05,
        optimizer: str = 'adam',
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 5.0,
        random_state: int = 42,
    ):
        if optimizer not in {'adam', 'sgd'}:
            raise ValueError(f"`optimizer` must be 'adam' or 'sgd', got {optimizer!r}.")
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.patience = patience
        self.tol = tol
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.local_steps = local_steps
        self.lr = lr
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.random_state = random_state

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **_: Any,
    ) -> OptimizeResult:
        """
        Run the Lamarckian memetic algorithm and return the optimization result.

        Parameters
        ----------
        objective : :class:`~empulse.metrics.LogitObjective`
            Prepared objective exposing the loss and gradient of the logit model.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix, used only to size the coefficient vector.
        **_ : Any
            Ignored; accepted for interface compatibility.

        Returns
        -------
        result : :class:`scipy.optimize.OptimizeResult`
            The optimization result; ``fun`` is reported as a loss to be minimized.
        """
        gen = LamarckianGeneration(
            grad_objective=objective,
            population_size=self.population_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            elitism=self.elitism,
            local_steps=self.local_steps,
            lr=self.lr,
            optimizer=self.optimizer,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            grad_clip=self.grad_clip,
            random_state=self.random_state,
            n_jobs=1,  # avoid nested parallelism when run inside joblib Parallel
        )

        bounds_list = [self.bounds] * X.shape[1]

        # Generation.optimize() maximizes, but objective.logit_loss is a loss to minimize. The Lamarckian
        # local search descends the true loss through `_grad_objective.logit_gradient_steps()`.
        fitness = _as_generation_fitness(objective.logit_loss)
        scale = objective_scale(objective)

        last_gen: Generation | None = None
        converged = False
        for i, last_gen in enumerate(gen.optimize(fitness, bounds_list)):
            if len(last_gen.fx_best) >= self.patience:
                recent = last_gen.fx_best[-self.patience :]
                if max(recent) - min(recent) < self.tol * scale:
                    converged = True
                    break
            if i + 1 >= self.max_iter:
                break

        # optimize() is an infinite generator, so last_gen is set after the first iteration.
        assert last_gen is not None
        ga_result = last_gen.result
        # Evaluate the final loss on the true hull; score() does not increment the epoch.
        loss = objective.logit_loss(ga_result.x)
        return OptimizeResult(  # type: ignore[call-arg]
            x=ga_result.x,
            success=converged,
            fun=float(loss),
            message='Converged.' if converged else 'Maximum number of iterations reached.',
            status=0 if converged else 1,
            nit=ga_result.nit,
            nfev=ga_result.nfev,
        )
