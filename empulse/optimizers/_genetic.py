from collections.abc import Callable
from itertools import islice
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from .._types import FloatNDArray
from ..metrics import LogitObjective
from ..metrics.metric.common import Direction
from ._base import Optimizer
from .generation import Generation, LamarckianGeneration


def _as_generation_fitness(
    logit_loss: Callable[[FloatNDArray], float], direction: Direction
) -> Callable[[FloatNDArray], float]:
    """Adapt a minimization loss into whatever direction :meth:`Generation.optimize` expects.

    ``LogitObjective.logit_loss`` is always a loss for *minimization*, while
    :class:`~empulse.optimizers.Generation` (``direction = Direction.MAXIMIZE``) selects and
    reports the *highest*-fitness individual. Negate here so the two conventions line up, rather
    than handing a minimization loss straight to a maximizer.
    """
    if direction is Direction.MAXIMIZE:
        return lambda weights: -logit_loss(weights)
    return logit_loss


class GeneticAlgorithmOptimizer(Optimizer):
    """Real-coded Genetic Algorithm (RGA) optimizer for logit models.

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
        Relative improvement below which the counter towards *patience* is incremented.
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
    random_state : int or None, default=None
        Seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs for fitness evaluation.
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

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """Run the genetic algorithm."""
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

        # Generation.optimize() always maximizes; objective.logit_loss is a loss for minimization.
        fitness = _as_generation_fitness(objective.logit_loss, rga.direction)

        previous_loss: float | None = None
        iter_stagnant = 0

        for _ in islice(rga.optimize(fitness, bounds_per_feature), self.max_iter):
            fitness_value = rga.result.fun  # type: ignore[attr-defined]
            loss = -fitness_value if rga.direction is Direction.MAXIMIZE else fitness_value
            if previous_loss is not None:
                denominator = max(abs(previous_loss), 1e-12)
                relative_improvement = (previous_loss - loss) / denominator
            else:
                relative_improvement = np.inf
            previous_loss = loss

            if relative_improvement < self.tolerance:
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
        if rga.direction is Direction.MAXIMIZE:
            # Report `fun` as a loss, per the Optimizer contract (see Optimizer.__call__ docstring).
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
    bounds : float, default=10.0
        Symmetric search-space half-width: each coefficient is initialised in ``[-bounds, +bounds]``.
    population_size : int, default=50
        Number of individuals in the population.
    max_iter : int, default=100
        Maximum number of GA generations.
    patience : int, default=20
        Stop early when the best fitness has not improved by more than ``tol``
        over the last ``patience`` generations.
    tol : float, default=1e-6
        Convergence tolerance for the patience criterion.
    crossover_rate : float, default=0.8
        Crossover probability (passed to :class:`LamarckianGeneration`).
    mutation_rate : float, default=0.1
        Mutation probability (passed to :class:`LamarckianGeneration`).
    elitism : float, default=0.05
        Elite fraction (passed to :class:`LamarckianGeneration`).
    local_steps : int, default=5
        Number of gradient steps per individual per generation.
    lr : float, default=0.05
        Learning rate for the local search.
    optimizer : {"adam", "sgd"}, default="adam"
        Local-search update rule passed to :class:`LamarckianGeneration`.
    beta1 : float, default=0.9
    beta2 : float, default=0.999
    eps : float, default=1e-8
    grad_clip : float, default=5.0
        Adam / gradient-clipping hyper-parameters.
    random_state : int, default=42
        Seed for the GA random-number generator.
    """

    def __init__(
        self,
        bounds: float = 10.0,
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
        """Run the Lamarckian GA with the given objective function."""
        gen = LamarckianGeneration(
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
        # Wire up the gradient objective so the local search can call gradient_steps()
        gen._grad_objective = objective

        bounds_list = [(-self.bounds, self.bounds)] * X.shape[1]

        # Generation.optimize() always maximizes; objective.logit_loss is a loss for
        # minimization. Adapt once here rather than handing a minimization loss to a maximizer.
        # (The Lamarckian local search itself descends the true loss directly via
        # `_grad_objective.logit_gradient_steps()`, independent of this adapter, so both the
        # local search and the population-level GA now pull in the same direction.)
        fitness = _as_generation_fitness(objective.logit_loss, gen.direction)

        last_gen: Generation | None = None
        for i, last_gen in enumerate(gen.optimize(fitness, bounds_list)):
            if i + 1 >= self.max_iter:
                break
            if len(last_gen.fx_best) >= self.patience:
                recent = last_gen.fx_best[-self.patience :]
                if max(recent) - min(recent) < self.tol:
                    break

        # optimize() is an infinite generator so last_gen is always set after at least one iteration.
        assert last_gen is not None
        ga_result = last_gen.result
        # Final loss evaluation on true hull (score() does not increment epoch)
        loss = objective.logit_loss(ga_result.x)
        return OptimizeResult(  # type: ignore[call-arg]
            x=ga_result.x,
            success=True,
            fun=float(loss),
            message='Lamarckian Memetic finished',
            status=0,
            nit=ga_result.nit,
            nfev=ga_result.nfev,
        )
