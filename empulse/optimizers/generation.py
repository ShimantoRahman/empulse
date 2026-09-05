from collections.abc import Callable, Generator, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from sklearn.utils import check_random_state

from ..metrics.metric.common import Direction

if TYPE_CHECKING:
    from ..metrics import LogitObjective

MIN_POP_SIZE: int = 10
FEATURE_TO_POP_SIZE_RATIO: int = 10


class Generation:
    """
    A single generation of a Real-coded Genetic Algorithm (RGA).

    :meth:`optimize` always **maximizes** the objective function passed to it.
    Read more in the :ref:`User Guide <proflogit>`.

    Parameters
    ----------
    population_size : int or None, default=None
        Number of individuals in the population.
        If ``None``, population size is set to ``10 * n_features``.

    crossover_rate : float, default=0.8
        Probability of crossover. Must be in [0, 1].

    mutation_rate : float, default=0.1
        Probability of mutation. Must be in [0, 1].

    elitism : float, default=0.05
        Fraction of the population that transferred to the next generation without change.
        Must be in [0, 1].

    verbose : bool, default=False
        If ``True``, print status messages.

    logging_fn : callable, default=print
        Function to use for logging.

    random_state : int, RandomState or None, default=None
        Random seed.  Accepts an ``int``, a ``numpy.random.RandomState`` instance,
        or ``None`` (uses the global NumPy random state).

    n_jobs : int or None, default=1
        Number of jobs to run in parallel.
        If ``-1``, use all available processors.
        If ``None``, use 1 processor.

    Attributes
    ----------
    name : str
        Name of the optimizer.

    direction : Direction
        Optimization direction, always ``Direction.MAXIMIZE``. Callers handing this class a loss
        (which is minimized by convention) must negate it first.

    population : ndarray, shape (population_size, n_dim)
        Current population.

    population_size : int or None
        The *population_size* constructor argument, unchanged by :meth:`optimize`. ``None`` means
        the actual size used each run is resolved from ``10 * n_features`` and kept internally.

    crossover_rate : float
        Probability of crossover.

    mutation_rate : float
        Probability of mutation.

    elitism : int
        The number of individuals of the population that are transferred to the next generation
        without change. Set to ``0`` until :meth:`optimize` resolves it from *elitism_fraction*
        once the population size is known.

    elitism_fraction : float
        The *elitism* constructor argument, stored under its own name since *elitism* itself is
        repurposed as the resolved individual count above.

    verbose : bool
        If ``True``, print status messages.

    logging_fn : callable
        Function to use for logging.

    rng : RandomState
        Random state object.

    n_jobs : int
        Number of jobs to run in parallel.
        If ``-1``, use all available processors.
        If ``None``, use 1 processor.

    fx_best : list
        List of best fitness values.

    fitness : ndarray, shape (population_size,)
        Fitness values of the current population.

    result : OptimizeResult
        Result of the optimization.

    lower_bounds : ndarray, shape (n_dim,)
        Lower bounds of the search space.

    upper_bounds : ndarray, shape (n_dim,)
        Upper bounds of the search space.

    delta_bounds : ndarray, shape (n_dim,)
        Difference between upper and lower bounds.

    n_dim : int
        Number of dimensions.

    _n_mating_pairs : int
        Number of mating pairs.

    elite_pool : list
        List of elite individuals.
    """

    direction: ClassVar[Direction] = Direction.MAXIMIZE

    def __init__(
        self,
        population_size: int | None = None,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: float = 0.05,
        verbose: bool = False,
        logging_fn: Callable[[str], None] = print,
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int | None = 1,
    ):
        super().__init__()
        self.name = 'RGA'
        self.population: NDArray[np.float64] = np.empty(0)

        if population_size is not None:
            if not isinstance(population_size, int):
                raise TypeError('`population_size` must be an int.')
            if population_size < MIN_POP_SIZE:
                raise ValueError(f'`population_size` must be >= {MIN_POP_SIZE}, got {population_size}.')
        # None is stored as-is; the actual size is resolved in optimize() once n_dim is known.
        self.population_size: int | None = population_size

        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError('`crossover_rate` must be in [0, 1].')
        self.crossover_rate = crossover_rate

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError('`mutation_rate` must be in [0, 1].')
        self.mutation_rate = mutation_rate

        if not 0.0 <= elitism <= 1.0:
            raise ValueError('`elitism` must be in [0, 1].')
        self.elitism_fraction: float = elitism
        self.elitism = 0  # determined later

        self.verbose = verbose
        self.logging_fn = logging_fn

        # Get random state object
        self.rng = check_random_state(random_state)

        self.n_jobs = n_jobs

        # Attributes
        self._pop_size_: int | None = None
        self._n_mating_pairs: int | None = None
        self.elite_pool: list[tuple[NDArray[np.float64], np.float64]] = []  # individual, fitness
        self.fx_best: list[np.float64] = []
        self.fitness: NDArray[np.float64] = np.empty(0)
        self.result = OptimizeResult(success=False, nfev=0, nit=0, fun=np.inf, x=None)  # type: ignore[call-arg]
        self.lower_bounds: NDArray[np.float64] = np.asarray(0.0)
        self.upper_bounds: NDArray[np.float64] = np.asarray(0.0)
        self.delta_bounds: NDArray[np.float64] = np.asarray(0.0)
        self.n_dim: int = 0

    def optimize(
        self, objective: Callable[[NDArray[np.float64]], float], bounds: list[tuple[float, float]]
    ) -> Generator['Generation', None, None]:
        """
        Optimize the objective function.

        Parameters
        ----------
        objective : Callable
            Objective function to optimize.
            Should be of signature ``objective(weights) -> float``.
        bounds : list[tuple[float, float]]
            List of tuples of lower and upper bounds for each weight.

        Yields
        ------
        self : Generation
            Current instance of the optimizer.

        Notes
        -----
        This is an **infinite generator**.  The caller is responsible for
        stopping iteration (e.g. via ``break`` or ``itertools.islice``).
        Calling ``optimize`` on the same instance a second time resets all
        accumulated state (``fx_best``, ``result``, ``elite_pool``, etc.).
        """
        # Reset state so that calling optimize() twice gives a clean run.
        self.fx_best = []
        self.elite_pool = []
        self.result = OptimizeResult(success=False, nfev=0, nit=0, fun=np.inf, x=None)  # type: ignore[call-arg]

        # Check bounds
        bounds = list(bounds)
        if not all(
            isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], int | float) and isinstance(t[1], int | float)
            for t in bounds
        ):
            raise ValueError('`bounds` must be a sequence of tuples of two numbers (lower_bound, upper_bound).')
        array_bounds: NDArray[np.float64] = np.asarray(bounds, dtype=np.float64).T
        self.lower_bounds = array_bounds[0]
        self.upper_bounds = array_bounds[1]
        self.delta_bounds = np.fabs(self.upper_bounds - self.lower_bounds)
        self.n_dim = len(bounds)

        # Resolve population size now that n_dim is known. Stored separately from
        # `population_size` (left untouched at whatever the constructor received) so a second
        # `optimize()` call on a different-dimensional problem re-resolves it instead of reusing
        # a stale size from the first call.
        self._pop_size_ = (
            self.population_size if self.population_size is not None else self.n_dim * FEATURE_TO_POP_SIZE_RATIO
        )

        self.elitism = int(max(1, round(self._pop_size * self.elitism_fraction)))
        self._n_mating_pairs = self._pop_size // 2
        self.fitness = np.full(self._pop_size, np.nan)

        self.population = self._generate_population()
        self._evaluate(objective)
        self._update_elite_pool()

        if self.verbose:
            self._log_start()

        while True:
            yield self
            if self.verbose:
                self._log_progress()
            self._select()
            self._crossover()
            self._mutate()
            self._evaluate(objective)
            self._insert_elites()  # survivor selection: overlapping-generation model
            self._update_elite_pool()

    def _generate_population(self) -> NDArray[np.float64]:
        population = self.rng.rand(self._pop_size, self.n_dim)
        return self.lower_bounds + population * self.delta_bounds  # type: ignore

    def _evaluate(self, objective: Callable[[NDArray[np.float64]], float]) -> None:
        nan_mask = np.isnan(self.fitness)
        fitness_values = Parallel(n_jobs=self.n_jobs)(
            delayed(self._update_fitness)(objective, ix) for ix in range(self._pop_size)
        )
        self.fitness = np.asarray(fitness_values)
        # Count re-evaluated individuals after the (possibly parallel) call to
        # avoid the race condition that arises from incrementing inside workers.
        self.result.nfev += int(nan_mask.sum())  # type: ignore[attr-defined]

    def _update_fitness(self, objective: Callable[[NDArray[np.float64]], float], index: int) -> float:
        fitness_value = float(self.fitness[index])
        if np.isnan(fitness_value):
            return objective(self.population[index])
        else:
            return fitness_value

    def _crossover(self) -> None:
        """Perform local arithmetic crossover."""
        for ix1, ix2 in self.rng.choice(self._pop_size, (self._n_pairs, 2), replace=False):
            if self.rng.uniform() < self.crossover_rate:
                parent1 = self.population[ix1]
                parent2 = self.population[ix2]
                w = self.rng.uniform(size=self.n_dim)
                child1 = w * parent1 + (1 - w) * parent2
                child2 = w * parent2 + (1 - w) * parent1
                self.population[ix1] = child1
                self.population[ix2] = child2
                self.fitness[ix1] = np.nan
                self.fitness[ix2] = np.nan

    def _mutate(self) -> None:
        """Perform uniform random mutation."""
        for ix in range(self._pop_size):
            if self.rng.uniform() < self.mutation_rate:
                mutant = self.population[ix]  # view — mutation writes through to self.population
                rnd_gene = self.rng.choice(self.n_dim)
                rnd_val = self.rng.uniform(
                    low=self.lower_bounds[rnd_gene],
                    high=self.upper_bounds[rnd_gene],
                )
                mutant[rnd_gene] = rnd_val
                self.fitness[ix] = np.nan

    def _select(self) -> None:
        """Perform linear scaling selection."""
        fitness_values = np.copy(self.fitness)
        min_fitness = float(np.min(fitness_values))
        avg_fitness = float(np.mean(fitness_values))
        max_fitness = float(np.max(fitness_values))

        # Shift all values above zero before applying linear scaling so that
        # the scaling formula always operates on non-negative inputs.
        if min_fitness < 0:
            fitness_values -= min_fitness
            avg_fitness -= min_fitness
            max_fitness -= min_fitness
            min_fitness = 0.0

        # Linear scaling
        if min_fitness > (2 * avg_fitness - max_fitness):
            denominator = max_fitness - avg_fitness
            a = avg_fitness / (denominator if denominator != 0 else 1e-10)
            b = a * (max_fitness - 2 * avg_fitness)
        else:
            denominator = avg_fitness - min_fitness
            a = avg_fitness / (denominator if denominator != 0 else 1e-10)
            b = -min_fitness * a
        scaled_fitness = np.abs(a * fitness_values + b)

        # Normalize
        if (normalization_factor := np.sum(scaled_fitness)) == 0:
            relative_fitness = np.ones(self._pop_size) / self._pop_size  # Uniform distribution
        else:
            relative_fitness = scaled_fitness / normalization_factor

        # Select individuals
        select_ix = self.rng.choice(
            self._pop_size,
            size=self._pop_size,
            replace=True,
            p=relative_fitness,
        )
        self.population = self.population[select_ix]
        self.fitness = self.fitness[select_ix]

    def _get_sorted_non_nan_ix(self) -> list[tuple[int, float]]:
        """Get indices sorted according to non-nan fitness values."""
        non_nan_fx = ((ix, fx) for ix, fx in enumerate(self.fitness) if ~np.isnan(fx))
        sorted_list = sorted(non_nan_fx, key=lambda t: t[1])
        return sorted_list

    def _insert_elites(self) -> None:
        """Update population by replacing the worst solutions of the current with the ones from the elite pool."""
        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            worst_ix: Iterable[int] = [t[0] for t in sorted_fx][: self.elitism]
        else:
            worst_ix = np.argsort(self.fitness)[: self.elitism]
        for i, ix in enumerate(worst_ix):
            elite, fitness_elite = self.elite_pool[i]
            self.population[ix] = elite
            self.fitness[ix] = fitness_elite

    def _update_elite_pool(self) -> None:
        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            elite_ix: Sequence[int] = [t[0] for t in sorted_fx][-self.elitism :]
        else:
            elite_ix = list(np.argsort(self.fitness)[-self.elitism :])
        self.elite_pool = [(self.population[ix].copy(), self.fitness[ix]) for ix in elite_ix]
        # Append best solution
        self.fx_best.append(self.fitness[elite_ix[-1]])
        # Store a copy so that subsequent population mutations don't corrupt result.x
        self.result.x = self.population[elite_ix[-1]].copy()  # type: ignore[attr-defined]
        self.result.fun = self.fx_best[-1]  # type: ignore[attr-defined]
        self.result.nit = len(self.fx_best)  # type: ignore[attr-defined]

    def _log_start(self) -> None:
        self.logging_fn(
            '# ---  {} ({})  --- #'.format(
                self.name,
                datetime.now().strftime('%a %b %d %H:%M:%S'),  # noqa: DTZ005
            )
        )

    def _log_progress(self) -> None:
        status_msg = f'Iter = {self.result.nit:5d}; nfev = {self.result.nfev:6d}; fx = {self.fx_best[-1]:.4f}'
        self.logging_fn(status_msg)

    # ------------------------------------------------------------------
    # Narrowing accessors
    # Private methods are only ever called from optimize(), which always
    # resolves both attributes to int before any of them are invoked.
    # cast() lets mypy see the non-optional type without scattering
    # assert-not-None guards across every method.
    # ------------------------------------------------------------------

    @property
    def _pop_size(self) -> int:
        """Resolved population size as int — valid only after optimize() has been called."""
        return cast('int', self._pop_size_)

    @property
    def _n_pairs(self) -> int:
        """_n_mating_pairs as int — valid only after optimize() has been called."""
        return cast('int', self._n_mating_pairs)


class LamarckianGeneration(Generation):
    """Real-coded GA generation with Lamarckian local gradient search.

    Before evaluating each individual's fitness the genome is improved in-place
    by ``local_steps`` gradient steps (Lamarckian learning: the refined weights
    replace the original ones).  This lets the GA operate on a much smoother
    fitness landscape while the population still maintains global diversity
    across the rugged high-alpha MaxProfit surface.

    The convex hull required by the gradient objective is computed **once** per
    individual at the start of the local search and then cached across all
    ``local_steps`` gradient evaluations via the
    :meth:`~empulse.metrics.metric.strategies.max_profit_strategy.gradient_piecewise.MaxProfitLogitGradientPiecewise.gradient_steps`
    generator.  This cuts hull-reconstruction overhead by a factor of
    ``local_steps``.

    Parameters
    ----------
    local_steps : int, default=5
        Number of gradient steps applied to each individual per generation.
    lr : float, default=0.05
        Learning rate (step size) for the local search.
    optimizer : {"adam", "sgd"}, default="adam"
        Local-search update rule.

        * ``"adam"`` – adaptive moment estimation (recommended for noisy
          gradients; uses ``beta1``, ``beta2``, and ``eps``).
        * ``"sgd"`` – plain gradient descent (``theta -= lr * grad``).
    beta1 : float, default=0.9
        Adam: exponential decay rate for the first moment estimate.
        Ignored when ``optimizer="sgd"``.
    beta2 : float, default=0.999
        Adam: exponential decay rate for the second moment estimate.
        Ignored when ``optimizer="sgd"``.
    eps : float, default=1e-8
        Adam: small term added to the denominator for numerical stability.
        Ignored when ``optimizer="sgd"``.
    grad_clip : float, default=5.0
        Gradient clipping threshold applied element-wise before the update.
    grad_objective : LogitObjective
        Objective providing the gradient steps for the local search
        (:meth:`~empulse.metrics.LogitObjective.logit_gradient_steps`).
    **kwargs
        Forwarded to :class:`Generation`.
    """

    def __init__(
        self,
        grad_objective: 'LogitObjective',
        local_steps: int = 5,
        lr: float = 0.05,
        optimizer: str = 'adam',
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if optimizer not in {'adam', 'sgd'}:
            raise ValueError(f"`optimizer` must be 'adam' or 'sgd', got {optimizer!r}.")
        self.local_steps = local_steps
        self.lr = lr
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self._grad_objective = grad_objective

    def _local_search(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run ``local_steps`` gradient steps on *theta*.

        The convex hull is built once from the initial *theta* and then reused
        for all subsequent steps via the ``gradient_steps`` generator.  The
        update rule is selected by ``self.optimizer``.  After the local search
        the result is clipped to the search bounds.
        """
        theta = theta.copy()

        # Adam moment buffers (only used when optimizer == "adam")
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)

        grad_gen = self._grad_objective.logit_gradient_steps()

        for t in range(1, self.local_steps + 1):
            grad = grad_gen.send(theta)
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

            if self.optimizer == 'sgd':
                theta = theta - self.lr * grad
            else:  # adam
                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * grad**2
                m_hat = m / (1.0 - self.beta1**t)
                v_hat = v / (1.0 - self.beta2**t)
                theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            theta = np.clip(theta, self.lower_bounds, self.upper_bounds)

        grad_gen.close()
        return theta

    def _evaluate(self, objective: Callable[[NDArray[np.float64]], float]) -> None:  # type: ignore[override]
        """Apply Lamarckian local search before scalar evaluation.

        Runs sequentially (no joblib) to avoid pickling the gradient objective.
        """
        for ix in range(self._pop_size):
            if np.isnan(self.fitness[ix]):
                self.population[ix] = self._local_search(self.population[ix])
                self.fitness[ix] = float(objective(self.population[ix]))
                self.result.nfev += 1  # type: ignore[attr-defined]
