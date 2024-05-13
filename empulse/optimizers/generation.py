from datetime import datetime
from typing import Optional, Callable, Generator

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import OptimizeResult
from sklearn.utils import check_random_state


class Generation:
    """
    A single generation of a Real-coded Genetic Algorithm (RGA)

    Parameters
    ----------
    population_size : int or None, default=None
        Number of individuals in the population.
        If ``None``, population size is set to ``10 * n_dim``.

    crossover_rate : float, default=0.8
        Probability of crossover. Must be in [0, 1].

    mutation_rate : float, default=0.1
        Probability of mutation. Must be in [0, 1].

    elitism : float, default=0.05
        Fraction of the population that is considered elite.
        Must be in [0, 1].

    verbose : bool, default=False
        If ``True``, print status messages.

    logging_fn : callable, default=print
        Function to use for logging.

    random_state : int or None, default=None
        Random seed.

    n_jobs : int or None, default=1
        Number of jobs to run in parallel.
        If ``-1``, use all available processors.
        If ``None``, use 1 processor.

    Attributes
    ----------
    name : str
        Name of the optimizer.

    population : ndarray, shape (population_size, n_dim)
        Current population.

    population_size : int
        Number of individuals in the population.

    crossover_rate : float
        Probability of crossover.

    mutation_rate : float
        Probability of mutation.

    elitism : float
        Fraction of the population that is considered elite.

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

    def __init__(
            self,
            population_size: Optional[int] = None,
            crossover_rate: float = 0.8,
            mutation_rate: float = 0.1,
            elitism: float = 0.05,
            verbose: bool = False,
            logging_fn: Callable = print,
            random_state: Optional[int] = None,
            n_jobs: Optional[int] = 1,
    ):
        super().__init__()
        self.name = "RGA"
        self.population = []

        if population_size is not None:
            if not isinstance(population_size, int):
                raise TypeError("`pop_size` must be an int.")
            if population_size < 10:
                raise ValueError("`pop_size` must be >= 10.")
        self.population_size = population_size

        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("`crossover_rate` must be in [0, 1].")
        self.crossover_rate = crossover_rate

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("`mutation_rate` must be in [0, 1].")
        self.mutation_rate = mutation_rate

        if not 0.0 <= elitism <= 1.0:
            raise ValueError("`elitism` must be in [0, 1].")
        self.elitism = elitism

        self.verbose = verbose
        self.logging_fn = logging_fn

        # Get random state object
        self.rng = check_random_state(random_state)

        self.n_jobs = n_jobs

        # Attributes
        self._n_mating_pairs = None
        self.population = None
        self.elite_pool = None
        self.fx_best = []
        self.fitness = None
        self.result = OptimizeResult(success=False, nfev=0, nit=0, fun=np.inf, x=None)
        self.lower_bounds = None
        self.upper_bounds = None
        self.delta_bounds = None
        self.n_dim = None

    def optimize(self, objective: Callable, bounds: list[tuple[float, float]]) -> Generator['Generation', None, None]:
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
        """

        # Check bounds
        bounds = list(bounds)
        if not all(
                isinstance(t, tuple)
                and len(t) == 2
                and isinstance(t[0], (int, float))
                and isinstance(t[1], (int, float))
                for t in bounds
        ):
            raise ValueError("`bounds` must be a sequence of tuples of two numbers (lower_bound, upper_bound).")
        array_bounds = np.asarray(bounds, dtype=np.float64).T
        self.lower_bounds = array_bounds[0]
        self.upper_bounds = array_bounds[1]
        self.delta_bounds = np.fabs(self.upper_bounds - self.lower_bounds)
        self.n_dim = len(bounds)

        # Check population size
        if self.population_size is None:
            self.population_size = self.n_dim * 10

        self.elitism = int(max(1, round(self.population_size * self.elitism)))
        self._n_mating_pairs = int(self.population_size / 2)  # Constant for crossover
        self.fitness = np.empty(self.population_size) * np.nan

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

    def _generate_population(self):
        population = self.rng.rand(self.population_size, self.n_dim)
        return self.lower_bounds + population * self.delta_bounds

    def _evaluate(self, objective) -> bool:
        fitness_values = Parallel(n_jobs=self.n_jobs)(
            delayed(self._update_fitness)(objective, ix) for ix in range(self.population_size)
        )
        self.fitness = np.asarray(fitness_values)
        return False

    def _update_fitness(self, objective, index: int) -> float:
        fitness_value = float(self.fitness[index])
        if np.isnan(fitness_value):
            self.result.nfev += 1
            return objective(self.population[index])
        else:
            return fitness_value

    def _crossover(self):
        """Perform local arithmetic crossover"""
        # Make iterator for pairs
        match_parents = (
            rnd_pair for rnd_pair in self.rng.choice(self.population_size, (self._n_mating_pairs, 2), replace=False)
        )

        # Crossover parents
        for ix1, ix2 in match_parents:
            if self.rng.uniform() < self.crossover_rate:
                parent1 = self.population[ix1]  # Pass-by-ref
                parent2 = self.population[ix2]
                w = self.rng.uniform(size=self.n_dim)
                child1 = w * parent1 + (1 - w) * parent2
                child2 = w * parent2 + (1 - w) * parent1
                self.population[ix1] = child1
                self.population[ix2] = child2
                self.fitness[ix1] = np.nan
                self.fitness[ix2] = np.nan

    def _mutate(self):
        """Perform uniform random mutation"""
        for ix in range(self.population_size):
            if self.rng.uniform() < self.mutation_rate:
                mutant = self.population[ix]  # inplace
                rnd_gene = self.rng.choice(self.n_dim)
                rnd_val = self.rng.uniform(
                    low=self.lower_bounds[rnd_gene], high=self.upper_bounds[rnd_gene],
                )
                mutant[rnd_gene] = rnd_val
                self.fitness[ix] = np.nan

    def _select(self):
        """Perform linear scaling selection"""
        fitness_values = np.copy(self.fitness)
        min_fitness = np.min(fitness_values)
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)

        # Linear scaling
        if min_fitness < 0:
            fitness_values -= min_fitness
            min_fitness = 0
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
            relative_fitness = np.ones(self.population_size) / self.population_size  # Uniform distribution
        else:
            relative_fitness = scaled_fitness / normalization_factor

        # Select individuals
        select_ix = self.rng.choice(
            self.population_size, size=self.population_size, replace=True, p=relative_fitness,
        )
        self.population = self.population[select_ix]
        self.fitness = self.fitness[select_ix]

    def _get_sorted_non_nan_ix(self):
        """Get indices sorted according to non-nan fitness values"""
        non_nan_fx = (
            (ix, fx) for ix, fx in enumerate(self.fitness) if ~np.isnan(fx)
        )
        sorted_list = sorted(non_nan_fx, key=lambda t: t[1])
        return sorted_list

    def _insert_elites(self):
        """
        Update population by replacing the worst solutions of the current
        with the ones from the elite pool.
        """
        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            worst_ix = [t[0] for t in sorted_fx][: self.elitism]
        else:
            worst_ix = np.argsort(self.fitness)[: self.elitism]  # TODO: replace with argpartition
            # worst_ix = np.argpartition(self.fitness, self.elitism)[: self.elitism]
        for i, ix in enumerate(worst_ix):
            elite, fitness_elite = self.elite_pool[i]
            self.population[ix] = elite
            self.fitness[ix] = fitness_elite

    def _update_elite_pool(self):
        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            elite_ix = [t[0] for t in sorted_fx][-self.elitism:]
        else:
            elite_ix = np.argsort(self.fitness)[-self.elitism:]  # TODO: replace with argpartition
            # elite_ix = np.argpartition(self.fitness, -self.elitism)[-self.elitism:]
        self.elite_pool = [
            (self.population[ix].copy(), self.fitness[ix]) for ix in elite_ix
        ]
        # Append best solution
        self.fx_best.append(self.fitness[elite_ix[-1]])
        self.result.x = self.population[elite_ix[-1]]
        self.result.fun = self.fx_best[-1]
        self.result.nit = len(self.fx_best)

    def _log_start(self):
        self.logging_fn(
            "# ---  {} ({})  --- #".format(
                self.name,
                datetime.now().strftime("%a %b %d %H:%M:%S"),
            )
        )

    def _log_progress(self):
        status_msg = "Iter = {:5d}; nfev = {:6d}; fx = {:.4f}".format(
            self.result.nit, self.result.nfev, self.fx_best[-1],
        )
        self.logging_fn(status_msg)

    def _log_end(self, stop_time):
        self.logging_fn(self.result)
        self.logging_fn("# ---  {} ({})  --- #".format(self.name, stop_time))
