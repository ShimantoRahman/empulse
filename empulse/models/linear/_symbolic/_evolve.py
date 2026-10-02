# Adapted from gplearn 0.4.3 (https://github.com/trevorstephens/gplearn), BSD-3-Clause,
# Copyright (c) 2015-2026 Trevor Stephens. See the license text in this package's ``__init__``.
"""The evolutionary search over expressions."""

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.utils.validation import check_random_state

from ...._types import FloatNDArray, IntNDArray
from ._program import Program, ProgramSpace

MAX_INT = np.iinfo(np.int32).max

# Scores a program's output on the given rows of the training data (all rows for ``None``) and
# returns the loss to minimize, or ``inf`` for an output that cannot be scored.
Fitness = Callable[[FloatNDArray, IntNDArray | None], float]

_ROW_SAMPLING_STREAM = 1
_TUNING_ROWS_STREAM = 2


@dataclass(frozen=True)
class SearchSettings:
    """Settings of the search that do not depend on the data."""

    max_iter: int
    population_size: int
    tournament_size: int
    crossover_rate: float
    mutate_subtree_rate: float
    hoist_rate: float
    mutate_point_rate: float
    parsimony_coefficient: float
    max_samples: float
    n_tuned_programs: int
    tuning_interval: int
    tuning_max_iter: int
    patience: int | None
    tolerance: float
    max_time: float | None
    n_jobs: int


class ParetoPoint(NamedTuple):
    """The shortest expression found to reach a loss that no shorter expression reached."""

    length: int
    loss: float
    program: Program


@dataclass
class EvolutionResult:
    """What a search found."""

    program: Program
    pareto_front: list[ParetoPoint]
    last_best: Program
    run_details: dict[str, list[float]]


def _draw_rows(seed: int, n_samples: int, size: int, stream: int) -> IntNDArray:
    rng = np.random.default_rng((int(seed), stream))
    return np.sort(rng.choice(n_samples, size=size, replace=False))


def _score(program: Program, X: FloatNDArray, fitness: Fitness, rows: IntNDArray | None) -> float:
    return fitness(program.execute(X if rows is None else X[rows]), rows)


def _score_chunk(
    programs_and_seeds: Sequence[tuple[Program, int]],
    X: FloatNDArray,
    fitness: Fitness,
    batch_size: int | None,
) -> list[float]:
    """Score programs, each on the random batch of rows its seed selects unless *batch_size* is ``None``."""
    losses = []
    for program, seed in programs_and_seeds:
        rows = None if batch_size is None else _draw_rows(seed, X.shape[0], batch_size, _ROW_SAMPLING_STREAM)
        losses.append(_score(program, X, fitness, rows))
    return losses


def _tune_program(
    program: Program, X: FloatNDArray, fitness: Fitness, rows: IntNDArray | None, max_iter: int
) -> tuple[Program, float]:
    """Tune the constants of a program with Nelder-Mead, keeping its structure. Never returns a worse program."""
    start = np.asarray(program.constants())

    def objective(constants: FloatNDArray) -> float:
        return _score(program.with_constants(constants), X, fitness, rows)

    baseline = objective(start)
    if not np.isfinite(baseline):
        return program, float(baseline)
    with np.errstate(invalid='ignore'):  # a simplex vertex whose expression cannot be scored has an infinite loss
        result = minimize(objective, start, method='Nelder-Mead', options={'maxiter': max_iter, 'maxfev': 2 * max_iter})
    if np.isfinite(result.fun) and result.fun < baseline:
        return program.with_constants(result.x), float(result.fun)
    return program, float(baseline)


def _tune_chunk(
    programs: Sequence[Program], X: FloatNDArray, fitness: Fitness, rows: IntNDArray | None, max_iter: int
) -> list[tuple[Program, float]]:
    return [_tune_program(program, X, fitness, rows, max_iter) for program in programs]


def _in_chunks(function: Callable[..., list[Any]], n_jobs: int, items: Sequence[Any], *args: Any) -> list[Any]:
    """Apply *function* to contiguous chunks of *items*, in parallel when ``n_jobs > 1``."""
    if n_jobs == 1 or len(items) < 2:
        return function(items, *args)
    bounds = np.linspace(0, len(items), min(n_jobs, len(items)) + 1).astype(int)
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(function)(items[start:stop], *args) for start, stop in itertools.pairwise(bounds)
    )
    return list(itertools.chain.from_iterable(chunks))


def _record(archive: dict[int, tuple[float, Program]], program: Program, loss: float) -> None:
    """Keep the program if it has the lowest loss seen at its length."""
    if not np.isfinite(loss):
        return
    held = archive.get(program.length_)
    if held is None or loss < held[0]:
        archive[program.length_] = (loss, program)


def _pareto_front(archive: dict[int, tuple[float, Program]]) -> list[ParetoPoint]:
    front: list[ParetoPoint] = []
    best = np.inf
    for length in sorted(archive):
        loss, program = archive[length]
        if loss < best:
            front.append(ParetoPoint(length, loss, program))
            best = loss
    return front


def _tournament(
    parents: Sequence[Program], parent_fitness: FloatNDArray, size: int, random_state: np.random.RandomState
) -> Program:
    """Return the fittest of *size* parents drawn at random."""
    contenders = random_state.randint(0, len(parents), size)
    return parents[int(contenders[np.argmin(parent_fitness[contenders])])]


def _breed(
    parents: Sequence[Program] | None,
    parent_fitness: FloatNDArray | None,
    seeds: IntNDArray,
    space: ProgramSpace,
    settings: SearchSettings,
) -> list[Program]:
    """
    Create a generation: a random one without *parents*, otherwise one bred from them by tournament selection.

    Every program draws its randomness from its own seed, so the result does not depend on how the
    programs are later divided over workers.
    """
    thresholds = np.cumsum([
        settings.crossover_rate,
        settings.mutate_subtree_rate,
        settings.hoist_rate,
        settings.mutate_point_rate,
    ])
    population = []
    # Reseeding one generator gives the stream of a new one, which is about 70 times as expensive to create.
    random_state = np.random.RandomState()
    for seed in seeds:
        random_state.seed(seed)
        if parents is None or parent_fitness is None:
            population.append(Program(space.build(random_state)))
            continue

        method = random_state.uniform()
        parent = _tournament(parents, parent_fitness, settings.tournament_size, random_state)
        if method < thresholds[0]:
            donor = _tournament(parents, parent_fitness, settings.tournament_size, random_state)
            nodes = space.crossover(parent.nodes, donor.nodes, random_state)
        elif method < thresholds[1]:
            nodes = space.subtree_mutation(parent.nodes, random_state)
        elif method < thresholds[2]:
            nodes = space.hoist_mutation(parent.nodes, random_state)
        elif method < thresholds[3]:
            nodes = space.point_mutation(parent.nodes, random_state)
        else:
            nodes = list(parent.nodes)
        if space.max_length is not None and len(nodes) > space.max_length:
            nodes = list(parent.nodes)
        population.append(Program(nodes))
    return population


def _score_population(
    population: list[Program],
    seeds: IntNDArray,
    X: FloatNDArray,
    fitness: Fitness,
    batch_size: int | None,
    cache: dict[tuple[object, ...], float],
    archive: dict[int, tuple[float, Program]],
    n_jobs: int,
) -> tuple[FloatNDArray, int]:
    """
    Score every program of a generation and return the losses and the number of programs actually scored.

    Without batches a program always scores the same, so an expression seen before in the run, or earlier
    in the generation, is scored once. With batches every program is scored on rows of its own.
    """
    losses = np.empty(len(population))
    if batch_size is not None:
        scored = _in_chunks(_score_chunk, n_jobs, list(zip(population, seeds, strict=True)), X, fitness, batch_size)
        losses[:] = scored
        for program, loss in zip(population, scored, strict=True):
            _record(archive, program, loss)
        return losses, len(population)

    pending: dict[tuple[object, ...], list[int]] = {}
    for i, program in enumerate(population):
        key = program.key
        if key in cache:
            losses[i] = cache[key]
        else:
            pending.setdefault(key, []).append(i)
    unique = [population[indices[0]] for indices in pending.values()]
    scored = _in_chunks(_score_chunk, n_jobs, [(program, 0) for program in unique], X, fitness, None)
    for (key, indices), program, loss in zip(pending.items(), unique, scored, strict=True):
        cache[key] = loss
        losses[indices] = loss
        _record(archive, program, loss)
    return losses, len(unique)


def evolve(
    X: FloatNDArray,
    fitness: Fitness,
    space: ProgramSpace,
    settings: SearchSettings,
    random_state: int | np.random.RandomState | None,
) -> EvolutionResult:
    """
    Evolve expressions that minimize *fitness* on the rows of *X*.

    Each generation is bred from the previous one, scored and, every ``tuning_interval`` generations,
    has the constants of its best expressions tuned. The search keeps the best expression found at
    every length over the whole run, which is the Pareto front of loss against length.

    Parameters
    ----------
    X : 2D numpy.ndarray, shape=(n_samples, n_features)
        Training features.

    fitness : callable
        Loss of an expression's output on given rows of the training data. It must return ``inf`` for
        an output that cannot be scored.

    space : ProgramSpace
        The expressions the search may create.

    settings : SearchSettings
        Settings of the search.

    random_state : int, RandomState or None
        Controls the randomness of the search.

    Returns
    -------
    result : EvolutionResult
        The program with the lowest regularised loss on the Pareto front, the front itself, the best
        program of the last generation, and a log of every generation.
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]
    batch_size = None if settings.max_samples >= 1.0 else max(1, int(np.ceil(settings.max_samples * n_samples)))
    if batch_size is not None and batch_size >= n_samples:
        batch_size = None
    cache: dict[tuple[object, ...], float] = {}
    archive: dict[int, tuple[float, Program]] = {}
    run_details: dict[str, list[float]] = {
        name: []
        for name in (
            'generation',
            'average_length',
            'average_fitness',
            'best_length',
            'best_fitness',
            'best_oob_fitness',
            'generation_time',
            'n_evaluated',
        )
    }

    start_time = time.perf_counter()
    parents: list[Program] | None = None
    parent_fitness: FloatNDArray | None = None
    best_so_far = np.inf
    stagnant = 0

    for generation in range(settings.max_iter):
        generation_start = time.perf_counter()
        seeds = random_state.randint(MAX_INT, size=settings.population_size)
        population = _breed(parents, parent_fitness, seeds, space, settings)

        losses, n_evaluated = _score_population(
            population, seeds, X, fitness, batch_size, cache, archive, settings.n_jobs
        )

        if settings.n_tuned_programs > 0 and (generation + 1) % settings.tuning_interval == 0:
            _tune_population(population, losses, X, fitness, settings, cache, archive, batch_size, seeds)

        lengths = np.array([program.length_ for program in population])
        penalised = losses + settings.parsimony_coefficient * lengths
        parents, parent_fitness = population, penalised

        best = int(np.argmin(losses))
        last_best = population[best]
        best_oob = np.nan
        if batch_size is not None:
            rows = _draw_rows(seeds[best], n_samples, batch_size, _ROW_SAMPLING_STREAM)
            held_out = np.setdiff1d(np.arange(n_samples), rows, assume_unique=True)
            best_oob = _score(last_best, X, fitness, held_out)
        run_details['generation'].append(generation)
        run_details['average_length'].append(float(np.mean(lengths)))
        run_details['average_fitness'].append(float(np.mean(losses)))
        run_details['best_length'].append(int(lengths[best]))
        run_details['best_fitness'].append(float(losses[best]))
        run_details['best_oob_fitness'].append(best_oob)
        run_details['generation_time'].append(time.perf_counter() - generation_start)
        run_details['n_evaluated'].append(n_evaluated)

        if settings.patience is not None:
            improvement = _relative_improvement(best_so_far, float(losses[best]))
            best_so_far = min(best_so_far, float(losses[best]))
            stagnant = stagnant + 1 if improvement < settings.tolerance else 0
            if stagnant >= settings.patience:
                break
        if settings.max_time is not None and time.perf_counter() - start_time >= settings.max_time:
            break

    if batch_size is not None:
        # Losses on small random batches are noisy and optimistic for the best of many; score the front on all rows.
        rescored: dict[int, tuple[float, Program]] = {}
        for _, program in archive.values():
            _record(rescored, program, _score(program, X, fitness, None))
        archive = rescored
    front = _pareto_front(archive)
    if front:
        chosen = min(front, key=lambda point: point.loss + settings.parsimony_coefficient * point.length).program
    else:
        chosen = last_best
    return EvolutionResult(program=chosen, pareto_front=front, last_best=last_best, run_details=run_details)


def _relative_improvement(previous: float, current: float) -> float:
    if not np.isfinite(previous):
        return np.inf if np.isfinite(current) else 0.0
    return (previous - current) / max(abs(previous), 1e-12)


def _tune_population(
    population: list[Program],
    losses: FloatNDArray,
    X: FloatNDArray,
    fitness: Fitness,
    settings: SearchSettings,
    cache: dict[tuple[object, ...], float],
    archive: dict[int, tuple[float, Program]],
    batch_size: int | None,
    seeds: IntNDArray,
) -> None:
    """Tune the constants of the best distinct expressions that have any; tuned copies replace the originals."""
    chosen: dict[tuple[object, ...], Program] = {}
    for i in np.argsort(losses, kind='stable'):
        if len(chosen) >= settings.n_tuned_programs or not np.isfinite(losses[i]):
            break
        program = population[i]
        if program.key not in chosen and program.constants():
            chosen[program.key] = program
    if not chosen:
        return

    rows = None if batch_size is None else _draw_rows(seeds[0], X.shape[0], batch_size, _TUNING_ROWS_STREAM)
    tuned = _in_chunks(_tune_chunk, settings.n_jobs, list(chosen.values()), X, fitness, rows, settings.tuning_max_iter)
    replacements: dict[tuple[object, ...], tuple[Program, float]] = {}
    for (key, original), (program, loss) in zip(chosen.items(), tuned, strict=True):
        if program is original:
            continue
        replacements[key] = (program, loss)
        if batch_size is None:
            cache[program.key] = loss
        _record(archive, program, loss)
    for i, program in enumerate(population):
        replacement = replacements.get(program.key)
        if replacement is not None:
            population[i], losses[i] = replacement
