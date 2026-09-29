"""Training-time sweep: fit every (model, N, seed) combination once and record the wall-clock time.

    python -m jss_benchmark.timing --impl current
    python -m jss_benchmark.timing --impl legacy [--max-n N] [--quick]

Rows are appended to ``results/timing_<impl>.csv`` as each fit completes, and combinations already
in the file are skipped, so the sweep can be interrupted and resumed. Each reference implementation
is run up to its own maximum size, as in the paper (``LEGACY_MAX_N``: 10^6 for CSLogit, 3 x 10^5
for CSTree, 10^5 for CSForest); ``--max-n`` lowers that limit further.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterable
from pathlib import Path

from .data import LEGACY_MAX_N, MODELS, N_GRID, SEEDS, make_dataset
from .models import build_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
COLUMNS = ('impl', 'model', 'n_samples', 'seed', 'fit_seconds')


def results_path(impl: str, results_dir: Path = RESULTS_DIR) -> Path:
    """The CSV the timing sweep for ``impl`` writes to."""
    return results_dir / f'timing_{impl}.csv'


def run_timing(
    impl: str,
    n_grid: Iterable[int] = N_GRID,
    seeds: Iterable[int] = SEEDS,
    models: Iterable[str] = MODELS,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Time ``fit`` for every missing (model, N, seed) combination of ``impl``; return the CSV path."""
    path = results_path(impl, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with path.open(encoding='utf-8', newline='') as f:
            done = {(row['model'], int(row['n_samples']), int(row['seed'])) for row in csv.DictReader(f)}
    write_header = not path.exists()
    seeds = list(seeds)
    models = list(models)

    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for n_samples in n_grid:
            for model_name in models:
                for seed in seeds:
                    if (model_name, n_samples, seed) in done:
                        continue
                    if impl == 'legacy' and n_samples > LEGACY_MAX_N[model_name]:
                        continue
                    X, y, fn_cost, fp_cost = make_dataset(n_samples, seed)
                    model = build_model(impl, model_name, fn_cost, fp_cost, seed)

                    start = time.perf_counter()
                    model.fit(X, y)
                    fit_seconds = time.perf_counter() - start

                    writer.writerow({
                        'impl': impl,
                        'model': model_name,
                        'n_samples': n_samples,
                        'seed': seed,
                        'fit_seconds': fit_seconds,
                    })
                    f.flush()
                    print(f'{impl:7s} {model_name:8s} N={n_samples:>9,} seed={seed}  {fit_seconds:9.3f}s', flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--impl', choices=['current', 'legacy'], required=True)
    parser.add_argument('--max-n', type=int, default=None, help='Largest sample size to run.')
    parser.add_argument('--quick', action='store_true', help='Only the two smallest sizes and two seeds.')
    args = parser.parse_args()
    n_grid = [n for n in N_GRID if n <= (args.max_n or max(N_GRID))]
    seeds = list(SEEDS)
    if args.quick:
        n_grid, seeds = n_grid[:2], seeds[:2]
    run_timing(args.impl, n_grid, seeds)


if __name__ == '__main__':
    main()
