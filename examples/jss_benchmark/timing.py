"""Fit the benchmark models and return their measured training times in memory."""

from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd

from .data import LEGACY_MAX_N, MODELS, N_GRID, SEEDS, make_dataset
from .models import build_model


def run_timing(
    impl: str,
    n_grid: Iterable[int] = N_GRID,
    seeds: Iterable[int] = SEEDS,
    models: Iterable[str] = MODELS,
) -> pd.DataFrame:
    """Measure one fit per model, sample size and seed. Every call runs fresh fits."""
    rows = []
    seeds, models = tuple(seeds), tuple(models)
    for n_samples in n_grid:
        for model_name in models:
            if impl == 'legacy' and n_samples > LEGACY_MAX_N[model_name]:
                continue
            for seed in seeds:
                X, y, fn_cost, fp_cost = make_dataset(n_samples, seed)
                model = build_model(impl, model_name, fn_cost, fp_cost, seed)
                start = time.perf_counter()
                model.fit(X, y)
                elapsed = time.perf_counter() - start
                rows.append({
                    'impl': impl,
                    'model': model_name,
                    'n_samples': n_samples,
                    'seed': seed,
                    'fit_seconds': elapsed,
                })
                print(f'{impl:7s} {model_name:8s} N={n_samples:>9,} seed={seed}  {elapsed:9.3f}s', flush=True)
    return pd.DataFrame(rows, columns=['impl', 'model', 'n_samples', 'seed', 'fit_seconds'])
