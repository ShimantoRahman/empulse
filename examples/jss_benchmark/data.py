"""Synthetic datasets for the training-time benchmark.

Both implementations are fitted on exactly the same data: 20 continuous features (15 informative, 5
redundant), a 15% positive rate, and false-negative and false-positive costs drawn independently per
instance from LogNormal(mu=3.0, sigma=0.8). True positives and true negatives cost nothing.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

N_GRID: list[int] = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
SEEDS: range = range(10)
MODELS: tuple[str, ...] = ('cslogit', 'cstree', 'csforest')
IMPLEMENTATIONS: tuple[str, ...] = ('legacy', 'current')
# Largest N each reference implementation is timed at; Empulse runs the full grid. CostCla's forest
# already takes about 7 minutes per fit at N = 10^5, its tree about 2 minutes at N = 3 x 10^5.
LEGACY_MAX_N: dict[str, int] = {'cslogit': 1_000_000, 'cstree': 300_000, 'csforest': 100_000}

N_FEATURES = 20
N_INFORMATIVE = 15
N_REDUNDANT = 5
POSITIVE_RATE = 0.15
COST_MU = 3.0
COST_SIGMA = 0.8


def make_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one benchmark dataset: features, labels, and the ``fn_cost`` and ``fp_cost`` arrays."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        weights=[1 - POSITIVE_RATE, POSITIVE_RATE],
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    fn_cost = rng.lognormal(mean=COST_MU, sigma=COST_SIGMA, size=n_samples)
    fp_cost = rng.lognormal(mean=COST_MU, sigma=COST_SIGMA, size=n_samples)
    return X, y, fn_cost, fp_cost
