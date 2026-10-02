"""
Time the package's known hot paths, for before/after comparisons of performance work.

Run with ``just bench`` (or ``uv run python scripts/benchmark.py``). Each case reports the median
wall time over its repeats. Pass case names to run a subset: ``just bench proflogit_mpc clone``.
"""

import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification

warnings.filterwarnings('ignore')


def _data(n_samples: int, n_features: int = 15) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(n_samples=n_samples, n_features=n_features, weights=[0.8], random_state=0)


def _import_models() -> None:
    code = 'import time; t = time.perf_counter(); import empulse.models; print(time.perf_counter() - t)'
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


def _cases() -> dict[str, tuple[Callable[[], object], int]]:
    from empulse.metrics import empc_score, mpc_score
    from empulse.models import CSLogitClassifier, CSTreeClassifier, ProfLogitClassifier, ProfSRClassifier
    from empulse.optimizers import GeneticAlgorithmOptimizer

    X, y = _data(10_000)
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100_000)
    y_score = rng.random(100_000)
    fn_cost = rng.uniform(2, 20, y.size)
    model = CSLogitClassifier(loss=empc_score)

    def proflogit(loss: object, n_jobs: int) -> Callable[[], object]:
        optimizer = GeneticAlgorithmOptimizer(max_iter=100, n_jobs=n_jobs, random_state=0)
        return lambda: ProfLogitClassifier(loss=loss, optimizer=optimizer).fit(X, y)

    return {
        'import_models': (_import_models, 3),
        'mpc_score_100k': (lambda: mpc_score(y_true, y_score), 20),
        'mpc_optimal_rate_100k': (lambda: mpc_score.optimal_rate(y_true, y_score), 20),
        'empc_score_100k': (lambda: empc_score(y_true, y_score), 20),
        'proflogit_mpc': (proflogit(mpc_score, 1), 3),
        'proflogit_empc': (proflogit(empc_score, 1), 3),
        'proflogit_empc_4_jobs': (proflogit(empc_score, 4), 3),
        'proflogit_mpc_4_jobs': (proflogit(mpc_score, 4), 3),
        'profsr': (
            lambda: ProfSRClassifier(max_iter=5, population_size=300, random_state=0).fit(
                X, y, tp_cost=-200, fp_cost=10
            ),
            3,
        ),
        'clone': (lambda: clone(model), 50),
        'cstree_fit': (lambda: CSTreeClassifier(random_state=0).fit(X, y, fp_cost=1.0, fn_cost=fn_cost), 5),
    }


def main(selected: list[str]) -> None:
    cases = _cases()
    unknown = set(selected) - cases.keys()
    if unknown:
        raise SystemExit(f'Unknown case(s): {sorted(unknown)}. Known: {sorted(cases)}')
    for name, (run, repeats) in cases.items():
        if selected and name not in selected:
            continue
        run()  # warm-up: first-call compilation and imports are not what is measured
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            timings.append(time.perf_counter() - start)
        print(f'{name:28s} {statistics.median(timings) * 1000:10.2f} ms', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
