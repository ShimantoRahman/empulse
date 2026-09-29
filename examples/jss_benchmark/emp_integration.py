"""EMP integration methods: execution time and accuracy of ``MaxProfit``'s ``integration_method`` options.

    python -m jss_benchmark.emp_integration [--quick]

Two sweeps on the stochastic churn cost matrix of the paper (clv = 200, d = 10, f = 1,
gamma ~ Beta(6, 14)), writing raw per-iteration rows to ``results/``:

1. ``emp_integration.csv``, dataset size with one stochastic variable. ``'auto'`` resolves to exact
   piecewise integration for this cost matrix, is reported as ``Exact``, and is the reference the
   other three methods (``Quad``, ``QMC``, ``MC``) are measured against.
2. ``emp_integration_dimensions.csv``, one to three stochastic variables at N = 10,000: ``clv``
   becomes Gamma-distributed (mean 200) and ``d`` uniform on [5, 15]. Beyond one variable there is
   no exact value, so the reference is a quasi-Monte Carlo estimate with 2^22 points.

Quadrature with three stochastic variables takes about 100 s per evaluation, so the full run takes
roughly ten minutes. Both files are rewritten from scratch on every run.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sympy.stats import Beta, Gamma, Uniform

from empulse.metrics import CostMatrix, MaxProfit, Metric

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

SIZES = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
# 'Exact' must come first: its value is the reference the other methods are measured against.
METHODS = {'Exact': 'auto', 'Quad': 'quad', 'QMC': 'quasi-monte-carlo', 'MC': 'monte-carlo'}
N_ITERATIONS = 10
ERROR_N_SAMPLES = 10_000  # the size at which the accuracy panel reports the error
COLUMNS = ('method', 'n_samples', 'iteration', 'time_seconds', 'value', 'error_pct')

DIM_N_SAMPLES = 10_000
DIMENSIONS = (1, 2, 3)
DIM_N_ITERATIONS = 5
REFERENCE_SAMPLES_EXP = 22
DIM_COLUMNS = ('method', 'n_stochastic', 'iteration', 'time_seconds', 'value', 'reference', 'error_pct')

RANDOM_STATE = 0


def build_cost_matrix(n_stochastic: int = 1) -> CostMatrix:
    """The stochastic churn cost matrix with one, two or three stochastic variables.

    The acceptance rate ``gamma`` is always stochastic. With ``n_stochastic >= 2``, ``clv`` is
    Gamma-distributed (mean 200, standard deviation 40), and with ``n_stochastic == 3``, ``d`` is
    uniform on [5, 15]. The means equal the deterministic defaults, so the three matrices describe the
    same business case with growing uncertainty.
    """
    alpha, beta, f = sympy.symbols('alpha beta f')
    gamma = Beta('gamma', alpha, beta)
    clv = Gamma('clv', 25, 8) if n_stochastic >= 2 else sympy.Symbol('clv')
    d = Uniform('d', 5, 15) if n_stochastic >= 3 else sympy.Symbol('d')

    cost_matrix = CostMatrix().add_tp_benefit(gamma * (clv - d - f) + (1 - gamma) * -f).add_fp_cost(d + f)
    defaults = {'f': 1, 'alpha': 6, 'beta': 14}
    if n_stochastic < 2:
        defaults['clv'] = 200
    if n_stochastic < 3:
        defaults['d'] = 10
    cost_matrix.set_default(**defaults)
    return cost_matrix


def _scores(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    X, y_true = make_classification(n_samples=n_samples, n_features=10, random_state=42)
    model = LogisticRegression().fit(X, y_true)
    return y_true, model.predict_proba(X)[:, 1]


def _metric(cost_matrix: CostMatrix, method: str, n_mc_samples_exp: int = 16) -> Metric:
    strategy = MaxProfit(integration_method=method, n_mc_samples_exp=n_mc_samples_exp, random_state=RANDOM_STATE)
    return Metric(cost_matrix=cost_matrix, strategy=strategy)


def _time(metric: Metric, y_true: np.ndarray, y_proba: np.ndarray, n_iterations: int) -> list[float]:
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        metric(y_true, y_proba)
        times.append(time.perf_counter() - start)
    return times


def run_size_sweep(sizes: list[int] = SIZES, n_iterations: int = N_ITERATIONS, results_dir: Path = RESULTS_DIR) -> Path:
    """Time and score every (method, size) combination with one stochastic variable."""
    cost_matrix = build_cost_matrix(1)
    path = results_dir / 'emp_integration.csv'
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for n_samples in sizes:
            y_true, y_proba = _scores(n_samples)
            exact_value = None
            for name, method in METHODS.items():
                metric = _metric(cost_matrix, method)
                value = metric(y_true, y_proba)  # warm-up run; also this method's reported value
                if name == 'Exact':
                    exact_value = value
                error_pct = 0.0 if name == 'Exact' else abs(value - exact_value) / abs(exact_value) * 100
                for iteration, elapsed in enumerate(_time(metric, y_true, y_proba, n_iterations)):
                    writer.writerow({
                        'method': name,
                        'n_samples': n_samples,
                        'iteration': iteration,
                        'time_seconds': elapsed,
                        'value': value,
                        'error_pct': error_pct,
                    })
                fh.flush()
                print(f'N={n_samples:>7,} {name:5s} EMP={value:.4f} error={error_pct:.2e}%', flush=True)
    return path


def run_dimension_sweep(
    dimensions: tuple[int, ...] = DIMENSIONS, n_iterations: int = DIM_N_ITERATIONS, results_dir: Path = RESULTS_DIR
) -> Path:
    """Time and score each method against the number of stochastic variables, at N = 10,000."""
    y_true, y_proba = _scores(DIM_N_SAMPLES)
    path = results_dir / 'emp_integration_dimensions.csv'
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=DIM_COLUMNS)
        writer.writeheader()
        for n_stochastic in dimensions:
            cost_matrix = build_cost_matrix(n_stochastic)
            if n_stochastic == 1:
                reference = _metric(cost_matrix, 'auto')(y_true, y_proba)
                methods = METHODS
            else:
                reference = _metric(cost_matrix, 'quasi-monte-carlo', REFERENCE_SAMPLES_EXP)(y_true, y_proba)
                methods = {name: method for name, method in METHODS.items() if name != 'Exact'}

            for name, method in methods.items():
                metric = _metric(cost_matrix, method)
                value = metric(y_true, y_proba)  # warm-up run; also this method's reported value
                error_pct = abs(value - reference) / abs(reference) * 100
                for iteration, elapsed in enumerate(_time(metric, y_true, y_proba, n_iterations)):
                    writer.writerow({
                        'method': name,
                        'n_stochastic': n_stochastic,
                        'iteration': iteration,
                        'time_seconds': elapsed,
                        'value': value,
                        'reference': reference,
                        'error_pct': error_pct,
                    })
                fh.flush()
                print(f'{n_stochastic} variable(s) {name:5s} EMP={value:.4f} error={error_pct:.2e}%', flush=True)
    return path


def run_emp_integration(quick: bool = False, results_dir: Path = RESULTS_DIR) -> None:
    """Run both sweeps. ``quick`` uses three sizes, three iterations and at most two variables."""
    if quick:
        run_size_sweep(SIZES[:3], 3, results_dir)
        run_dimension_sweep(DIMENSIONS[:2], 3, results_dir)
    else:
        run_size_sweep(results_dir=results_dir)
        run_dimension_sweep(results_dir=results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='A reduced run of a few seconds.')
    args = parser.parse_args()
    run_emp_integration(quick=args.quick)


if __name__ == '__main__':
    main()
