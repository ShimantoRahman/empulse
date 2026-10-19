"""EMP execution-time and accuracy sweeps; results are returned as dataframes."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sympy.stats import Beta, Gamma, Uniform

from empulse.metrics import CostMatrix, MaxProfit, Metric

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
DIM_COLUMNS = ('method', 'n_stochastic', 'n_samples', 'iteration', 'time_seconds', 'value', 'reference', 'error_pct')

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


def run_size_sweep(sizes: list[int] = SIZES, n_iterations: int = N_ITERATIONS) -> pd.DataFrame:
    """Compare integration methods across sample sizes, using Exact as the accuracy baseline."""
    cost_matrix = build_cost_matrix(1)
    rows = []
    for n_samples in sizes:
        y_true, y_proba = _scores(n_samples)
        exact_value = None
        for name, method in METHODS.items():
            metric = _metric(cost_matrix, method)
            value = metric(y_true, y_proba)  # warm-up and reported EMP value
            if name == 'Exact':
                exact_value = value
            error_pct = 0.0 if name == 'Exact' else abs(value - exact_value) / abs(exact_value) * 100
            for iteration, elapsed in enumerate(_time(metric, y_true, y_proba, n_iterations)):
                rows.append({
                    'method': name,
                    'n_samples': n_samples,
                    'iteration': iteration,
                    'time_seconds': elapsed,
                    'value': value,
                    'error_pct': error_pct,
                })
            print(f'N={n_samples:>7,} {name:5s} EMP={value:.4f} error={error_pct:.2e}%', flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)


def run_dimension_sweep(
    dimensions: tuple[int, ...] = DIMENSIONS,
    n_iterations: int = DIM_N_ITERATIONS,
    reference_samples_exp: int = REFERENCE_SAMPLES_EXP,
    n_samples: int = DIM_N_SAMPLES,
) -> pd.DataFrame:
    """Compare stochastic dimensions; use QMC with 2**22 points when Exact is unavailable."""
    y_true, y_proba = _scores(n_samples)
    rows = []
    for n_stochastic in dimensions:
        cost_matrix = build_cost_matrix(n_stochastic)
        if n_stochastic == 1:
            reference = _metric(cost_matrix, 'auto')(y_true, y_proba)
            methods = METHODS
        else:
            reference = _metric(cost_matrix, 'quasi-monte-carlo', reference_samples_exp)(y_true, y_proba)
            methods = {name: method for name, method in METHODS.items() if name != 'Exact'}
        for name, method in methods.items():
            metric = _metric(cost_matrix, method)
            value = metric(y_true, y_proba)  # warm-up and reported EMP value
            error_pct = abs(value - reference) / abs(reference) * 100
            for iteration, elapsed in enumerate(_time(metric, y_true, y_proba, n_iterations)):
                rows.append({
                    'method': name,
                    'n_stochastic': n_stochastic,
                    'n_samples': n_samples,
                    'iteration': iteration,
                    'time_seconds': elapsed,
                    'value': value,
                    'reference': reference,
                    'error_pct': error_pct,
                })
            print(f'{n_stochastic} variable(s) {name:5s} EMP={value:.4f} error={error_pct:.2e}%', flush=True)
    return pd.DataFrame(rows, columns=DIM_COLUMNS)
