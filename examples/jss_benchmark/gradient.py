"""Replicate the analytical-gradient comparison discussed in Section 10.

The vendored CSLogit remains unchanged. This additional adapter supplies the
derivative of its AEC + L1 objective to the same L-BFGS-B solver. Runtime ratios
are hardware-dependent, and final objective values are reported for comparison.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .data import SEEDS, make_dataset
from .models import LBFGS_MAX_ITER, _CSLogit, _CSLogitAdapter, build_model


def aec_gradient(theta, X, y, cost_matrix, l1_weight):
    """Derivative of the reference AEC objective, with no intercept penalty."""
    scores = expit(theta[0] + X @ theta[1:])
    slope = y * (cost_matrix[:, 1, 1] - cost_matrix[:, 0, 1]) + (1 - y) * (cost_matrix[:, 1, 0] - cost_matrix[:, 0, 0])
    residual = scores * (1 - scores) * slope
    return np.concatenate(([residual.mean()], X.T @ residual / len(y) + l1_weight * np.sign(theta[1:])))


class _AnalyticalCSLogit(_CSLogit):
    def fitting(self, x, y, cost_matrix):
        self._X, self._y, self._cost_matrix = x, y, cost_matrix
        return super().fitting(x, y, cost_matrix)

    def optimization(self, obj_func, initial_theta):
        result = minimize(
            obj_func,
            initial_theta,
            method='L-BFGS-B',
            jac=lambda theta: aec_gradient(theta, self._X, self._y, self._cost_matrix, self.lambda1),
            options={'ftol': 1e-6, 'maxiter': LBFGS_MAX_ITER},
        )
        self.theta_opt = result.x
        return result.x, result.fun, result.nfev


class _AnalyticalAdapter(_CSLogitAdapter):
    def fit(self, X, y):
        l1_weight = float(np.mean(np.where(y == 1, self.fn_cost, self.fp_cost))) / len(y)
        self.model = _AnalyticalCSLogit(np.zeros(X.shape[1] + 1), lambda1=l1_weight, obj='aec')
        self.model.fitting(X, y.astype(np.float64), self.cost_matrix)
        return self


def run_gradient_comparison(n_samples: int, seeds: Iterable[int] = SEEDS) -> pd.DataFrame:
    """Fit the finite-difference reference, analytical reference and Empulse once per seed."""
    rows = []
    for seed in seeds:
        X, y, fn_cost, fp_cost = make_dataset(n_samples, seed)
        reference = _CSLogitAdapter(fn_cost, fp_cost)
        for name, model in (
            ('finite_difference', reference),
            ('analytical_gradient', _AnalyticalAdapter(fn_cost, fp_cost)),
            ('empulse', build_model('current', 'cslogit', fn_cost, fp_cost, seed)),
        ):
            started = time.perf_counter()
            model.fit(X, y)
            elapsed = time.perf_counter() - started
            if name == 'empulse':
                theta = np.concatenate((np.atleast_1d(model.intercept_), model.coef_.ravel()))
            else:
                theta = model.model.theta
            objective = reference.model.aec(theta, X, y, reference.cost_matrix)
            rows.append({
                'implementation': name,
                'n_samples': n_samples,
                'seed': seed,
                'fit_seconds': elapsed,
                'reference_objective': float(objective),
            })
    return pd.DataFrame(rows)


def summarize_gradient_comparison(results: pd.DataFrame) -> pd.Series:
    """The two speedups quoted in Section 10, from the mean fit times."""
    seconds = results.groupby('implementation')['fit_seconds'].mean()
    return pd.Series({
        'Speedup, reference: analytical gradient over finite differences': (
            f'{seconds["finite_difference"] / seconds["analytical_gradient"]:.1f}x'
        ),
        'Speedup, Empulse over reference with analytical gradient': (
            f'{seconds["analytical_gradient"] / seconds["empulse"]:.1f}x'
        ),
    })
