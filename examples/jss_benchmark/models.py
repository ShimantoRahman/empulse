"""The benchmarked models, for Empulse (``current``) and for the reference implementations (``legacy``).

``build_model(impl, name, fn_cost, fp_cost, seed)`` returns an unfitted object whose ``fit(X, y)``
is what the benchmark times. The reference implementations do not share scikit-learn's ``fit(X, y)``
signature, so they are wrapped in small adapters whose ``fit`` makes exactly the call the reference
expects. Their cost matrices are assembled in ``build_model``, outside the timed call, since that is
the caller's input format rather than part of training.

Hyperparameters are identical wherever both implementations have the concept:

- CSLogit: zero initial coefficients, an L1 penalty with Empulse's default strength (``C=1``), and
  L-BFGS-B capped at 100 iterations. The reference hard-codes its optimizer options, so ``_CSLogit``
  overrides ``optimization`` to add the iteration cap and keeps the reference's own function
  tolerance (``ftol=1e-6``). The reference supplies no gradient, so SciPy approximates it by finite
  differences, as in the original code.
- CSTree: maximum depth 10 and at least 20 samples to split. CostCla's post-pruning pass, which
  Empulse has no equivalent of, is disabled, so both time tree induction only. Everything else keeps
  CostCla's defaults.
- CSForest: 100 such trees on bootstrap samples, ``sqrt(n_features)`` candidate features per split
  (the default of both implementations), fitted on one thread.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.optimize

from .reference_costcla import CostSensitiveDecisionTreeClassifier, CostSensitiveRandomForestClassifier
from .reference_cslogit import CSLogit

MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 20
N_ESTIMATORS = 100
LBFGS_MAX_ITER = 100
LBFGS_TOLERANCE = 1e-4


def build_model(impl: str, name: str, fn_cost: np.ndarray, fp_cost: np.ndarray, seed: int) -> Any:
    """Construct one of the benchmarked models, unfitted."""
    if impl == 'current':
        return _build_current(name, fn_cost, fp_cost, seed)
    if impl == 'legacy':
        return _build_legacy(name, fn_cost, fp_cost, seed)
    raise ValueError(f'Unknown implementation {impl!r}')


def _build_current(name: str, fn_cost: np.ndarray, fp_cost: np.ndarray, seed: int) -> Any:
    from empulse.models import CSForestClassifier, CSLogitClassifier, CSTreeClassifier
    from empulse.optimizers import LBFGSBOptimizer

    if name == 'cslogit':
        return CSLogitClassifier(
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            optimizer=LBFGSBOptimizer(max_iter=LBFGS_MAX_ITER, tolerance=LBFGS_TOLERANCE),
        )
    if name == 'cstree':
        return CSTreeClassifier(
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            random_state=seed,
        )
    if name == 'csforest':
        return CSForestClassifier(
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f'Unknown model {name!r}')


def _build_legacy(name: str, fn_cost: np.ndarray, fp_cost: np.ndarray, seed: int) -> Any:
    if name == 'cslogit':
        return _CSLogitAdapter(fn_cost, fp_cost)
    if name == 'cstree':
        tree = CostSensitiveDecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            pruned=False,
        )
        return _CostClaAdapter(tree, fn_cost, fp_cost, seed)
    if name == 'csforest':
        forest = CostSensitiveRandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=1, pruned=False)
        # CostCla's forest has no depth or split-size arguments of its own; they live on its base tree.
        forest.estimator.set_params(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)
        # The forest hard-codes random_state=None in its constructor, so the seed is set directly.
        forest.random_state = seed
        return _CostClaAdapter(forest, fn_cost, fp_cost, seed)
    raise ValueError(f'Unknown model {name!r}')


class _CSLogit(CSLogit):
    """The reference CSLogit with the iteration cap used for every CSLogit in the benchmark."""

    def optimization(self, obj_func, initial_theta):
        opt_res = scipy.optimize.minimize(
            obj_func, initial_theta, method='L-BFGS-B', options={'ftol': 1e-6, 'maxiter': LBFGS_MAX_ITER}
        )
        theta_opt, func_min, n_iter = opt_res.x, opt_res.fun, opt_res.nfev
        self.theta_opt = theta_opt
        return theta_opt, func_min, n_iter


class _CSLogitAdapter:
    def __init__(self, fn_cost: np.ndarray, fp_cost: np.ndarray) -> None:
        # Indexed [:, predicted, true]; true positives and true negatives cost nothing.
        self.cost_matrix = np.zeros((len(fn_cost), 2, 2))
        self.cost_matrix[:, 1, 0] = fp_cost
        self.cost_matrix[:, 0, 1] = fn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.model: _CSLogit | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _CSLogitAdapter:
        # Empulse scales its L1 penalty by the mean absolute derivative of the expected cost with
        # respect to the predicted probability, divided by C * n_samples. With tp = tn = 0 that
        # derivative is fp_cost for a negative and -fn_cost for a positive, and C = 1.
        l1_weight = float(np.mean(np.where(y == 1, self.fn_cost, self.fp_cost))) / len(y)
        self.model = _CSLogit(np.zeros(X.shape[1] + 1), lambda1=l1_weight, obj='aec')
        self.model.fitting(X, y.astype(np.float64), self.cost_matrix)
        return self


class _CostClaAdapter:
    def __init__(self, model: Any, fn_cost: np.ndarray, fp_cost: np.ndarray, seed: int) -> None:
        zeros = np.zeros_like(fn_cost)
        # CostCla's column order: false positive, false negative, true positive, true negative.
        self.cost_mat = np.column_stack([fp_cost, fn_cost, zeros, zeros])
        self.model = model
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> _CostClaAdapter:
        # CostCla's trees draw candidate features from NumPy's global random state.
        np.random.seed(self.seed)
        self.model.fit(X, y, self.cost_mat)
        return self
