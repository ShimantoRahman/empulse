"""
Concurrency tests for the free-threaded build.

Two distinct properties are covered here:

1. Importing Empulse must not re-enable the GIL. This only means anything on a free-threaded
   interpreter, so those tests are skipped elsewhere.
2. Fitting or scoring the same object from several threads must agree with doing it sequentially.
   These run on *every* build: the races they guard against are reachable under the GIL too,
   because the evolution loop and the metric scoring path both call back into Python, which lets
   the interpreter switch threads mid-fit. Free-threading only widens the window.
"""

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.datasets import make_classification

from empulse.metrics import empc_score
from empulse.models import CSForestClassifier, CSTreeClassifier, ProfTreeClassifier

FREE_THREADED = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))

requires_free_threading = pytest.mark.skipif(not FREE_THREADED, reason='only meaningful on a free-threaded interpreter')


@pytest.fixture(scope='module')
def data():
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    return X.astype(np.float64), y


class TestGilStaysDisabled:
    """The `freethreading_compatible` directive in setup.py is what keeps these passing."""

    @requires_free_threading
    def test_importing_empulse_does_not_reenable_the_gil(self):
        # Importing any extension module built without the directive re-enables the GIL
        # process-wide, which silently destroys free-threading for the whole program.
        import empulse.samplers  # ruff: ignore[unused-import]

        assert not sys._is_gil_enabled(), 'importing empulse re-enabled the GIL'

    @requires_free_threading
    @pytest.mark.parametrize(
        'module',
        [
            'empulse.metrics._loss.loss',
            'empulse.metrics._cy_convex_hull.convex_hull',
            'empulse.models.cost_sensitive._impurity.cost_impurity',
            'empulse.models.cy_proftree.random',
            'empulse.models.cy_proftree.node',
            'empulse.models.cy_proftree.tree',
            'empulse.models.cy_proftree.forest',
            'empulse.models.cy_proftree.operators',
            'empulse.models.cy_proftree.evolution',
            'empulse.models.cy_proftree.max_profit',
            'empulse.models.cy_proftree.evolutionary_tree',
        ],
    )
    def test_every_extension_module_declares_compatibility(self, module):
        import importlib

        importlib.import_module(module)
        assert not sys._is_gil_enabled(), f'{module} re-enabled the GIL'


def _run_threaded(fn, args, max_workers=8):
    """Run `fn` over `args` sequentially and again across threads, returning both result lists."""
    sequential = [fn(arg) for arg in args]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        concurrent = list(executor.map(fn, args))
    return sequential, concurrent


class TestConcurrentFitsAreIndependent:
    """Each estimator instance owns its randomness; threads must not perturb one another."""

    def test_proftree_fits_do_not_share_rng_state(self, data):
        # ProfTree used to draw from libc rand()/srand(), which is process-global: one fit
        # reseeded every other, so threaded fits diverged from their sequential counterparts.
        X, y = data

        def fit(seed):
            model = ProfTreeClassifier(max_depth=3, max_iter=25, fp_cost=1, fn_cost=5, random_state=seed)
            return model.fit(X, y).predict_proba(X)

        sequential, concurrent = _run_threaded(fit, [1, 2, 3, 4, 5, 6, 7, 8])
        for seed, expected, actual in zip([1, 2, 3, 4, 5, 6, 7, 8], sequential, concurrent, strict=True):
            np.testing.assert_allclose(actual, expected, err_msg=f'threaded fit with random_state={seed} diverged')

    def test_cstree_fits_are_independent(self, data):
        X, y = data

        def fit(seed):
            model = CSTreeClassifier(max_depth=3, fp_cost=1, fn_cost=5, random_state=seed)
            return model.fit(X, y).predict_proba(X)

        sequential, concurrent = _run_threaded(fit, list(range(8)))
        for expected, actual in zip(sequential, concurrent, strict=True):
            np.testing.assert_allclose(actual, expected)

    def test_csforest_fits_are_independent(self, data):
        # CSForest hands one CostImpurity instance to sklearn, which deep-copies it per tree.
        # That per-tree isolation is what keeps its mutable C buffers from being shared.
        X, y = data

        def fit(seed):
            model = CSForestClassifier(n_estimators=8, max_depth=3, fp_cost=1, fn_cost=5, n_jobs=2, random_state=seed)
            return model.fit(X, y).predict_proba(X)

        sequential, concurrent = _run_threaded(fit, list(range(4)), max_workers=4)
        for expected, actual in zip(sequential, concurrent, strict=True):
            np.testing.assert_allclose(actual, expected)


class TestSharedMetricObjectIsThreadSafe:
    """The prebuilt `*_score` metrics are module-level singletons shared by every caller."""

    def test_concurrent_scoring_of_one_metric_object(self, data):
        _, y = data
        rng = np.random.default_rng(0)
        y_score = rng.random(y.size)
        clv_values = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]

        def score(clv):
            return empc_score(y, y_score, clv=clv, incentive_cost=10, contact_cost=1)

        sequential, concurrent = _run_threaded(score, clv_values)
        for clv, expected, actual in zip(clv_values, sequential, concurrent, strict=True):
            assert actual == pytest.approx(expected), f'threaded empc_score(clv={clv}) diverged'
