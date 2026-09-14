"""
Smoke-test an installed Empulse wheel.

Run by cibuildwheel (``CIBW_TEST_COMMAND``) against each built wheel, in a fresh environment with
only the runtime dependencies installed. It is deliberately import-light: the boosting and symbolic
extras are not installed there, so nothing here may touch them.

Its most important job is the free-threading assertion. A Cython extension built without the
``freethreading_compatible`` directive re-enables the GIL process-wide the moment it is imported,
which would silently take free-threading away from every program that imports Empulse. That failure
is invisible in ordinary use, so it is checked here, at the point where the wheel is built.
"""

import sys
import sysconfig

import numpy as np

FREE_THREADED = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))


def check_imports() -> None:
    """Import every subpackage, which pulls in all 11 compiled extension modules."""
    import empulse

    print(f'imported empulse {empulse.__version__} on {sys.version}')


def check_gil_is_still_disabled() -> None:
    """On a free-threaded build, importing Empulse must leave the GIL disabled."""
    if not FREE_THREADED:
        print('GIL-enabled build: skipping free-threading check')
        return

    if sys._is_gil_enabled():
        raise SystemExit(
            'importing empulse re-enabled the GIL on a free-threaded interpreter. '
            "The 'freethreading_compatible' Cython directive in setup.py did not reach the build."
        )
    print('free-threaded build: GIL is still disabled after importing empulse')


def check_models_fit() -> None:
    """Exercise the two extension-backed estimators end to end."""
    from empulse.metrics import expected_cost_loss
    from empulse.models import CSTreeClassifier, ProfTreeClassifier

    rng = np.random.default_rng(0)
    n_samples = 120
    X = rng.normal(size=(n_samples, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_samples) > 0).astype(int)

    tree = CSTreeClassifier(max_depth=3, fp_cost=1.0, fn_cost=5.0, random_state=0).fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (n_samples, 2), proba.shape

    proftree = ProfTreeClassifier(max_depth=3, max_iter=10, fp_cost=1.0, fn_cost=5.0, random_state=0)
    proftree.fit(X, y)
    assert proftree.predict(X).shape == (n_samples,)

    # Exercises the compiled convex hull and loss extensions.
    loss = expected_cost_loss(y, proba[:, 1], fp_cost=1.0, fn_cost=5.0)
    assert np.isfinite(loss), loss

    print('CSTreeClassifier, ProfTreeClassifier and expected_cost_loss all ran')


if __name__ == '__main__':
    check_imports()
    check_gil_is_still_disabled()
    check_models_fit()
    print('wheel smoke test passed')
