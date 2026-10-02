"""
Fixtures shared across ``tests/models/``.

The single-letter names ``X`` and ``y`` follow the scikit-learn convention; this conftest is scoped
to the model tests, where that reading is unambiguous.
"""

import numpy as np
import pytest


@pytest.fixture(scope='session')
def X():
    """The 10-sample, 2-feature toy design matrix, small enough to reason about by hand."""
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='session')
def y():
    """Balanced alternating labels for :func:`X`."""
    return np.array([0, 1] * 5)


@pytest.fixture(scope='session')
def sensitive_feature():
    """The 5/5 split protected attribute paired with :func:`X`."""
    return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


@pytest.fixture(scope='session')
def classification_data(make_data):
    """The 100-sample, 20-feature problem that most model tests use."""
    return make_data(n_samples=100, n_features=20)


@pytest.fixture(scope='session')
def cost_dataset(make_data):
    """A 50-sample problem with a per-sample ``fn_cost`` and a scalar ``fp_cost``: ``(X, y, fn_cost, fp_cost)``."""
    X, y = make_data(n_samples=50)
    fn_cost = np.random.default_rng(42).random(y.size)
    return X, y, fn_cost, 5
