"""
Fixtures shared across the whole test suite.

Before this file existed every test module built its own synthetic data, which produced eight
groups of byte-identical fixture bodies and two names -- ``data`` and ``dataset`` -- that each
meant something different in ten different files.

Only genuinely cross-cutting fixtures belong here. Per-subpackage fixtures live in
``tests/metrics/conftest.py`` and ``tests/models/conftest.py``.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

# The seed used wherever a fixture needs randomness. Fixed so a failure is reproducible from the
# test id alone; several tests previously called `np.random.rand` unseeded -- one of them inside a
# `parametrize` decorator, so its data changed on every collection.
RANDOM_STATE = 42


@pytest.fixture
def seeded_rng():
    """A NumPy generator seeded with :data:`RANDOM_STATE`."""
    return np.random.default_rng(RANDOM_STATE)


@pytest.fixture(scope='session')
def make_data():
    """
    Factory for :func:`sklearn.datasets.make_classification` problems.

    Twelve modules previously defined a ``make_classification`` fixture with nine different
    signatures. Call this with whatever shape the test needs instead::

        def test_something(make_data):
            X, y = make_data(n_samples=200, n_features=5)

    ``random_state`` defaults to :data:`RANDOM_STATE`; pass it explicitly to override.
    """

    def _make(**kwargs):
        kwargs.setdefault('random_state', RANDOM_STATE)
        return make_classification(**kwargs)

    return _make
