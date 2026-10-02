"""Fixtures shared across ``tests/datasets/``."""

import pytest


@pytest.fixture(scope='session')
def remote_data_home(tmp_path_factory):
    """
    One data home for every ``remote`` test in the session.

    The weekly job then fetches each dataset once instead of once per test. The one test about the
    download itself (populating the cache) uses its own empty directory.
    """
    return tmp_path_factory.mktemp('empulse_data')
