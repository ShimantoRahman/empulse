"""Fixtures shared across ``tests/datasets/``."""

import pytest


@pytest.fixture(scope='session')
def remote_data_home(tmp_path_factory):
    """
    One data home for every ``remote`` test in the session.

    Each remote test used to download into its own ``tmp_path``, so the weekly job fetched Give Me
    Some Credit five times and most other datasets twice. The one test that is about the download
    itself (populating the cache) still uses its own empty directory.
    """
    return tmp_path_factory.mktemp('empulse_data')
