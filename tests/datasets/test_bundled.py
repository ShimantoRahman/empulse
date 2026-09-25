"""The contract every dataset bundled with the package satisfies, for each dataframe backend."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from empulse.datasets import (
    Dataset,
    load_churn_tv_subscriptions,
    load_credit_scoring_pakdd,
    load_upsell_bank_telemarketing,
    load_vub_credit_scoring,
)
from empulse.metrics import Cost, CostMatrix, Metric

from ._helpers import BACKENDS, bahnsen_fp_cost

LOADERS = [
    pytest.param(load_churn_tv_subscriptions, id='churn_tv'),
    pytest.param(load_upsell_bank_telemarketing, id='bank_telemarketing'),
    pytest.param(load_credit_scoring_pakdd, id='credit_pakdd'),
    pytest.param(load_vub_credit_scoring, id='vub_credit_scoring'),
]


@pytest.fixture(scope='module')
def load():
    """
    Load a bundled dataset with a backend, reading each (loader, backend) pair only once.

    The per-backend and the cross-backend fixtures below used to load every pair independently.
    Nothing here modifies a dataset, so they can share one copy.
    """
    cache = {}

    def _load(loader, backend):
        key = (loader, backend)
        if key not in cache:
            cache[key] = loader(backend=backend)
        return cache[key]

    return _load


@pytest.fixture(params=LOADERS)
def loader(request):
    return request.param


@pytest.fixture(params=BACKENDS)
def local_dataset(request, load, loader):
    backend = request.param
    return load(loader, backend), backend


def test_local_dataset_returns_correct_types(local_dataset):
    """Loaded dataset must expose the right types for all public attributes."""
    dataset, backend = local_dataset

    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.cost_matrix, CostMatrix)
    assert isinstance(dataset.name, str) and dataset.name
    assert isinstance(dataset.DESCR, str) and dataset.DESCR
    assert isinstance(dataset.feature_names, list)
    assert all(isinstance(n, str) for n in dataset.feature_names)

    if backend is pd:
        assert isinstance(dataset.data, pd.DataFrame)
        assert isinstance(dataset.target, pd.Series)
    else:
        assert isinstance(dataset.data, pl.DataFrame)
        assert isinstance(dataset.target, pl.Series)


def test_local_dataset_instance_costs_are_arrays(local_dataset):
    """Every entry in instance_costs must be a numpy array aligned with the data."""
    dataset, _ = local_dataset
    n_rows = len(dataset.data)

    if dataset.instance_costs is not None:
        assert isinstance(dataset.instance_costs, dict)
        for key, arr in dataset.instance_costs.items():
            assert isinstance(key, str)
            assert isinstance(arr, np.ndarray)
            assert arr.shape[0] == n_rows, f"instance_costs['{key}'] length mismatch"


def test_local_dataset_feature_names_match_columns(local_dataset):
    """feature_names must exactly match the DataFrame column order."""
    dataset, _ = local_dataset
    assert list(dataset.data.columns) == dataset.feature_names


def test_local_dataset_cross_backend_shape(load, loader):
    """Both backends must return the same number of rows and features."""
    ds_pd, ds_pl = load(loader, pd), load(loader, pl)
    assert ds_pd.data.shape == ds_pl.data.shape
    assert len(ds_pd.target) == len(ds_pl.target)


def test_local_dataset_target_row_count(local_dataset):
    """Number of rows in data and target must match."""
    dataset, _ = local_dataset
    assert len(dataset.data) == len(dataset.target)


def test_local_dataset_target_is_binary(local_dataset):
    """Target column must contain only 0 and 1."""
    dataset, backend = local_dataset
    unique = set(dataset.target.unique()) if backend is pd else set(dataset.target.unique().to_list())
    assert unique <= {0, 1}, f'Non-binary target values: {unique}'


def test_local_dataset_target_has_both_classes(local_dataset):
    """Both class 0 and class 1 must be present."""
    dataset, backend = local_dataset
    unique = set(dataset.target.unique()) if backend is pd else set(dataset.target.unique().to_list())
    assert 0 in unique and 1 in unique, f'Missing class in target: {unique}'


def test_local_dataset_feature_names_non_empty(local_dataset):
    """No feature name should be an empty string."""
    dataset, _ = local_dataset
    assert all(name != '' for name in dataset.feature_names)


def test_local_dataset_feature_names_unique(local_dataset):
    """Feature names must be unique."""
    dataset, _ = local_dataset
    assert len(dataset.feature_names) == len(set(dataset.feature_names))


def test_local_dataset_no_double_underscores_in_names(local_dataset):
    """Sanitised feature names must not contain consecutive underscores."""
    dataset, _ = local_dataset
    for name in dataset.feature_names:
        assert '__' not in name, f'Double underscore in feature name: {name!r}'


def test_local_dataset_instance_costs_non_negative(local_dataset):
    """Instance cost drivers (CLV, balance, credit lines) must be non-negative."""
    dataset, _ = local_dataset
    if dataset.instance_costs is None:
        pytest.skip('No instance costs for this dataset')
    for key, arr in dataset.instance_costs.items():
        assert np.all(arr >= 0), f"Negative values in instance_costs['{key}']"


@pytest.mark.parametrize('backend', BACKENDS)
def test_vub_credit_scoring_matches_its_source(load, backend):
    """The VUB data and its cost matrix, against the published counts and the Bahnsen costs."""
    ds = load(load_vub_credit_scoring, backend)
    assert len(ds.data) == 18917
    assert len(ds.feature_names) == 16
    assert int(np.asarray(ds.target).sum()) == 3206
    assert np.all(ds.instance_costs['cl'] > 0)
    np.testing.assert_allclose(
        ds.instance_costs['fp_cost'],
        bahnsen_fp_cost(ds.instance_costs['cl'], pi_1=3206 / 18917),
    )

    metric = Metric(ds.cost_matrix, Cost())
    score = metric(ds.target, np.full(len(ds.target), 0.5), **ds.instance_costs)
    assert np.isfinite(score)
