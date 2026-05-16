import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals.typing import EagerAllowed, IntoBackend

from empulse.datasets import (
    Dataset,
    fetch_give_me_some_credit,
    fetch_iranian_churn,
    load_churn_tv_subscriptions,
    load_credit_scoring_pakdd,
    load_upsell_bank_telemarketing,
)
from empulse.metrics import CostMatrix

_BACKENDS = [
    pytest.param(pd, id='pandas'),
    pytest.param(pl, id='polars'),
]

_LOADER_PARAMS = [
    pytest.param(load_churn_tv_subscriptions, id='churn_tv'),
    pytest.param(load_upsell_bank_telemarketing, id='bank_telemarketing'),
    pytest.param(load_credit_scoring_pakdd, id='credit_pakdd'),
]

# Cartesian product of loaders x backends
_LOADER_BACKEND_PARAMS = [
    pytest.param((lp.values[0], bp.values[0]), id=f'{lp.id}-{bp.id}')  # noqa: PD011
    for lp in _LOADER_PARAMS
    for bp in _BACKENDS
]


@pytest.fixture(scope='module', params=_LOADER_BACKEND_PARAMS)
def local_dataset(request: pytest.FixtureRequest) -> tuple[Dataset, IntoBackend[EagerAllowed]]:
    """Load a local dataset once per (loader, backend) pair for the whole module."""
    loader, backend = request.param
    return loader(backend=backend), backend


@pytest.fixture(scope='module', params=_LOADER_PARAMS)
def local_dataset_both_backends(request: pytest.FixtureRequest) -> tuple[Dataset, Dataset]:
    """Load a local dataset with both backends once per loader."""
    loader = request.param
    return loader(backend=pd), loader(backend=pl)


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
    dataset, backend = local_dataset
    if backend is pd:
        assert list(dataset.data.columns) == dataset.feature_names
    else:
        assert list(dataset.data.columns) == dataset.feature_names


def test_local_dataset_cross_backend_shape(local_dataset_both_backends):
    """Both backends must return the same number of rows and features."""
    ds_pd, ds_pl = local_dataset_both_backends
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


def test_fetch_give_me_some_credit_missing(tmp_path):
    """OSError when data is absent and download_if_missing=False."""
    with pytest.raises(OSError, match='download_if_missing'):
        fetch_give_me_some_credit(backend=pd, data_home=tmp_path, download_if_missing=False)


def test_fetch_iranian_churn_missing(tmp_path):
    """OSError when data is absent and download_if_missing=False."""
    with pytest.raises(OSError, match='download_if_missing'):
        fetch_iranian_churn(backend=pd, data_home=tmp_path, download_if_missing=False)


@pytest.mark.remote
@pytest.mark.parametrize(
    'backend',
    [
        pytest.param(pd, id='pandas'),
        pytest.param(pl, id='polars'),
    ],
)
def test_fetch_give_me_some_credit(tmp_path, backend):
    """Requires network access — run with ``pytest -m remote``."""
    dataset = fetch_give_me_some_credit(backend=backend, data_home=tmp_path)

    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.cost_matrix, CostMatrix)
    assert isinstance(dataset.feature_names, list)
    assert dataset.instance_costs is not None
    assert 'cl' in dataset.instance_costs
    assert 'fp_cost' in dataset.instance_costs

    n_rows = len(dataset.data)
    assert n_rows == len(dataset.target)
    for key, arr in dataset.instance_costs.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == n_rows, f"instance_costs['{key}'] length mismatch"


@pytest.mark.remote
@pytest.mark.parametrize(
    'backend',
    [
        pytest.param(pd, id='pandas'),
        pytest.param(pl, id='polars'),
    ],
)
def test_fetch_iranian_churn(tmp_path, backend):
    """Requires network access — run with ``pytest -m remote``."""
    dataset = fetch_iranian_churn(backend=backend, data_home=tmp_path)

    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.cost_matrix, CostMatrix)
    assert isinstance(dataset.feature_names, list)
    assert dataset.instance_costs is not None
    assert 'clv' in dataset.instance_costs

    n_rows = len(dataset.data)
    assert n_rows == len(dataset.target)
    for key, arr in dataset.instance_costs.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == n_rows, f"instance_costs['{key}'] length mismatch"


@pytest.mark.remote
def test_fetch_give_me_some_credit_cache(tmp_path):
    """Second call must load from cache without hitting the network."""
    ds1 = fetch_give_me_some_credit(backend=pd, data_home=tmp_path)
    ds2 = fetch_give_me_some_credit(backend=pd, data_home=tmp_path)
    pd.testing.assert_frame_equal(ds1.data, ds2.data)


@pytest.mark.remote
def test_fetch_iranian_churn_no_double_underscores(tmp_path):
    """Feature names from the remote Iranian Churn dataset must not contain '__'."""
    dataset = fetch_iranian_churn(backend=pd, data_home=tmp_path)
    for name in dataset.feature_names:
        assert '__' not in name, f'Double underscore in feature name: {name!r}'


@pytest.mark.remote
def test_fetch_give_me_some_credit_target_binary(tmp_path):
    dataset = fetch_give_me_some_credit(backend=pd, data_home=tmp_path)
    unique = set(dataset.target.unique())
    assert unique <= {0, 1}


@pytest.mark.remote
@pytest.mark.parametrize(
    'loader',
    [
        pytest.param(fetch_give_me_some_credit, id='give_me_some_credit'),
        pytest.param(fetch_iranian_churn, id='iranian_churn'),
    ],
)
def test_remote_cross_backend_shape(tmp_path, loader):
    """Both backends must return the same shape for remote datasets."""
    ds_pd = loader(backend=pd, data_home=tmp_path)
    ds_pl = loader(backend=pl, data_home=tmp_path)
    assert ds_pd.data.shape == ds_pl.data.shape
    assert len(ds_pd.target) == len(ds_pl.target)
