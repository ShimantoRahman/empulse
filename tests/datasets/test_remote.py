"""
The ``fetch_*`` loaders: their offline behaviour, and (marked ``remote``) the downloaded data.

The ``remote`` tests download from third-party hosts, so CI runs them weekly rather than on every
pull request -- see ``.github/workflows/tests.yml``. Run them locally with ``pytest -m remote``.
"""

import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from empulse.datasets import (
    Dataset,
    fetch_cell2cell,
    fetch_credit_card_fraud,
    fetch_default_credit_card_clients,
    fetch_give_me_some_credit,
    fetch_home_equity,
    fetch_ieee_fraud_detection,
    fetch_iranian_churn,
    fetch_kdd98,
    fetch_kddcup09_churn,
    fetch_south_german_credit,
    fetch_telco_customer_churn,
)
from empulse.metrics import Cost, CostMatrix, Metric

from ._helpers import BACKENDS, MOCK_RAW_CELL2CELL

FETCHERS = [
    fetch_cell2cell,
    fetch_credit_card_fraud,
    fetch_default_credit_card_clients,
    fetch_give_me_some_credit,
    fetch_home_equity,
    fetch_ieee_fraud_detection,
    fetch_iranian_churn,
    fetch_kdd98,
    fetch_kddcup09_churn,
    fetch_south_german_credit,
    fetch_telco_customer_churn,
]


@pytest.mark.parametrize('fetcher', FETCHERS, ids=lambda fetcher: fetcher.__name__)
def test_fetch_missing_raises_oserror(tmp_path, fetcher):
    """OSError when data is absent and download_if_missing=False."""
    with pytest.raises(OSError, match='download_if_missing'):
        fetcher(backend=pd, data_home=tmp_path, download_if_missing=False)


@patch('empulse.datasets._remote._fetch_cell2cell_raw', return_value=MOCK_RAW_CELL2CELL)
def test_fetch_cell2cell_mocked(mock_fetch, tmp_path):
    ds = fetch_cell2cell(backend=pd, data_home=tmp_path)
    assert isinstance(ds, Dataset)
    assert ds.name == 'Cell2Cell Customer Churn'
    assert len(ds.target) == 3
    assert 'monthly_revenue' in ds.instance_costs
    assert ds.target_names == ['no churn', 'churn']
    assert len(ds.feature_names) == 56


_SNAKE_CASE = re.compile(r'^[a-z0-9]+(_[a-z0-9]+)*$')


def _check_remote(ds, *, n_samples, n_positives, n_features, cost_keys):
    assert isinstance(ds, Dataset)
    assert isinstance(ds.cost_matrix, CostMatrix)
    assert len(ds.data) == n_samples
    assert len(ds.target) == n_samples
    assert len(ds.feature_names) == n_features
    assert [name for name in ds.feature_names if not _SNAKE_CASE.match(name)] == []
    assert int(np.asarray(ds.target).sum()) == n_positives
    assert set(ds.instance_costs) == cost_keys
    for arr in ds.instance_costs.values():
        assert arr.shape == (n_samples,)

    score = Metric(ds.cost_matrix, Cost())(ds.target, np.full(n_samples, 0.5), **ds.instance_costs)
    assert np.isfinite(score)


def _check_instance_costs_align(dataset):
    n_rows = len(dataset.data)
    assert n_rows == len(dataset.target)
    for key, arr in dataset.instance_costs.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == n_rows, f"instance_costs['{key}'] length mismatch"


@pytest.mark.remote
class TestRemoteDatasets:
    @pytest.mark.parametrize('backend', BACKENDS)
    def test_fetch_give_me_some_credit(self, remote_data_home, backend):
        dataset = fetch_give_me_some_credit(backend=backend, data_home=remote_data_home)
        assert isinstance(dataset, Dataset)
        assert isinstance(dataset.cost_matrix, CostMatrix)
        assert isinstance(dataset.feature_names, list)
        assert {'cl', 'fp_cost'} <= set(dataset.instance_costs)
        assert set(dataset.target.unique() if backend is pd else dataset.target.unique().to_list()) <= {0, 1}
        _check_instance_costs_align(dataset)

    def test_fetch_give_me_some_credit_cache(self, tmp_path):
        """Second call must load from cache without hitting the network."""
        ds1 = fetch_give_me_some_credit(backend=pd, data_home=tmp_path)
        ds2 = fetch_give_me_some_credit(backend=pd, data_home=tmp_path)
        pd.testing.assert_frame_equal(ds1.data, ds2.data)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_fetch_iranian_churn(self, remote_data_home, backend):
        dataset = fetch_iranian_churn(backend=backend, data_home=remote_data_home)
        assert isinstance(dataset, Dataset)
        assert isinstance(dataset.cost_matrix, CostMatrix)
        assert isinstance(dataset.feature_names, list)
        assert 'clv' in dataset.instance_costs
        for name in dataset.feature_names:
            assert '__' not in name, f'Double underscore in feature name: {name!r}'
        _check_instance_costs_align(dataset)

    @pytest.mark.parametrize(
        'loader',
        [
            pytest.param(fetch_give_me_some_credit, id='give_me_some_credit'),
            pytest.param(fetch_iranian_churn, id='iranian_churn'),
        ],
    )
    def test_cross_backend_shape(self, remote_data_home, loader):
        """Both backends must return the same shape for remote datasets."""
        ds_pd = loader(backend=pd, data_home=remote_data_home)
        ds_pl = loader(backend=pl, data_home=remote_data_home)
        assert ds_pd.data.shape == ds_pl.data.shape
        assert len(ds_pd.target) == len(ds_pl.target)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_fetch_telco_customer_churn(self, remote_data_home, backend):
        ds = fetch_telco_customer_churn(backend=backend, data_home=remote_data_home)
        _check_remote(ds, n_samples=7032, n_positives=1869, n_features=19, cost_keys={'monthly_charges'})

    def test_fetch_default_credit_card_clients(self, remote_data_home):
        ds = fetch_default_credit_card_clients(backend=pd, data_home=remote_data_home)
        _check_remote(ds, n_samples=30000, n_positives=6636, n_features=23, cost_keys={'cl', 'fp_cost'})

    def test_fetch_credit_card_fraud(self, remote_data_home):
        ds = fetch_credit_card_fraud(backend=pd, data_home=remote_data_home)
        _check_remote(ds, n_samples=282982, n_positives=465, n_features=29, cost_keys={'amount'})
        assert np.all(ds.instance_costs['amount'] > 0)

    def test_fetch_ieee_fraud_detection(self, remote_data_home):
        ds = fetch_ieee_fraud_detection(backend=pd, data_home=remote_data_home)
        _check_remote(ds, n_samples=590540, n_positives=20663, n_features=431, cost_keys={'amount'})

    def test_fetch_kdd98(self, remote_data_home):
        ds = fetch_kdd98(backend=pd, data_home=remote_data_home)
        _check_remote(ds, n_samples=191779, n_positives=9716, n_features=22, cost_keys={'amount'})
        donated = np.asarray(ds.target) == 1
        assert np.all(ds.instance_costs['amount'][donated] > 0)
        assert np.all(ds.instance_costs['amount'][~donated] == 0)

    def test_fetch_home_equity(self, remote_data_home):
        ds = fetch_home_equity(backend=pd, data_home=remote_data_home)
        _check_remote(ds, n_samples=5960, n_positives=1189, n_features=12, cost_keys={'cl', 'fp_cost'})

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_fetch_south_german_credit(self, remote_data_home, backend):
        ds = fetch_south_german_credit(backend=backend, data_home=remote_data_home)
        _check_remote(ds, n_samples=1000, n_positives=300, n_features=20, cost_keys={'cl', 'fp_cost'})

    def test_fetch_kddcup09_churn(self, remote_data_home):
        ds = fetch_kddcup09_churn(backend=pd, data_home=remote_data_home)
        _check_remote(ds, n_samples=50000, n_positives=3672, n_features=230, cost_keys={'clv'})
        np.testing.assert_array_equal(ds.instance_costs['clv'], 200.0)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_fetch_cell2cell(self, remote_data_home, backend):
        ds = fetch_cell2cell(backend=backend, data_home=remote_data_home)
        _check_remote(ds, n_samples=50891, n_positives=14641, n_features=56, cost_keys={'monthly_revenue'})
