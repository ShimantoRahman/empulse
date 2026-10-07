"""Unit tests for empulse.datasets._io helpers."""

from __future__ import annotations

import csv
import gzip
import http.server
import threading
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from empulse.datasets import get_data_home
from empulse.datasets._io import (
    _download_to_file,
    _find_column,
    _read_csv_gz,
    _sanitize_column_name,
    _write_csv_gz,
    load_or_fetch,
    load_or_fetch_openml,
)

from ._helpers import BACKENDS


class TestSanitizeColumnName:
    @pytest.mark.parametrize(
        'raw, expected',
        [
            ('simple', 'simple'),
            ('UPPER', 'upper'),
            ('with space', 'with_space'),
            ('with  double  spaces', 'with_double_spaces'),
            ('has-hyphen', 'has_hyphen'),
            ('has.dot', 'has_dot'),
            ('has/slash', 'has_slash'),
            ('has(parens)', 'has_parens'),
            ('has:colon', 'has_colon'),
            ('has,comma', 'has_comma'),
            ("has'apostrophe", 'hasapostrophe'),
            ('  leading trailing  ', 'leading_trailing'),
            ('___underscores___', 'underscores'),
            ('multiple___underscores', 'multiple_underscores'),
            ('CamelCase', 'camelcase'),
            ('Customer Value', 'customer_value'),
            # accent stripping
            ('café', 'cafe'),
            ('naïve', 'naive'),
            # non-breaking space
            ('a\xa0b', 'a_b'),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert _sanitize_column_name(raw) == expected

    def test_strip_accents_false(self):
        result = _sanitize_column_name('café', strip_accents=False)
        assert result == 'café'

    def test_empty_string(self):
        # Empty input returns empty string (no crash)
        assert _sanitize_column_name('') == ''

    def test_only_special_chars(self):
        # All chars replaced / stripped → empty string
        assert _sanitize_column_name('---') == ''


class TestFindColumn:
    DATA: ClassVar[dict[str, list[int]]] = {'Alpha': [1], 'beta': [2], 'gamma_col': [3]}

    def test_first_candidate_wins(self):
        assert _find_column(self.DATA, ('Alpha', 'beta')) == 'Alpha'

    def test_second_candidate_fallback(self):
        assert _find_column(self.DATA, ('missing', 'beta')) == 'beta'

    def test_fallback_prefix(self):
        assert _find_column(self.DATA, ('nope',), fallback_prefix='gamma') == 'gamma_col'

    def test_fallback_prefix_case_insensitive(self):
        assert _find_column(self.DATA, ('nope',), fallback_prefix='GAMMA') == 'gamma_col'

    def test_raises_when_not_found(self):
        with pytest.raises(KeyError, match='nope'):
            _find_column(self.DATA, ('nope', 'also_nope'))

    def test_raises_with_no_fallback_match(self):
        with pytest.raises(KeyError):
            _find_column(self.DATA, ('nope',), fallback_prefix='xyz')


class TestReadWriteCsvGz:
    def _sample_data(self) -> dict:
        return {
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': ['1.0', '2.5', None],
            'flag': ['yes', 'no', 'yes'],
        }

    def test_round_trip(self, tmp_path):
        path = tmp_path / 'test.csv.gz'
        data = self._sample_data()
        _write_csv_gz(path, data)
        result = _read_csv_gz(path, null_values=[''])
        assert result['name'] == data['name']
        assert result['flag'] == data['flag']
        # None was written as '' and read back as None
        assert result['score'][2] is None

    def test_none_written_as_empty_string(self, tmp_path):
        path = tmp_path / 'none_test.csv.gz'
        _write_csv_gz(path, {'col': [None, 'value', None]})
        # Without null_values, reads back as empty string
        result = _read_csv_gz(path)
        assert result['col'][0] == ''
        assert result['col'][1] == 'value'

    def test_null_values_replaced(self, tmp_path):
        path = tmp_path / 'null_test.csv.gz'
        _write_csv_gz(path, {'a': ['N', 'hello', '']})
        result = _read_csv_gz(path, null_values=['N', ''])
        assert result['a'][0] is None
        assert result['a'][1] == 'hello'
        assert result['a'][2] is None

    def test_custom_delimiter(self, tmp_path):
        path = tmp_path / 'tsv.csv.gz'
        # Write tab-delimited manually
        with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['a', 'b'])
            writer.writerow(['1', '2'])
        result = _read_csv_gz(path, delimiter='\t')
        assert result['a'] == ['1']
        assert result['b'] == ['2']

    def test_empty_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / 'empty.csv.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write('')
        assert _read_csv_gz(path) == {}


class TestLoadOrFetch:
    def _make_fetcher(self, data: dict, call_count: list) -> dict[str, list[str]]:
        def fetcher():
            call_count.append(1)
            return data

        return fetcher

    def test_fetches_and_caches_when_missing(self, tmp_path):
        cache_file = tmp_path / 'data.csv.gz'
        call_count: list = []
        raw = {'x': ['1', '2'], 'y': ['a', 'b']}
        fetcher = self._make_fetcher(raw, call_count)

        result = load_or_fetch(cache_file, fetcher)

        assert len(call_count) == 1, 'fetcher should have been called exactly once'
        assert cache_file.exists(), 'cache file should have been created'
        assert result['x'] == ['1', '2']

    def test_reads_from_cache_on_second_call(self, tmp_path):
        cache_file = tmp_path / 'data.csv.gz'
        call_count: list = []
        raw = {'x': ['1', '2'], 'y': ['a', 'b']}
        fetcher = self._make_fetcher(raw, call_count)

        load_or_fetch(cache_file, fetcher)
        result2 = load_or_fetch(cache_file, fetcher)

        assert len(call_count) == 1, 'fetcher should only be called once across both calls'
        assert result2['x'] == ['1', '2']

    def test_raises_oserror_when_missing_and_not_allowed(self, tmp_path):
        cache_file = tmp_path / 'missing.csv.gz'
        with pytest.raises(OSError, match='download_if_missing'):
            load_or_fetch(cache_file, dict, download_if_missing=False)

    def test_error_message_includes_dataset_name(self, tmp_path):
        cache_file = tmp_path / 'missing.csv.gz'
        with pytest.raises(OSError, match='My Dataset'):
            load_or_fetch(cache_file, dict, download_if_missing=False, dataset_name='My Dataset')

    def test_none_values_survive_round_trip(self, tmp_path):
        cache_file = tmp_path / 'data.csv.gz'
        raw = {'col': ['a', None, 'c']}
        load_or_fetch(cache_file, lambda: raw)
        result = load_or_fetch(cache_file, dict)
        assert result['col'][1] is None

    def test_interrupted_write_leaves_no_cache_file(self, tmp_path):
        # Another process sharing the data home treats an existing cache file as complete, so a
        # write that fails part-way must not leave a truncated one behind -- nor its partial file.
        class Unwritable:
            def __str__(self):
                raise RuntimeError('interrupted')

        cache_file = tmp_path / 'data.csv.gz'
        with pytest.raises(RuntimeError, match='interrupted'):
            load_or_fetch(cache_file, lambda: {'col': ['a', Unwritable()]})
        assert list(tmp_path.iterdir()) == []


class TestLoadOrFetchOpenml:
    """The Parquet cache in front of OpenML; the network is patched out throughout."""

    FRAME = pl.DataFrame({'amount': [1.5, None], 'label': ['a', 'b']})

    @staticmethod
    def _write_frame(_url, destination):
        TestLoadOrFetchOpenml.FRAME.write_parquet(destination)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_downloads_once_then_reads_the_cache(self, tmp_path, backend):
        cache_file = tmp_path / 'data.parquet'
        with (
            patch('empulse.datasets._io._openml_parquet_url', return_value='url') as resolve,
            patch('empulse.datasets._io._download_to_file', side_effect=self._write_frame) as download,
        ):
            first = load_or_fetch_openml(cache_file, backend=backend, data_id=1)
            second = load_or_fetch_openml(cache_file, backend=backend, data_id=1)
        assert resolve.call_count == download.call_count == 1
        for df in (first, second):
            assert df.columns == ['amount', 'label']
            assert df['amount'].is_null().to_list() == [False, True]
            assert df['label'].to_list() == ['a', 'b']

    def test_raises_oserror_when_missing_and_not_allowed(self, tmp_path):
        with pytest.raises(OSError, match=r'My dataset not found.*download_if_missing'):
            load_or_fetch_openml(
                tmp_path / 'data.parquet', backend=pl, data_id=1, download_if_missing=False, dataset_name='My dataset'
            )

    def test_failed_download_names_the_dataset_and_leaves_no_file(self, tmp_path):
        def fail_halfway(_url, destination):
            destination.write_bytes(b'partial')
            raise OSError('connection reset')

        with (
            patch('empulse.datasets._io._openml_parquet_url', return_value='url'),
            patch('empulse.datasets._io._download_to_file', side_effect=fail_halfway),
            pytest.raises(OSError, match=r'My dataset.*connection reset'),
        ):
            load_or_fetch_openml(tmp_path / 'data.parquet', backend=pl, data_id=1, dataset_name='My dataset')
        assert list(tmp_path.iterdir()) == []

    def test_pandas_without_pyarrow_names_the_extra(self, tmp_path):
        self.FRAME.write_parquet(tmp_path / 'data.parquet')
        with (
            patch('empulse.datasets._io.importlib.util.find_spec', return_value=None),
            pytest.raises(ImportError, match=r'pip install empulse\[datasets\]'),
        ):
            load_or_fetch_openml(tmp_path / 'data.parquet', backend=pd, data_id=1)

    def test_polars_does_not_need_pyarrow(self, tmp_path):
        self.FRAME.write_parquet(tmp_path / 'data.parquet')
        with patch('empulse.datasets._io.importlib.util.find_spec', return_value=None):
            df = load_or_fetch_openml(tmp_path / 'data.parquet', backend=pl, data_id=1)
        assert df.columns == ['amount', 'label']


class _ShortBodyHandler(http.server.BaseHTTPRequestHandler):
    """Promise 1000 bytes and send 500, as a dropped connection would."""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', '1000')
        self.end_headers()
        self.wfile.write(b'x' * 500)

    def log_message(self, *args):
        pass


def test_download_to_file_rejects_a_truncated_body(tmp_path):
    server = http.server.HTTPServer(('127.0.0.1', 0), _ShortBodyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}/data.parquet'
        with pytest.raises(OSError, match='500 of 1000 bytes'):
            _download_to_file(url, tmp_path / 'data.parquet', n_retries=0)
    finally:
        server.shutdown()
        server.server_close()


class TestGetDataHome:
    """``get_data_home`` was the only name in the public API with no test reference at all."""

    def test_explicit_path_wins_and_is_created(self, tmp_path):
        target = tmp_path / 'explicit'
        assert not target.exists()
        assert get_data_home(target) == target
        assert target.is_dir()

    def test_environment_variable_is_used_when_no_path_is_given(self, tmp_path, monkeypatch):
        target = tmp_path / 'from_env'
        monkeypatch.setenv('EMPULSE_DATA_HOME', str(target))
        assert get_data_home() == target
        assert target.is_dir()

    def test_environment_variable_expands_the_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        monkeypatch.setenv('EMPULSE_DATA_HOME', '~/from_env')
        assert get_data_home() == tmp_path / 'from_env'

    def test_explicit_path_overrides_the_environment_variable(self, tmp_path, monkeypatch):
        monkeypatch.setenv('EMPULSE_DATA_HOME', str(tmp_path / 'from_env'))
        explicit = tmp_path / 'explicit'
        assert get_data_home(explicit) == explicit

    def test_defaults_to_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv('EMPULSE_DATA_HOME', raising=False)
        monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
        assert get_data_home() == tmp_path / 'empulse_data'

    def test_is_idempotent_on_an_existing_directory(self, tmp_path):
        target = tmp_path / 'twice'
        assert get_data_home(target) == get_data_home(target)
        assert target.is_dir()

    def test_accepts_a_string_path(self, tmp_path):
        target = tmp_path / 'as_string'
        assert get_data_home(str(target)) == target
        assert target.is_dir()
