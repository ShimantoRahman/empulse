"""Unit tests for empulse.datasets._io helpers."""

from __future__ import annotations

import csv
import gzip
from typing import ClassVar

import pytest

from empulse.datasets._io import (
    _find_column,
    _read_csv_gz,
    _sanitize_column_name,
    _write_csv_gz,
    load_or_fetch,
)


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
