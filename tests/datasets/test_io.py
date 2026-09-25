"""Unit tests for empulse.datasets._io helpers."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import ClassVar

import pytest

from empulse.datasets import get_data_home
from empulse.datasets._io import (
    _find_column,
    _parse_arff,
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


class TestParseArff:
    def test_single_quoted_values_lose_their_quotes(self):
        content = (
            '@RELATION telco\n'
            "@ATTRIBUTE Contract {Month-to-month,'One year'}\n"
            '@ATTRIBUTE MonthlyCharges NUMERIC\n'
            '@ATTRIBUTE Churn {No,Yes}\n'
            '@DATA\n'
            'Month-to-month,29.85,No\n'
            "'One year',56.95,Yes\n"
        )
        assert _parse_arff(content) == {
            'Contract': ['Month-to-month', 'One year'],
            'MonthlyCharges': ['29.85', '56.95'],
            'Churn': ['No', 'Yes'],
        }

    def test_quoted_value_may_contain_a_comma_and_an_escaped_quote(self):
        content = "@ATTRIBUTE name STRING\n@ATTRIBUTE n NUMERIC\n@DATA\n'O\\'Brien, Jr.',1\n?,2\n"
        assert _parse_arff(content) == {'name': ["O'Brien, Jr.", '?'], 'n': ['1', '2']}


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
