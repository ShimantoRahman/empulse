"""Internal I/O helpers — stdlib + numpy only, no dataframe library required."""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import numpy as np

#: Characters that are replaced by a single underscore (includes non-breaking space \xa0).
_NORMALIZE_RE = re.compile(r'[ /:,?()\.\-\xa0]+')
#: Characters that are removed entirely (apostrophes, backticks, …).
_REMOVE_RE = re.compile(r"['\u2019`]")
#: Runs of two or more underscores collapsed to one.
_MULTI_UNDERSCORE_RE = re.compile(r'_+')


def _sanitize_column_name(name: str, *, strip_accents: bool = True) -> str:
    """Normalize a raw column name into a clean snake_case identifier.

    Steps applied (inspired by pyjanitor's ``clean_names``):

    1. Strip leading/trailing whitespace.
    2. Optionally decompose Unicode accents and drop combining characters.
    3. Replace punctuation / separators (spaces, ``/``, ``:``, ``(``, ``)``
       ``?``, ``,``, ``.``, ``-``) with ``_``.
    4. Remove apostrophes and other decoration characters entirely.
    5. Lowercase everything.
    6. Collapse consecutive underscores to a single ``_``.
    7. Strip any remaining leading or trailing ``_``.

    Parameters
    ----------
    name : str
        Raw column name.
    strip_accents : bool, default True
        If *True*, decompose accented characters (e.g. ``é`` → ``e``).

    Returns
    -------
    str
        Clean snake_case column name.
    """
    name = name.strip()
    if strip_accents:
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if not unicodedata.combining(c))
    name = _NORMALIZE_RE.sub('_', name)
    name = _REMOVE_RE.sub('', name)
    name = name.lower()
    name = _MULTI_UNDERSCORE_RE.sub('_', name)
    name = name.strip('_')
    return name


def _read_csv_gz(
    path: str | Path,
    delimiter: str = ',',
    null_values: list[str] | None = None,
) -> dict[str, list[str | None]]:
    """Read a gzip-compressed CSV into an ordered dict of raw string lists.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv.gz`` file.
    delimiter : str, default=','
        Field delimiter.
    null_values : list of str, optional
        Values to replace with ``None`` (treated as missing).
        E.g. ``['N', '', '?']``.

    Returns
    -------
    dict[str, list[str | None]]
        Column-oriented mapping: header → list of raw string values (or
        ``None`` for values in *null_values*).
    """
    null_set: frozenset[str] = frozenset(null_values) if null_values else frozenset()
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return {}
    columns = list(rows[0].keys())
    if null_set:
        return {col: [None if row[col] in null_set else row[col] for row in rows] for col in columns}
    return {col: [row[col] for row in rows] for col in columns}


def _write_csv_gz(path: str | Path, data: dict[str, Any]) -> None:
    """Write a column-oriented dict to a gzip-compressed CSV (stdlib only).

    Parameters
    ----------
    path : str or Path
        Destination path.
    data : dict[str, array-like]
        Column-oriented data to write.  Values are converted to ``str``.
        ``None`` values are written as empty strings and will be read back
        as ``None`` when ``null_values=['']`` is passed to :func:`_read_csv_gz`.
    """
    columns = list(data.keys())
    n = len(next(iter(data.values())))
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(n):
            writer.writerow({col: '' if data[col][i] is None else str(data[col][i]) for col in columns})


def _find_column(
    data: dict[str, Any],
    candidates: tuple[str, ...],
    fallback_prefix: str | None = None,
) -> str:
    """Return the first candidate column name that exists in *data*.

    Parameters
    ----------
    data : dict
        Column-oriented data mapping.
    candidates : tuple of str
        Column names to try, in priority order.
    fallback_prefix : str, optional
        If no candidate matches, return the first column whose name starts
        with this prefix (case-insensitive).

    Raises
    ------
    KeyError
        When no candidate and no fallback match is found.
    """
    for name in candidates:
        if name in data:
            return name
    if fallback_prefix is not None:
        for col in data:
            if col.lower().startswith(fallback_prefix.lower()):
                return col
    raise KeyError(f'Could not find any of the expected columns {candidates}. Available columns: {list(data.keys())}')


def load_or_fetch(
    cache_file: Path,
    fetcher: Callable[[], dict[str, list[Any]]],
    *,
    download_if_missing: bool = True,
    dataset_name: str = 'dataset',
) -> dict[str, list[str | None]]:
    """Load a cached dataset or fetch it and write to cache.

    The cache is stored as a gzip-compressed CSV so that no dataframe
    library is needed at cache time.

    Parameters
    ----------
    cache_file : Path
        Path to the ``.csv.gz`` cache file.
    fetcher : callable
        Zero-argument callable that downloads the raw data and returns a
        column-oriented ``dict[str, list]``.
    download_if_missing : bool, default=True
        When *False* and the cache file does not exist, raise an
        :exc:`OSError` instead of downloading.
    dataset_name : str, default='dataset'
        Human-readable name used in the error message.

    Returns
    -------
    dict[str, list[str | None]]
        Column-oriented dict of raw string values suitable for
        :func:`narwhals.from_dict`.

    Raises
    ------
    OSError
        When *download_if_missing* is *False* and the cache is absent.
    """
    if cache_file.exists():
        return _read_csv_gz(cache_file, null_values=[''])
    if not download_if_missing:
        raise OSError(
            f'{dataset_name} not found at {cache_file}. Set download_if_missing=True to download it automatically.'
        )
    raw = fetcher()
    _write_csv_gz(cache_file, raw)
    # Re-read so callers always get the same string-only representation
    return _read_csv_gz(cache_file, null_values=[''])


_UCI_API_BASE = 'https://archive.ics.uci.edu/api/dataset'


def _fetch_uci(dataset_id: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fetch a UCI ML Repository dataset and return ``(features, targets)``.

    Uses only Python stdlib (urllib, json, csv, ssl) and numpy.

    Parameters
    ----------
    dataset_id : int
        UCI dataset numeric ID.

    Returns
    -------
    features : dict[str, numpy.ndarray]
        Column-oriented dict of feature arrays (string dtype).
    targets : dict[str, numpy.ndarray]
        Column-oriented dict of target arrays (string dtype).
    """
    # ------------------------------------------------------------------
    # Step 1 — query the metadata API
    # ------------------------------------------------------------------
    api_url = f'{_UCI_API_BASE}?id={dataset_id}'
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(api_url, context=ctx, timeout=30) as resp:
            metadata_json: dict[str, Any] = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        raise OSError(
            f'Failed to reach the UCI ML Repository API for dataset {dataset_id}. '
            f'Check your internet connection.  Original error: {exc}'
        ) from exc

    if metadata_json.get('status') != 200:
        msg = metadata_json.get('message', 'Dataset not found')
        raise OSError(f'UCI API error for dataset {dataset_id}: {msg}')

    metadata = metadata_json['data']
    data_url: str | None = metadata.get('data_url')
    if not data_url:
        raise OSError(
            f'UCI dataset {dataset_id} exists but has no downloadable CSV.  '
            'See https://archive.ics.uci.edu/datasets for available datasets.'
        )

    # Determine feature vs. target column names from variable metadata
    variables: list[dict[str, Any]] = metadata.get('variables', [])
    feature_names = [v['name'] for v in variables if v.get('role') == 'Feature']
    target_col_names = [v['name'] for v in variables if v.get('role') == 'Target']

    # ------------------------------------------------------------------
    # Step 2 — download the CSV (may be plain text or gzip-compressed)
    # ------------------------------------------------------------------
    try:
        ctx2 = ssl.create_default_context()
        with urllib.request.urlopen(data_url, context=ctx2, timeout=60) as resp:
            raw_bytes = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        raise OSError(f'Failed to download UCI dataset {dataset_id} from {data_url}.  Original error: {exc}') from exc

    # Try gzip first, fall back to plain text
    try:
        with gzip.open(io.BytesIO(raw_bytes), 'rt', encoding='utf-8') as gz:
            content = gz.read()
    except OSError:
        content = raw_bytes.decode('utf-8')

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        raise OSError(f'Downloaded empty CSV for UCI dataset {dataset_id}.')

    all_cols = list(rows[0].keys())
    raw: dict[str, np.ndarray] = {col: np.array([row[col] for row in rows]) for col in all_cols}

    # Fall back if the API didn't return role information
    if not feature_names:
        feature_names = [c for c in all_cols if c not in target_col_names]
    if not target_col_names:
        target_col_names = [c for c in all_cols if c not in feature_names]

    features = {col: raw[col] for col in feature_names if col in raw}
    targets = {col: raw[col] for col in target_col_names if col in raw}
    return features, targets


_OPENML_API_BASE = 'https://api.openml.org/api/v1/json'
_OPENML_SEARCH_NAME = _OPENML_API_BASE + '/data/list/data_name/{}/limit/2'
_OPENML_DATA_INFO = _OPENML_API_BASE + '/data/{}'
_OPENML_DATA_FEATURES = _OPENML_API_BASE + '/data/features/{}'


class OpenMLError(ValueError):
    """OpenML API error (HTTP 412 — OpenML-specific generic error)."""


def _openml_api_request(
    url: str,
    n_retries: int = 3,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Make a GET request to the OpenML JSON API with retry logic.

    Retries on network errors (URLError, TimeoutError, OSError).
    Does *not* retry on HTTP 412 (OpenML-specific error code).

    Parameters
    ----------
    url : str
        OpenML API endpoint URL.
    n_retries : int, default=3
        Number of additional attempts after the first failure.
    delay : float, default=1.0
        Seconds to wait between attempts.

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    OpenMLError
        When the server returns HTTP 412.
    OSError
        When all retry attempts are exhausted.
    """
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url)
    req.add_header('Accept-encoding', 'gzip')

    last_exc: Exception | None = None
    for attempt in range(n_retries + 1):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                data: bytes = resp.read()
                if resp.info().get('Content-Encoding', '') == 'gzip':
                    data = gzip.decompress(data)
                result: dict[str, Any] = json.loads(data.decode('utf-8'))
                return result
        except urllib.error.HTTPError as exc:
            if exc.code == 412:
                raise OpenMLError(
                    f'OpenML returned HTTP 412 for {url}. This usually means the requested resource does not exist.'
                ) from exc
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc

        if attempt < n_retries:
            time.sleep(delay)

    raise OSError(
        f'Failed to reach the OpenML API at {url} after {n_retries + 1} attempts. Last error: {last_exc}'
    ) from last_exc


def _parse_arff(content: str) -> dict[str, list[str]]:
    """Parse a dense ARFF string into a column-oriented dict of raw strings.

    Missing values (``?`` in ARFF) are preserved as the string ``'?'``;
    callers are responsible for treating them as nulls.

    Parameters
    ----------
    content : str
        Full text content of an ARFF file.

    Returns
    -------
    dict[str, list[str]]
        Column-oriented mapping: attribute name → list of raw string values.
    """
    attributes: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        lower = stripped.lower()
        if lower.startswith('@data'):
            in_data = True
        elif lower.startswith('@attribute'):
            # @ATTRIBUTE <name> <type>   — name may be quoted
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                attributes.append(parts[1].strip('\'"'))
        elif in_data:
            reader = csv.reader([stripped])
            data_rows.append([v.strip() for v in next(reader)])

    if not data_rows:
        return {name: [] for name in attributes}

    result: dict[str, list[str]] = {name: [] for name in attributes}
    for row in data_rows:
        for i, name in enumerate(attributes):
            result[name].append(row[i] if i < len(row) else '')
    return result


def _fetch_openml(
    name: str | None = None,
    *,
    version: int | str = 'active',
    data_id: int | None = None,
    n_retries: int = 3,
    delay: float = 1.0,
) -> dict[str, list[str]]:
    """Fetch a dataset from OpenML and return a flat dict of raw string lists.

    Uses only Python stdlib (``urllib``, ``json``, ``csv``, ``ssl``, ``gzip``)
    — no extra dependencies required.  Includes the same retry logic that
    scikit-learn uses in ``fetch_openml``.

    Parameters
    ----------
    name : str, optional
        Dataset name on OpenML.  Either *name* or *data_id* must be given.
    version : int or 'active', default='active'
        Dataset version.  Only used together with *name*.
    data_id : int, optional
        OpenML numeric dataset ID.  Either *name* or *data_id* must be given.
    n_retries : int, default=3
        Number of retry attempts on transient network errors.
    delay : float, default=1.0
        Seconds to wait between retry attempts.

    Returns
    -------
    dict[str, list[str]]
        Column-oriented dict of raw string values suitable for
        :func:`_write_csv_gz` / :func:`_read_csv_gz`.
        Missing ARFF values (``?``) are preserved as the string ``'?'``.

    Raises
    ------
    ValueError
        When neither or both of *name* / *data_id* are given.
    OSError
        When the dataset cannot be found or downloaded.
    OpenMLError
        When OpenML returns a HTTP 412 error.
    """
    if name is None and data_id is None:
        raise ValueError('Either name or data_id must be provided.')
    if name is not None and data_id is not None:
        raise ValueError('Provide either name or data_id, not both.')

    # ------------------------------------------------------------------
    # Step 1 — resolve data_id from name + version
    # ------------------------------------------------------------------
    if name is not None:
        name_lower = name.lower()
        if version == 'active':
            url = _OPENML_SEARCH_NAME.format(name_lower) + '/status/active/'
        else:
            url = _OPENML_SEARCH_NAME.format(name_lower) + f'/data_version/{version}'

        try:
            json_data = _openml_api_request(url, n_retries=n_retries, delay=delay)
        except OpenMLError:
            if version != 'active':
                url += '/status/deactivated'
                json_data = _openml_api_request(url, n_retries=n_retries, delay=delay)
            else:
                raise

        datasets_list = json_data.get('data', {}).get('dataset', [])
        if not datasets_list:
            raise OSError(f'No OpenML dataset found with name={name!r}, version={version!r}.')
        data_id = int(datasets_list[0]['did'])

    # ------------------------------------------------------------------
    # Step 2 — get dataset description (contains the ARFF download URL)
    # ------------------------------------------------------------------
    desc_url = _OPENML_DATA_INFO.format(data_id)
    desc_json = _openml_api_request(desc_url, n_retries=n_retries, delay=delay)
    description: dict[str, Any] = desc_json.get('data_set_description', {})

    arff_url: str | None = description.get('url')
    if not arff_url:
        raise OSError(f'OpenML dataset {data_id} exists but has no downloadable ARFF URL.')

    if description.get('status') != 'active':
        import warnings

        warnings.warn(
            f'OpenML dataset {data_id} ({description.get("name")!r}) '
            f'has status {description.get("status")!r}; it may have known issues.',
            RuntimeWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------
    # Step 3 — download the ARFF file with retry logic
    # ------------------------------------------------------------------
    ctx = ssl.create_default_context()
    arff_req = urllib.request.Request(arff_url)
    arff_req.add_header('Accept-encoding', 'gzip')

    last_exc: Exception | None = None
    arff_bytes: bytes | None = None
    for attempt in range(n_retries + 1):
        try:
            with urllib.request.urlopen(arff_req, context=ctx, timeout=120) as resp:
                chunk: bytes = resp.read()
                if resp.info().get('Content-Encoding', '') == 'gzip':
                    chunk = gzip.decompress(chunk)
                arff_bytes = chunk
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < n_retries:
                time.sleep(delay)

    if arff_bytes is None:
        raise OSError(
            f'Failed to download ARFF for OpenML dataset {data_id} from {arff_url} '
            f'after {n_retries + 1} attempts. Last error: {last_exc}'
        ) from last_exc

    # ------------------------------------------------------------------
    # Step 4 — decompress (the ARFF file itself may be gzip-compressed)
    #          and parse
    # ------------------------------------------------------------------
    try:
        with gzip.open(io.BytesIO(arff_bytes), 'rt', encoding='utf-8') as gz:
            content = gz.read()
    except OSError:
        try:
            content = arff_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content = arff_bytes.decode('latin-1')

    return _parse_arff(content)
