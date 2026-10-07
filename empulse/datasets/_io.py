"""Internal I/O helpers for the dataset loaders."""

from __future__ import annotations

import csv
import gzip
import http.client
import importlib.util
import io
import json
import os
import re
import shutil
import ssl
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

import narwhals as nw
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


#: A lowercase letter or digit followed by an uppercase letter: ``monthlyRevenue``, ``id2Card``.
_CAMEL_BOUNDARY_RE = re.compile(r'([a-z0-9])([A-Z])')
#: The end of an acronym followed by a capitalised word: ``RVOwner``, ``USTravel``.
_ACRONYM_BOUNDARY_RE = re.compile(r'([A-Z]+)([A-Z][a-z])')


def _snake_case_column_name(name: str) -> str:
    """Normalize a CamelCase column name into snake_case.

    Like :func:`_sanitize_column_name`, but word boundaries marked only by capitalisation are
    turned into underscores first: ``PaymentMethod`` → ``payment_method``, ``StreamingTV`` →
    ``streaming_tv``, ``NonUSTravel`` → ``non_us_travel``, ``AgeHH1`` → ``age_hh1``.

    Parameters
    ----------
    name : str
        Raw column name.

    Returns
    -------
    str
        Clean snake_case column name.
    """
    name = _ACRONYM_BOUNDARY_RE.sub(r'\1_\2', _CAMEL_BOUNDARY_RE.sub(r'\1_\2', name.strip()))
    return _sanitize_column_name(name)


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
    # Write beside the cache file and move it into place, so that another process sharing the data
    # home sees either no file or a complete one -- never a half-written archive.
    partial_file = cache_file.with_name(f'{cache_file.name}.{os.getpid()}-{threading.get_ident()}.partial')
    try:
        _write_csv_gz(partial_file, raw)
        os.replace(partial_file, cache_file)
    finally:
        partial_file.unlink(missing_ok=True)
    # Re-read so callers always get the same string-only representation.
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
    # Query the metadata API.
    api_url = f'{_UCI_API_BASE}?id={dataset_id}'
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(api_url, context=ctx, timeout=30) as resp:
            metadata_json: dict[str, Any] = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, http.client.HTTPException) as exc:
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

    variables: list[dict[str, Any]] = metadata.get('variables', [])
    feature_names = [v['name'] for v in variables if v.get('role') == 'Feature']
    target_col_names = [v['name'] for v in variables if v.get('role') == 'Target']

    # Download the CSV, which is plain text or gzip-compressed.
    try:
        ctx2 = ssl.create_default_context()
        with urllib.request.urlopen(data_url, context=ctx2, timeout=60) as resp:
            raw_bytes = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, http.client.HTTPException) as exc:
        raise OSError(f'Failed to download UCI dataset {dataset_id} from {data_url}.  Original error: {exc}') from exc

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

    # The API may not return roles.
    if not feature_names:
        feature_names = [c for c in all_cols if c not in target_col_names]
    if not target_col_names:
        target_col_names = [c for c in all_cols if c not in feature_names]

    features = {col: raw[col] for col in feature_names if col in raw}
    targets = {col: raw[col] for col in target_col_names if col in raw}
    return features, targets


def _fetch_csv_url(
    urls: str | list[str],
    *,
    timeout: int = 60,
) -> dict[str, list[str | None]]:
    """
    Download a CSV from one or more URLs and return a column dict.

    Parameters
    ----------
    urls : str or list of str
        Candidate URLs to try in order.
    timeout : int, default=60
        Request timeout in seconds.

    Returns
    -------
    dict[str, list[str | None]]
        Column-oriented dictionary of raw string values.
    """
    url_list = [urls] if isinstance(urls, str) else urls
    last_exc: Exception | None = None
    ctx = ssl.create_default_context()
    for url in url_list:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'empulse'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                raw_bytes = resp.read()
            try:
                with gzip.open(io.BytesIO(raw_bytes), 'rt', encoding='utf-8') as gz:
                    content = gz.read()
            except OSError:
                content = raw_bytes.decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            if not rows:
                continue
            cols = list(rows[0].keys())
            return {col: [row[col] for row in rows] for col in cols}
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, http.client.HTTPException) as exc:
            last_exc = exc
            continue
    raise OSError(f'Failed to download CSV from {url_list}. Original error: {last_exc}')


def _fetch_url_bytes(url: str, *, timeout: int = 120) -> bytes:
    """
    Download *url* and return the raw response body.

    Parameters
    ----------
    url : str
        URL to download.
    timeout : int, default=120
        Request timeout in seconds.

    Returns
    -------
    bytes
        The undecoded response body.
    """
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'empulse'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            body: bytes = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, http.client.HTTPException) as exc:
        raise OSError(f'Failed to download {url}. Original error: {exc}') from exc
    return body


def _read_csv_columns(
    raw_bytes: bytes,
    columns: Sequence[str],
    *,
    encoding: str = 'latin-1',
) -> dict[str, list[str]]:
    """
    Read selected columns from a CSV, optionally wrapped in a single-member zip archive.

    Only the requested columns are kept, so wide files (hundreds of columns) can be read
    without materialising every cell.

    Parameters
    ----------
    raw_bytes : bytes
        Contents of a CSV file, or of a zip archive whose first member is a CSV file.
    columns : sequence of str
        Header names of the columns to keep.
    encoding : str, default='latin-1'
        Text encoding of the CSV.

    Returns
    -------
    dict[str, list[str]]
        Column-oriented mapping: header → list of raw string values.

    Raises
    ------
    KeyError
        When a requested column is not in the header.
    """
    if zipfile.is_zipfile(io.BytesIO(raw_bytes)):
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            raw_bytes = archive.read(archive.namelist()[0])
    reader = csv.reader(io.StringIO(raw_bytes.decode(encoding)))
    header = [name.strip() for name in next(reader)]
    missing = [col for col in columns if col not in header]
    if missing:
        raise KeyError(f'Columns {missing} not found. Available columns: {header}')
    indices = [header.index(col) for col in columns]
    result: dict[str, list[str]] = {col: [] for col in columns}
    for row in reader:
        if not row:
            continue
        for col, idx in zip(columns, indices, strict=True):
            result[col].append(row[idx])
    return result


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
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last_exc = exc

        if attempt < n_retries:
            time.sleep(delay)

    raise OSError(
        f'Failed to reach the OpenML API at {url} after {n_retries + 1} attempts. Last error: {last_exc}'
    ) from last_exc


def _openml_parquet_url(
    name: str | None = None,
    *,
    version: int | str = 'active',
    data_id: int | None = None,
    n_retries: int = 3,
    delay: float = 1.0,
) -> str:
    """Resolve an OpenML dataset to the URL of its Parquet file.

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
    str
        Download URL of the dataset's Parquet file.

    Raises
    ------
    ValueError
        When neither or both of *name* / *data_id* are given.
    OSError
        When the dataset cannot be found or has no Parquet file.
    OpenMLError
        When OpenML returns a HTTP 412 error.
    """
    if name is None and data_id is None:
        raise ValueError('Either name or data_id must be provided.')
    if name is not None and data_id is not None:
        raise ValueError('Provide either name or data_id, not both.')

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

    desc_json = _openml_api_request(_OPENML_DATA_INFO.format(data_id), n_retries=n_retries, delay=delay)
    description: dict[str, Any] = desc_json.get('data_set_description', {})

    parquet_url: str | None = description.get('parquet_url')
    if not parquet_url:
        raise OSError(f'OpenML dataset {data_id} exists but has no downloadable Parquet file.')

    if description.get('status') != 'active':
        warnings.warn(
            f'OpenML dataset {data_id} ({description.get("name")!r}) '
            f'has status {description.get("status")!r}; it may have known issues.',
            RuntimeWarning,
            stacklevel=3,
        )
    return parquet_url


def _download_to_file(url: str, destination: Path, *, n_retries: int = 3, delay: float = 1.0) -> None:
    """Stream *url* into *destination*, retrying on network errors.

    Parameters
    ----------
    url : str
        URL to download.
    destination : Path
        File to write; overwritten on every attempt.
    n_retries : int, default=3
        Number of retry attempts on transient network errors.
    delay : float, default=1.0
        Seconds to wait between retry attempts.

    Raises
    ------
    OSError
        When every attempt fails.
    """
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'empulse'})
    last_exc: Exception | None = None
    for attempt in range(n_retries + 1):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp, open(destination, 'wb') as f:
                shutil.copyfileobj(resp, f)
                expected_size = resp.headers.get('Content-Length')
                received_size = f.tell()
            # A dropped connection ends a streamed body early without raising.
            if expected_size is not None and int(expected_size) != received_size:
                raise OSError(f'Received {received_size} of {expected_size} bytes.')
            return
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last_exc = exc
            if attempt < n_retries:
                time.sleep(delay)
    raise OSError(f'Failed to download {url} after {n_retries + 1} attempts. Last error: {last_exc}') from last_exc


def _require_parquet_reader(backend: Any) -> None:
    """Raise :exc:`ImportError` when *backend* needs pyarrow to read Parquet and it is missing.

    Polars reads Parquet natively; every other backend goes through pyarrow.
    """
    implementation = nw.Implementation.from_backend(backend)
    if implementation is nw.Implementation.POLARS or importlib.util.find_spec('pyarrow') is not None:
        return
    raise ImportError(
        f'Loading this dataset with the {implementation} backend requires pyarrow. '
        'Install it with `pip install empulse[datasets]` or `pip install pyarrow`, '
        'or pass `backend=polars`.'
    )


def load_or_fetch_openml(
    cache_file: Path,
    *,
    backend: Any,
    name: str | None = None,
    version: int | str = 'active',
    data_id: int | None = None,
    download_if_missing: bool = True,
    dataset_name: str = 'dataset',
) -> nw.DataFrame[Any]:
    """Load an OpenML dataset from its cached Parquet file, downloading it first if needed.

    The file is cached exactly as OpenML serves it, so columns keep their OpenML types: numeric
    attributes are numbers, nominal ones strings or categoricals, and missing values are null.

    Parameters
    ----------
    cache_file : Path
        Path to the ``.parquet`` cache file.
    backend : module
        Narwhals-compatible dataframe backend to read the file with.
    name : str, optional
        Dataset name on OpenML.  Either *name* or *data_id* must be given.
    version : int or 'active', default='active'
        Dataset version.  Only used together with *name*.
    data_id : int, optional
        OpenML numeric dataset ID.  Either *name* or *data_id* must be given.
    download_if_missing : bool, default=True
        When *False* and the cache file does not exist, raise an
        :exc:`OSError` instead of downloading.
    dataset_name : str, default='dataset'
        Human-readable name used in error messages.

    Returns
    -------
    narwhals.DataFrame
        The dataset as OpenML stores it.

    Raises
    ------
    ImportError
        When the backend needs pyarrow to read Parquet and it is not installed.
    OSError
        When *download_if_missing* is *False* and the cache is absent, or the download fails.
    """
    _require_parquet_reader(backend)
    if not cache_file.exists():
        if not download_if_missing:
            raise OSError(
                f'{dataset_name} not found at {cache_file}. Set download_if_missing=True to download it automatically.'
            )
        # Download beside the cache file and move it into place, so that another process sharing the
        # data home sees either no file or a complete one.
        partial_file = cache_file.with_name(f'{cache_file.name}.{os.getpid()}-{threading.get_ident()}.partial')
        try:
            url = _openml_parquet_url(name, version=version, data_id=data_id)
            _download_to_file(url, partial_file)
            os.replace(partial_file, cache_file)
        except (OSError, OpenMLError) as exc:
            raise OSError(f'Failed to download the {dataset_name} from OpenML. Original error: {exc}') from exc
        finally:
            partial_file.unlink(missing_ok=True)
    return nw.read_parquet(cache_file, backend=backend)
