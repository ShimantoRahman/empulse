# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///
"""
Report the latest scikit-learn release and how its interpreter support lines up with ours.

Used by `.github/workflows/sklearn-watch.yml`. Emits a single JSON object on stdout:

    {
      "version": "1.9.1",
      "sklearn_pythons": ["3.11", ..., "3.15"],
      "empulse_pythons": ["3.11", ..., "3.14"],
      "sklearn_only": ["3.15"],
      "empulse_only": [],
      "matrix": [{"os": "ubuntu-latest", "python": "3.11", "env": "py311-sklearn-pinned"}, ...]
    }

Two questions, because a scikit-learn release can break this package in two unrelated ways.

*Does it still work?* -- answered by the `matrix`, which the workflow tests against. The interpreters
are derived rather than hardcoded, because a hardcoded one goes wrong in both directions: pinning
scikit-learn 1.8.0 on Python 3.10 does not resolve at all (1.8.0 dropped it) and reports as a test
failure when nothing is actually broken, while a hardcoded ceiling silently never tests the newest
interpreter scikit-learn just started shipping wheels for.

*Did its interpreter support move?* -- answered by `sklearn_only`/`empulse_only`, and that is news
in its own right. scikit-learn 1.7.2 -> 1.8.0 dropped cp310; 1.9.1 added cp315 while this package's
classifiers still stopped at 3.14. Neither shows up as a test failure, and both are things a
maintainer wants to act on.

Release candidates are included: scikit-learn publishes an `rc` weeks before the matching final
release, and that gap is the window in which an incompatibility can still be reported upstream.

Usage:
    python scripts/check_sklearn_release.py
    python scripts/check_sklearn_release.py --version 1.9.0rc1
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / 'pyproject.toml'
PYPI_URL = 'https://pypi.org/pypi/scikit-learn/json'

# Matches the interpreter tag of a CPython wheel: the `cp313` of
# `scikit_learn-1.9.1-cp313-cp313-manylinux_2_17_x86_64.whl`, and the `cp314` of a `cp314-cp314t`
# free-threaded build. Two digits for 3.9, three from 3.10 onwards.
_CP_TAG_RE = re.compile(r'-cp(\d)(\d{1,2})-')


def fetch_release_data() -> dict:
    """Fetch scikit-learn's full release index from PyPI."""
    with urllib.request.urlopen(PYPI_URL, timeout=30) as resp:
        return json.loads(resp.read())


def latest_version(data: dict) -> str:
    """Return the highest non-yanked, non-dev scikit-learn version, release candidates included."""
    versions = []
    for v_str, files in data['releases'].items():
        if not files or any(f.get('yanked') for f in files):
            continue
        try:
            v = Version(v_str)
        except InvalidVersion:
            continue
        if v.is_devrelease:
            continue
        versions.append(v)
    if not versions:
        raise SystemExit('No scikit-learn releases found on PyPI')
    return str(max(versions))


def sklearn_pythons(data: dict, version: str) -> list[str]:
    """Return the CPython versions a scikit-learn release publishes wheels for, as `3.x` strings.

    Read from the wheel tags rather than from `requires_python`, which only gives a lower bound and
    so cannot say which interpreters actually got a build.
    """
    files = data['releases'].get(version)
    if not files:
        raise SystemExit(f'scikit-learn {version} is not on PyPI')

    found = set()
    for f in files:
        match = _CP_TAG_RE.search(f['filename'])
        if match:
            found.add((int(match.group(1)), int(match.group(2))))
    return [f'{major}.{minor}' for major, minor in sorted(found)]


def empulse_pythons() -> list[str]:
    """Return the CPython versions this package claims to support, from its trove classifiers.

    The classifiers are the authoritative list -- `requires-python` gives a floor but no ceiling --
    and they are what a maintainer edits when adding support for a new interpreter.
    """
    with open(PYPROJECT, 'rb') as f:
        data = tomllib.load(f)

    found = set()
    for classifier in data['project']['classifiers']:
        match = re.fullmatch(r'Programming Language :: Python :: (\d)\.(\d{1,2})', classifier)
        if match:
            found.add((int(match.group(1)), int(match.group(2))))
    if not found:
        raise SystemExit('No `Programming Language :: Python :: X.Y` classifiers in pyproject.toml')
    return [f'{major}.{minor}' for major, minor in sorted(found)]


def build_matrix(shared: list[str]) -> list[dict[str, str]]:
    """Build the GitHub Actions matrix legs for the interpreters both projects support.

    Brackets the shared range rather than covering all of it: the oldest and newest interpreters are
    where a release's support window moves, and the legs in between rarely fail alone. Windows earns
    a leg of its own because the Cython ABI mismatch this package is exposed to surfaces there as an
    access violation rather than as a Python exception.
    """
    if not shared:
        return []

    def leg(os_name: str, python: str) -> dict[str, str]:
        return {'os': os_name, 'python': python, 'env': f'py{python.replace(".", "")}-sklearn-pinned'}

    oldest, newest = shared[0], shared[-1]
    matrix = [leg('ubuntu-latest', oldest)]
    if newest != oldest:
        matrix.append(leg('ubuntu-latest', newest))
    matrix.append(leg('windows-latest', newest))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', help='inspect this version instead of the latest on PyPI')
    args = parser.parse_args()

    data = fetch_release_data()
    version = args.version or latest_version(data)

    theirs = sklearn_pythons(data, version)
    ours = empulse_pythons()
    shared = [p for p in ours if p in theirs]

    report = {
        'version': version,
        'sklearn_pythons': theirs,
        'empulse_pythons': ours,
        # Named by side rather than as added/dropped, which would only be true reading forwards:
        # `--version` can inspect an older release, where the same asymmetry means the opposite. Both
        # sides are worth reporting -- the first is support we could pick up, the second is support
        # we advertise that our main dependency does not have.
        'sklearn_only': [p for p in theirs if p not in ours],
        'empulse_only': [p for p in ours if p not in theirs],
        'matrix': build_matrix(shared),
    }
    print(json.dumps(report))


if __name__ == '__main__':
    main()
