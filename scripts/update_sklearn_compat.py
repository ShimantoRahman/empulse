"""
Update the scikit-learn compatibility tox environments in tox.ini.

Queries PyPI for all scikit-learn releases, finds the latest patch version for
each minor version that satisfies the minimum requirement declared in pyproject.toml,
and regenerates the [testenv:sklearnXYZ-tests] blocks.

Usage:
    python scripts/update_sklearn_compat.py
    python scripts/update_sklearn_compat.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
import urllib.request
from collections import defaultdict
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).parent.parent
TOX_INI = ROOT / 'tox.ini'
PYPROJECT = ROOT / 'pyproject.toml'

# Markers that delimit the auto-generated section in tox.ini
SECTION_START = '# >>> sklearn-compat-start (auto-generated, do not edit manually)\n'
SECTION_END = '# >>> sklearn-compat-end\n'

ENV_TEMPLATE = """\
[testenv:sklearn{tag}-tests]
runner = uv-venv-runner
basepython = python3.12
description = run tests against scikit-learn {version}
dependency_groups =
    test
deps =
    scikit-learn=={version}
commands =
    python setup.py build_ext --inplace
    pytest --basetemp={{envtmpdir}} -m "not slow" {{posargs}}
"""


def get_min_sklearn_version() -> Version:
    """Read the minimum scikit-learn version from pyproject.toml."""
    with open(PYPROJECT, 'rb') as f:
        data = tomllib.load(f)
    deps: list[str] = data['project']['dependencies']
    for dep in deps:
        if dep.startswith('scikit-learn'):
            # e.g. "scikit-learn>=1.5.2"
            match = re.search(r'>=\s*([\d.]+)', dep)
            if match:
                return Version(match.group(1))
    raise ValueError('Could not find scikit-learn minimum version in pyproject.toml')


def fetch_sklearn_versions() -> list[Version]:
    """Fetch all released scikit-learn versions from PyPI."""
    url = 'https://pypi.org/pypi/scikit-learn/json'
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    versions = []
    for v_str, files in data['releases'].items():
        # Skip pre-releases and yanked releases
        if not files:
            continue
        if any(f.get('yanked') for f in files):
            continue
        try:
            v = Version(v_str)
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        versions.append(v)
    return versions


def latest_patch_per_minor(versions: list[Version], min_version: Version) -> list[Version]:
    """For each minor version >= min_version, return the latest patch release."""
    by_minor: dict[tuple[int, int], Version] = defaultdict(lambda: Version('0'))
    for v in versions:
        if v < min_version:
            continue
        key = (v.major, v.minor)
        by_minor[key] = max(by_minor[key], v)
    return sorted(by_minor.values())


def version_tag(v: Version) -> str:
    """Convert a version like 1.5.2 to a tox env tag like 152."""
    return f'{v.major}{v.minor}{v.micro}'


def build_section(versions: list[Version]) -> str:
    blocks = [SECTION_START]
    for v in versions:
        blocks.append(ENV_TEMPLATE.format(tag=version_tag(v), version=str(v)))
    blocks.append(SECTION_END)
    return '\n'.join(blocks)


def update_tox_ini(versions: list[Version], dry_run: bool = False) -> None:
    content = TOX_INI.read_text(encoding='utf-8')

    new_section = build_section(versions)

    if SECTION_START in content and SECTION_END in content:
        # Replace existing auto-generated section
        pattern = re.escape(SECTION_START) + r'.*?' + re.escape(SECTION_END)
        new_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
    else:
        # Remove any manually written sklearn compat envs (old style) and append new section
        # Strip old [testenv:sklearnNNN-tests] blocks if present
        old_block_re = re.compile(r'\[testenv:sklearn\d+-tests\].*?(?=\[testenv:|$)', re.DOTALL)
        new_content = old_block_re.sub('', content).rstrip() + '\n\n' + new_section

    if dry_run:
        print('--- dry run: would write the following tox.ini ---')
        print(new_content)
    else:
        TOX_INI.write_text(new_content, encoding='utf-8')
        print(f'Updated {TOX_INI} with {len(versions)} sklearn compat environments:')
        for v in versions:
            print(f'  sklearn{version_tag(v)}-tests  (scikit-learn=={v})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print changes without writing')
    args = parser.parse_args()

    print('Reading minimum scikit-learn version from pyproject.toml...')
    min_ver = get_min_sklearn_version()
    print(f'  Minimum: scikit-learn>={min_ver}')

    print('Fetching scikit-learn releases from PyPI...')
    all_versions = fetch_sklearn_versions()
    print(f'  Found {len(all_versions)} releases')

    versions = latest_patch_per_minor(all_versions, min_ver)
    print(f'  Latest patch per minor (>={min_ver}): {[str(v) for v in versions]}')

    update_tox_ini(versions, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
