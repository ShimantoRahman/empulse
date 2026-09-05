"""
Check that the installed Cython matches the Cython scikit-learn's compiled extensions were built with.

Background
----------
`empulse/models/cost_sensitive/_impurity/cost_impurity.pyx` cimports
`sklearn.tree._criterion.ClassificationCriterion` and subclasses it, inheriting several
C-level typed-memoryview attributes (e.g. ``n_classes``) directly from scikit-learn's precompiled
extension. Cython's binary representation for typed memoryviews is not guaranteed to be ABI-stable
across Cython *minor* versions. If our extension is compiled with a different Cython minor version
than the one scikit-learn's wheel was built with, inherited memoryview attributes can be read at
the wrong offset - this doesn't raise a Python exception, it corrupts memory and crashes hard (a
"Windows fatal exception: access violation" / segfault), exactly the failure this script exists to
catch before it reaches a release.

There is no official metadata recording the exact Cython version a wheel was built with (PyPI
metadata only records the loose ``cython>=X`` build requirement). The most reliable available
signal is the version tag Cython embeds into every compiled extension's binary for its internal
per-version runtime module (e.g. the literal bytes ``cython_3_2_5``). This script extracts that tag
from one of scikit-learn's compiled Cython extensions and compares it against the Cython version
installed in this environment (the one that will be used to (re)compile empulse's own extensions).

Usage:
    python scripts/check_cython_abi.py
"""

from __future__ import annotations

import re
import sys

_VERSION_TAG_RE = re.compile(rb'cython_(\d+)_(\d+)_(\d+)')


def _extract_embedded_cython_versions(binary_path: str) -> set[tuple[int, int, int]]:
    """Return every Cython version tag embedded in the compiled extension at *binary_path*."""
    with open(binary_path, 'rb') as f:
        data = f.read()
    return {(int(a), int(b), int(c)) for a, b, c in _VERSION_TAG_RE.findall(data)}


def main() -> int:
    try:
        import Cython
    except ImportError:
        print('Cython is not installed - skipping Cython ABI compatibility check.')
        return 0

    try:
        import sklearn.tree._criterion as sklearn_criterion
    except ImportError:
        print('scikit-learn is not installed - skipping Cython ABI compatibility check.')
        return 0

    installed = Cython.__version__
    installed_match = re.match(r'(\d+)\.(\d+)\.(\d+)', installed)
    if installed_match is None:
        print(f'Could not parse installed Cython version {installed!r} - skipping check.')
        return 0
    installed_tuple = tuple(int(x) for x in installed_match.groups())

    binary_path = sklearn_criterion.__file__
    sklearn_tags = _extract_embedded_cython_versions(binary_path)
    if not sklearn_tags:
        print(
            f'Could not find an embedded Cython version tag in {binary_path}. '
            'This check may need updating for a newer Cython release; skipping.'
        )
        return 0

    mismatched = {tag for tag in sklearn_tags if tag[:2] != installed_tuple[:2]}
    if mismatched:
        sklearn_versions = ', '.join('.'.join(map(str, tag)) for tag in sorted(sklearn_tags))
        print(
            'Cython ABI mismatch detected:\n'
            f'  Installed Cython:              {installed}\n'
            f'  scikit-learn compiled against:  {sklearn_versions}\n'
            '\n'
            "empulse's Cython extensions (e.g. CostImpurity, which subclasses scikit-learn's "
            "ClassificationCriterion) inherit C-level attributes directly from scikit-learn's "
            'compiled extensions. A Cython *minor* version mismatch between the two can silently '
            'corrupt memory at runtime (observed as a hard crash / access violation) instead of '
            'raising a Python exception.\n'
            '\n'
            f'Fix: pin `cython=={".".join(map(str, max(sklearn_tags)))}` (or the closest '
            'available release with the same major.minor) in pyproject.toml / your lockfile, '
            'then rebuild with `just compile`.'
        )
        return 1

    print(f"Cython ABI check passed: installed Cython {installed} matches scikit-learn's build.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
