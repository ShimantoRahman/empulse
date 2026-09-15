"""
Fail unless the installed scikit-learn is exactly the version named on the command line.

Guards the `py3XX-sklearn-pinned` tox environment. A pinned-dependency environment that quietly
installs a different version is worse than one that errors: it reports a pass for a version it never
exercised, which is precisely the signal the scikit-learn release watch exists to produce.

Usage:
    python scripts/assert_sklearn_version.py 1.9.1
"""

from __future__ import annotations

import sys

import sklearn


def main() -> int:
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <expected-version>', file=sys.stderr)
        return 2

    expected, installed = sys.argv[1], sklearn.__version__
    if installed != expected:
        print(
            f'Expected scikit-learn {expected}, but {installed} is installed. The pin in tox.ini did '
            f'not survive the install phase, so this environment would have tested the wrong version.',
            file=sys.stderr,
        )
        return 1

    print(f'scikit-learn {installed} confirmed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
