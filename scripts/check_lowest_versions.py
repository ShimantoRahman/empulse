"""
Fail unless every dependency is installed at exactly the minimum version ``pyproject.toml`` declares.

Guards the ``py311-lowest`` tox environment, which installs the lowest installable version of each
runtime dependency and of the ``optional`` extra, then runs the test suite against them. That only
proves the declared minimums work if the declared minimums are what got installed, and two things
can quietly install something newer:

* **Another dependency requires more.** ``imbalanced-learn>=0.14.2`` needs ``numpy>=1.25.2``, so a
  declared ``numpy>=1.24.4`` can never be installed alongside it. A user reading ``pyproject.toml``
  would believe 1.24.4 is supported, yet no environment could ever have tested it.
* **The minimum has no wheel for this interpreter**, and the environment only installs wheels.

Either way, the declared minimum is a promise nothing checks, so the fix is to raise it to the
version reported here -- the lowest one that is actually installable, and the one the suite ran
against.

Usage:
    python scripts/check_lowest_versions.py [pyproject.toml]
"""

from __future__ import annotations

import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

# The extras the environment installs; `optional` itself only names the others.
EXTRAS = ('boosting', 'symbolic')


def declared_minimums(pyproject: Path) -> dict[str, Version]:
    """The ``>=`` or ``==`` bound of every requirement that applies to this interpreter."""
    project = tomllib.loads(pyproject.read_text(encoding='utf-8'))['project']
    requirements = [Requirement(spec) for spec in project['dependencies']]
    for extra in EXTRAS:
        requirements += [Requirement(spec) for spec in project['optional-dependencies'][extra]]

    minimums = {}
    for requirement in requirements:
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        bounds = [Version(spec.version) for spec in requirement.specifier if spec.operator in {'>=', '=='}]
        if bounds:
            minimums[canonicalize_name(requirement.name)] = max(bounds)
    return minimums


def main() -> int:
    pyproject = Path(sys.argv[1] if len(sys.argv) > 1 else 'pyproject.toml')
    problems = []
    for name, minimum in sorted(declared_minimums(pyproject).items()):
        try:
            installed = Version(version(name))
        except PackageNotFoundError:
            problems.append(f'{name}: declares >={minimum}, but it is not installed')
            continue
        if installed != minimum:
            problems.append(f'{name}: declares >={minimum}, but the lowest installable version is {installed}')
        else:
            print(f'{name} {installed}')

    if problems:
        print(
            'These declared minimums cannot be installed on this interpreter, so the suite did not run '
            'against them. Raise each to the version shown:\n  ' + '\n  '.join(problems),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
