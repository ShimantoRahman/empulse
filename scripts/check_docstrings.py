"""Report numpydoc validation failures across the public API.

Walks ``__all__`` of the public subpackages plus every class member whose defining
class lives in ``empulse`` (so scikit-learn's inherited methods are excluded), runs
:func:`numpydoc.validate.validate` on each, and prints the failure count per check
code together with the set of codes that currently pass everywhere.

Run it after each remediation phase; the shrinking counts are the progress metric,
and the zero-failure set is what :data:`numpydoc_validation_checks` in ``docs/conf.py``
can safely enable.
"""

import collections
import functools
import importlib
import inspect
import sys
import types

from numpydoc.validate import ERROR_MSGS, validate

_DOC_CARRIERS = (property, staticmethod, classmethod, functools.cached_property, types.FunctionType)

PACKAGES = (
    'empulse.metrics',
    'empulse.models',
    'empulse.samplers',
    'empulse.optimizers',
    'empulse.datasets',
)


def public_names() -> 'collections.abc.Iterator[str]':
    """Yield the fully qualified name of every public object worth validating."""
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, '__all__', []):
            obj = getattr(module, name)
            yield f'{package}.{name}'
            if not inspect.isclass(obj):
                continue
            for attr in dir(obj):
                if attr.startswith('_') and attr != '__call__':
                    continue
                owner = next((klass for klass in obj.__mro__ if attr in vars(klass)), None)
                if owner is None or not owner.__module__.startswith('empulse'):
                    continue
                # only methods / properties carry docstrings worth validating; a plain data
                # attribute (ClassVar list, enum member, ...) would validate its *value's*
                # builtin docstring, which Sphinx autodoc never renders
                if not isinstance(vars(owner)[attr], _DOC_CARRIERS):
                    continue
                yield f'{package}.{name}.{attr}'


def main() -> int:
    """Print the per-code failure counts and the zero-failure set."""
    counts: collections.Counter[str] = collections.Counter()
    checked = 0
    skipped = []
    for fq_name in public_names():
        try:
            result = validate(fq_name)
        except Exception:  # noqa: BLE001 - properties and C extensions cannot be introspected
            skipped.append(fq_name)
            continue
        checked += 1
        counts.update(code for code, _ in result['errors'])

    print(f'objects validated: {checked}   (skipped/unvalidatable: {len(skipped)})')
    for code, count in counts.most_common():
        print(f'  {code}: {count:>4}  {ERROR_MSGS[code][:72]}')

    zero = sorted(code for code in ERROR_MSGS if counts[code] == 0)
    print(f'\nZERO ({len(zero)}/{len(ERROR_MSGS)}): {" ".join(zero)}')

    if '-v' in sys.argv:
        print('\nskipped:')
        for name in skipped:
            print(f'  {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
