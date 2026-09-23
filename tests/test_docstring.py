import functools
import importlib
import inspect
import pkgutil
import textwrap
import types

import pytest

from tests._docs_common import extract_code_blocks

# Top-level module to start from
TOP_MODULE = 'empulse'

# Member kinds that can carry their own docstring, as opposed to a plain data attribute
# (a ClassVar list, an enum value, ...) whose ``__doc__`` belongs to its *type*, not to us.
_DOC_CARRIERS = (property, staticmethod, classmethod, functools.cached_property, types.FunctionType)


def get_all_functions_and_classes(module):
    """
    Collect every object in *module* whose docstring examples are worth executing.

    This includes module-level functions and classes (as before), but also:

    * their methods and properties, walked via the MRO so an inherited docstring is only
      collected once, from the class that actually owns it;
    * module-level *instances* carrying their own ``__doc__`` (the prebuilt ``*_score``
      metrics are :class:`~empulse.metrics.Metric` objects with a hand-assigned docstring,
      not classes, so ``inspect.isfunction``/``isclass`` never see them).
    """
    found = []
    seen_ids = set()

    def add(obj):
        if id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))
        found.append(obj)

    for _, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj, '__module__', None) == module.__name__:
            add(obj)
            if inspect.isclass(obj):
                for attr_name in dir(obj):
                    if attr_name.startswith('_') and attr_name != '__call__':
                        continue
                    if attr_name not in vars(obj):
                        continue  # inherited; collected once, when the owning class's own
                        # module is iterated, instead of once per subclass here
                    raw = vars(obj)[attr_name]
                    if not isinstance(raw, _DOC_CARRIERS):
                        continue
                    member = raw.fget if isinstance(raw, property) else getattr(obj, attr_name, raw)
                    if member is not None:
                        add(member)
        elif (
            type(obj).__module__.startswith(TOP_MODULE)
            and hasattr(obj, '__dict__')
            and isinstance(obj.__dict__.get('__doc__'), str)
        ):
            # a module-level instance with its own hand-assigned docstring, e.g. a prebuilt
            # `Metric`/`MixtureMetric` such as `empc_score`. An instance's `__module__`
            # resolves through its *type*, not where the instance itself was created, so
            # (unlike the function/class branch above) module ownership can't be checked this
            # way; the explicit instance-level `__doc__` is itself a strong enough signal.
            # Requiring the *type* to be an empulse class (rather than excluding classes/
            # functions/modules one by one) is what keeps this from also matching a re-exported
            # third-party class (whose own docstring is out of scope) or a `types.GenericAlias`
            # type annotation (e.g. `FloatNDArray = NDArray[np.float64]`), both of which turned
            # up here during testing. `add()`'s id-based dedup keeps a metric re-imported into
            # several modules from being collected (and thus executed) more than once.
            add(obj)

    return found


def _object_label(obj):
    name = getattr(obj, '__name__', None)
    if name is not None:
        return name
    return f'{type(obj).__module__}.{type(obj).__name__} instance'


def iter_modules(module_name):
    """Iterate through all leaf submodules (non-packages) without duplicates."""
    module = importlib.import_module(module_name)
    if not hasattr(module, '__path__'):
        yield module
        return
    # walk_packages already recurses into sub-packages; no need to recurse manually
    for _, submodule_name, ispkg in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
        if not ispkg:
            yield importlib.import_module(submodule_name)


# The examples in these modules download datasets, so they are also deselected with the `remote` tests.
_REMOTE_MODULES = frozenset({'empulse.datasets._remote'})


def _module_params():
    for module in iter_modules(TOP_MODULE):
        marks = [pytest.mark.remote] if module.__name__ in _REMOTE_MODULES else []
        yield pytest.param(module, id=module.__name__, marks=marks)


@pytest.mark.slow
@pytest.mark.parametrize('module', _module_params())
def test_code_blocks_in_docstrings(module):
    """Test that code blocks in docstrings execute without errors."""
    functions_and_classes = get_all_functions_and_classes(module)

    for obj in functions_and_classes:
        docstring = obj.__doc__ if isinstance(obj.__dict__.get('__doc__'), str) else inspect.getdoc(obj)
        if not docstring:
            continue
        code_blocks = extract_code_blocks(docstring)
        for code in code_blocks:
            # Remove common leading indentation
            code = textwrap.dedent(code)
            # Execute the code block
            exec_globals = {}
            try:
                exec(code, exec_globals)
            except Exception as e:  # ruff: ignore[blind-except]
                pytest.fail(f'Code block in {_object_label(obj)} docstring failed to execute: {e}')
