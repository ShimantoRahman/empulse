import importlib
import inspect
import pkgutil
import re
import textwrap

import pytest

# Top-level module to start from
TOP_MODULE = 'empulse'

# Regular expression to find code blocks
CODE_BLOCK_RE = re.compile(r'\.\. code-block:: python\n\s*([\s\S]*?)(?=\n\S|$)')


def extract_code_blocks(docstring):
    """Extract code blocks from a docstring."""
    return ['    ' + match.group(1) for match in CODE_BLOCK_RE.finditer(docstring)]


def get_all_functions_and_classes(module):
    """Get all functions and classes from a module."""
    functions_and_classes = []
    for _, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == module.__name__:
            functions_and_classes.append(obj)
    return functions_and_classes


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


@pytest.mark.slow
@pytest.mark.parametrize('module', iter_modules(TOP_MODULE), ids=lambda m: m.__name__)
def test_code_blocks_in_docstrings(module):
    """Test that code blocks in docstrings execute without errors."""
    functions_and_classes = get_all_functions_and_classes(module)

    for obj in functions_and_classes:
        docstring = inspect.getdoc(obj)
        if docstring:
            code_blocks = extract_code_blocks(docstring)
            for code in code_blocks:
                # Remove common leading indentation
                code = textwrap.dedent(code)
                # Execute the code block
                exec_globals = {}
                try:
                    exec(code, exec_globals)
                except Exception as e:  # noqa: BLE001
                    pytest.fail(f'Code block in {obj.__name__} docstring failed to execute: {e}')
