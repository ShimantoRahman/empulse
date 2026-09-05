import os
import re
import textwrap
import traceback

import pytest
from sklearn import set_config

# Directory containing the documentation. All prose pages are walked, not just docs/guide,
# so the code blocks in the getting-started and tutorial pages are executed too.
DOCS_DIR = 'docs'

# Adjust DOCS_DIR if the current working directory is "tests/"
if os.getcwd().endswith('tests'):
    DOCS_DIR = '../' + DOCS_DIR

# Build output and generated API stubs are not hand-written prose and contain no examples worth
# executing (the stubs' content comes from docstrings, which tests/test_docstring.py already covers).
EXCLUDED_DIRS = {'_build', '_static', '_templates', 'sphinxext', 'generated'}


def _iter_doc_files():
    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith('.rst'):
                yield os.path.join(root, file)


# Regular expression to find code blocks
CODE_BLOCK_RE = re.compile(r'\.\. code-block:: python\n\s*([\s\S]*?)(?=\n\S|$)')


def extract_code_blocks(file_content):
    """Extract code blocks from the file content."""
    return ['    ' + match.group(1) for match in CODE_BLOCK_RE.finditer(file_content)]


def execute_code_blocks(code_blocks):
    """Execute each code block and report any errors."""
    set_config(enable_metadata_routing=False)  # reset the global configuration
    exec_globals = {}  # shared environment for all code blocks
    for code in code_blocks:
        code = textwrap.dedent(code)
        try:
            exec(code, exec_globals)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f'Error executing code block:\n{code}\nError: {e}\n{traceback.format_exc()}')


@pytest.mark.slow
@pytest.mark.parametrize('file_path', sorted(_iter_doc_files()))
def test_code_blocks_in_user_guides(file_path):
    """Test that code blocks in user guide files execute without errors."""
    with open(file_path, encoding='utf-8') as f:
        content = f.read()
    code_blocks = extract_code_blocks(content)
    execute_code_blocks(code_blocks)
