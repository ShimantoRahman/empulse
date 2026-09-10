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


# Matches the start of a Python code block at any indentation, so blocks nested inside a directive
# (a ``tab-item``, for instance) are found too.
CODE_BLOCK_START_RE = re.compile(r'^(?P<indent>[ \t]*)\.\. code-block:: python\s*$')

# Directive options such as ``:caption:`` or ``:linenos:`` sit between the directive and its body.
DIRECTIVE_OPTION_RE = re.compile(r'^[ \t]*:[\w-]+:.*$')


def extract_code_blocks(file_content):
    """Extract Python code blocks, including any nested inside another directive.

    A block's body is every following line indented further than the directive itself, so the end
    of a block is found by indentation rather than by looking for the next line in column zero. The
    naive version of this stopped at the first unindented line, which silently swallowed the prose
    after a nested block and then failed to compile it.
    """
    lines = file_content.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        match = CODE_BLOCK_START_RE.match(lines[index])
        if match is None:
            index += 1
            continue

        indent = len(match.group('indent').expandtabs(8))
        index += 1

        # Skip the directive's own options and the blank line separating them from the body.
        while index < len(lines) and (DIRECTIVE_OPTION_RE.match(lines[index]) or not lines[index].strip()):
            index += 1

        body = []
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                body.append('')  # a blank line does not end the block
                index += 1
                continue
            if len(line.expandtabs(8)) - len(line.expandtabs(8).lstrip()) <= indent:
                break
            body.append(line)
            index += 1

        if body:
            blocks.append(textwrap.dedent('\n'.join(body).rstrip()))
    return blocks


def execute_code_blocks(code_blocks):
    """Execute each code block and report any errors."""
    set_config(enable_metadata_routing=False)  # reset the global configuration
    exec_globals = {}  # shared environment for all code blocks
    for code in code_blocks:
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
