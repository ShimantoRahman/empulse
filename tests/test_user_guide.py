import os
import re
import textwrap
import traceback

import pytest
from sklearn import set_config

# Directory containing the user guide files
GUIDE_DIR = 'docs/guide'

# Adjust GUIDE_DIR if the current working directory is "tests/"
if os.getcwd().endswith('tests'):
    GUIDE_DIR = '../' + GUIDE_DIR

# Regular expression to find code blocks
CODE_BLOCK_RE = re.compile(r'\.\. code-block:: python\n\s*([\s\S]*?)(?=\n\S|$)')


def extract_code_blocks(file_content):
    """Extract code blocks from the file content."""
    return ['    ' + match.group(1) for match in CODE_BLOCK_RE.finditer(file_content)]


def execute_code_blocks(code_blocks):
    """Execute each code block and report any errors."""
    set_config(enable_metadata_routing=False)   # reset the global configuration
    for code in code_blocks:
        code = textwrap.dedent(code)
        exec_globals = {}
        try:
            exec(code, exec_globals)
        except Exception as e:
            pytest.fail(f"Error executing code block:\n{code}\nError: {e}\n{traceback.format_exc()}")


@pytest.mark.parametrize("file_path", [
    os.path.join(root, file)
    for root, _, files in os.walk(GUIDE_DIR)
    for file in files if file.endswith('.rst')
])
def test_code_blocks_in_user_guides(file_path):
    """Test that code blocks in user guide files execute without errors."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    code_blocks = extract_code_blocks(content)
    execute_code_blocks(code_blocks)
