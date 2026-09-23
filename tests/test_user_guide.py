import os
import traceback

import pytest
from sklearn import set_config

from tests._docs_common import extract_code_blocks

# Directory containing the documentation. All prose pages are walked, not just docs/guide,
# so the code blocks in the getting-started and tutorial pages are executed too.
DOCS_DIR = 'docs'

# Adjust DOCS_DIR if the current working directory is "tests/"
if os.getcwd().endswith('tests'):
    DOCS_DIR = '../' + DOCS_DIR

# Build output and generated API stubs are not hand-written prose and contain no examples worth
# executing (the stubs' content comes from docstrings, which tests/test_docstring.py already covers).
EXCLUDED_DIRS = {'_build', '_static', '_templates', 'sphinxext', 'generated'}


# Pages whose examples download datasets too large to fetch on every run (hundreds of megabytes for
# the fraud datasets). They are deselected with the other `remote` tests.
REMOTE_PAGES = frozenset({
    'cell2cell.rst',
    'credit_card_fraud.rst',
    'default_credit_card_clients.rst',
    'home_equity.rst',
    'ieee_fraud_detection.rst',
    'kdd98.rst',
    'kddcup09_churn.rst',
    'south_german_credit.rst',
    'telco_customer_churn.rst',
})


def _iter_doc_files():
    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith('.rst'):
                yield os.path.join(root, file)


def _doc_file_params():
    for file_path in sorted(_iter_doc_files()):
        remote = os.path.basename(os.path.dirname(file_path)) == 'datasets' and (
            os.path.basename(file_path) in REMOTE_PAGES
        )
        yield pytest.param(file_path, id=file_path, marks=[pytest.mark.remote] if remote else [])


def execute_code_blocks(code_blocks):
    """Execute each code block and report any errors."""
    set_config(enable_metadata_routing=False)  # reset the global configuration
    exec_globals = {}  # shared environment for all code blocks
    for code in code_blocks:
        try:
            exec(code, exec_globals)
        except Exception as e:  # ruff: ignore[blind-except]
            pytest.fail(f'Error executing code block:\n{code}\nError: {e}\n{traceback.format_exc()}')


@pytest.mark.slow
@pytest.mark.parametrize('file_path', _doc_file_params())
def test_code_blocks_in_user_guides(file_path):
    """Test that code blocks in user guide files execute without errors."""
    with open(file_path, encoding='utf-8') as f:
        content = f.read()
    code_blocks = extract_code_blocks(content)
    execute_code_blocks(code_blocks)
