"""The documentation homepage's code is executed, like every other code block in ``docs/``.

``tests/test_user_guide.py`` runs every ``.. code-block:: python`` in the prose pages. The homepage
is rendered from a template rather than from reStructuredText, so its snippets live in
``docs/homepage_content.py`` instead and are covered here: they run in order against one shared
namespace, the way a reader would type them, and each one's result is checked against the value the
page prints beside it.

Checking the results matters because a snippet can still run while no longer producing what the
front page says it produces.
"""

import sys
from pathlib import Path

import pytest
from sklearn import set_config

DOCS = Path(__file__).resolve().parents[1] / 'docs'
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

import homepage_content  # ruff: ignore[module-import-not-at-top-of-file]

STEPS = homepage_content.STEPS


@pytest.fixture(scope='module')
def executed():
    """Run the setup and every step in order, returning the namespace and the value each step shows."""
    set_config(enable_metadata_routing=False)  # reset the global configuration
    namespace: dict = {}
    exec(homepage_content.SETUP, namespace)

    collected = []
    for step in STEPS:
        exec(step.code, namespace)
        collected.append(eval(step.result_of, namespace))
    return namespace, collected


@pytest.fixture(scope='module')
def results(executed):
    """The value each step's page shows, in order."""
    return executed[1]


@pytest.mark.slow
@pytest.mark.parametrize('index', range(len(STEPS)), ids=[step.label for step in STEPS])
def test_step_produces_what_the_homepage_shows(results, index):
    """Each snippet produces the value printed beside it on the homepage."""
    step = STEPS[index]
    result = results[index]

    try:
        expected = float(step.result)
    except ValueError:
        assert repr(result) == step.result, f'step {step.label!r} no longer produces what the homepage shows'
        return

    # Numbers are compared at the precision the page prints them at: a change too small to alter
    # the printed figure is not a failure, and one large enough to alter it is.
    decimals = len(step.result.partition('.')[2])
    assert round(float(result), decimals) == expected, (
        f'step {step.label!r} produces {result!r}, but the homepage shows {step.result}'
    )


def test_result_expressions_are_expressions():
    """``result_of`` is evaluated, so it has to be an expression rather than a statement."""
    for step in STEPS:
        compile(step.result_of, f'<{step.label}>', 'eval')


@pytest.mark.slow
def test_pipeline_snippet_runs(executed):
    """The scikit-learn card's snippet runs where the walkthrough leaves off and cross-validates."""
    namespace = dict(executed[0])
    *statements, last = homepage_content.PIPELINE_SNIPPET.strip().splitlines()
    exec('\n'.join(statements), namespace)
    scores = eval(last, namespace)
    assert len(scores) == 5


@pytest.mark.slow
def test_metric_results_match_the_walkthrough(executed):
    """Each result tile on the metrics card prints what the walkthrough's code produces."""
    namespace = executed[0]
    assert len(namespace['y_test']) == homepage_content.TOUR_TEST_CUSTOMERS

    for result in homepage_content.METRIC_RESULTS:
        value = float(eval(result.value_of, namespace))
        if result.value.startswith('€'):
            printed = f'€{round(value):,}'
        else:
            decimals = len(result.value.partition('.')[2])
            printed = f'{value:.{decimals}f}'
        assert printed == result.value, f'tile {result.label!r} shows {result.value}, but the code gives {printed}'


def test_cost_matrix_snippet_runs(executed):
    """The metrics card's snippet builds the cost matrix the walkthrough's first step defines."""
    namespace = dict(executed[0])
    matrix = eval(homepage_content.COST_MATRIX_SNIPPET.strip(), namespace)
    assert repr(matrix) == STEPS[0].result
