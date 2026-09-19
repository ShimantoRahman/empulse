"""The documentation homepage's code is executed, like every other code block in ``docs/``.

``tests/test_user_guide.py`` runs every ``.. code-block:: python`` in the prose pages. The homepage
is rendered from a template rather than from reStructuredText, so its snippets live in
``docs/homepage_content.py`` instead and are covered here: they run in order against one shared
namespace, the way a reader would type them, and each one's result is checked against the value the
page prints beside it.

That last part is the point. A snippet that still runs but no longer produces what the front page
says it produces is exactly the kind of drift a landing page invites.
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
def results():
    """Run the setup and every step in order, collecting the value each one's page shows."""
    set_config(enable_metadata_routing=False)  # reset the global configuration
    namespace: dict = {}
    exec(homepage_content.SETUP, namespace)

    collected = []
    for step in STEPS:
        exec(step.code, namespace)
        collected.append(eval(step.result_of, namespace))
    return collected


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
