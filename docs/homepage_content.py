"""The copy the documentation homepage is built from.

The homepage is rendered by ``_templates/homepage.html`` rather than from reStructuredText, so its
content lives here instead of in ``index.rst``. Keeping it in one module rather than inline in the
template has two reasons:

* the code snippets are executed by ``tests/test_homepage_snippets.py``, the way every
  ``.. code-block:: python`` in ``docs/`` is executed by ``tests/test_user_guide.py``, so a snippet
  on the front page cannot go stale without a test turning red;
* ``sphinxext/homepage.py`` highlights the snippets with Pygments at build time, which needs them
  as plain strings.

Every ``:ref:`` target named here has to exist; ``sphinxext/homepage.py`` resolves them through
Sphinx's own reference machinery, so a renamed label fails the ``-W`` build rather than shipping a
dead link.
"""

from __future__ import annotations

from typing import Final, NamedTuple

# --- Hero ----------------------------------------------------------------------------------

EYEBROW: Final[str] = 'Cost-sensitive machine learning for scikit-learn'

MOTTO: Final[str] = 'Not every mistake costs the same.'

LEAD: Final[str] = (
    'Accuracy, F1 and AUC price a false positive and a false negative the same. Your business '
    'does not. Empulse lets you write down what each outcome is actually worth, then '
    '<strong>evaluate</strong>, <strong>train</strong> and <strong>threshold</strong> your models '
    'as scikit-learn estimators.'
)

INSTALL_COMMAND: Final[str] = 'pip install empulse'

# --- The three selling points --------------------------------------------------------------


class Pillar(NamedTuple):
    """One of the three claims the homepage makes, with the guide page that backs it up."""

    kicker: str
    title: str
    body: str
    link_text: str
    link_ref: str


PILLARS: Final[tuple[Pillar, ...]] = (
    Pillar(
        kicker='Measure',
        title='Score models by value',
        body=(
            'Evaluate expected cost, savings, or profit. Use prebuilt metrics for churn, customer '
            'acquisition, and credit scoring, or define a custom cost matrix.'
        ),
        link_text='Measuring a model',
        link_ref='measuring',
    ),
    Pillar(
        kicker='Train',
        title='Optimise the cost matrix directly',
        body=(
            'Train logistic regression, gradient boosting, and trees to minimise cost instead of '
            'log-loss. Costs can vary per customer.'
        ),
        link_text='Training a model',
        link_ref='training',
    ),
    Pillar(
        kicker='Decide',
        title='Tune decision thresholds',
        body=(
            'Find the probability threshold or population fraction that minimizes total cost, then '
            'wrap any classifier to apply it.'
        ),
        link_text='Making a decision',
        link_ref='deciding',
    ),
)

# --- The code walkthrough --------------------------------------------------------------------

# Not displayed. It gives the walkthrough the names an ordinary scikit-learn script would already
# have bound, so each step below can stay short enough to read in one glance. The caption under
# the code panel tells the reader this much, so nothing on the page appears out of thin air.
SETUP: Final[str] = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from empulse.datasets import load_churn_tv_subscriptions

dataset = load_churn_tv_subscriptions(backend=pd)
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=42
)
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
"""

SETUP_CAPTION: Final[str] = (
    'Picking up after an ordinary <code>train_test_split</code>, with <code>baseline</code> '
    'a plain scikit-learn <code>LogisticRegression</code> on a TV-subscription churn dataset.'
)


class Step(NamedTuple):
    """One tab of the code walkthrough: a snippet, and what it produces.

    ``result`` is what the page prints in the result panel. ``result_of`` is the expression it is
    the value of, evaluated once the snippet has run; ``tests/test_homepage_snippets.py`` checks
    the two against each other, so the number beside a snippet cannot drift away from it.
    """

    label: str
    title: str
    blurb: str
    code: str
    result_of: str
    result: str
    result_note: str
    link_text: str
    link_ref: str


STEPS: Final[tuple[Step, ...]] = (
    Step(
        label='Define',
        title='Define the cost matrix',
        blurb='Assign costs to false positives and false negatives using named variables or fixed amounts.',
        code="""from empulse.metrics import CostMatrix

# Assign costs using business variables.
cost_matrix = (
    CostMatrix()
    .add_fp_cost('incentive')  # a wasted discount
    .add_fn_cost('clv')        # a lost customer
    .set_default(incentive=10, clv=200)
)
""",
        result_of='cost_matrix',
        result='CostMatrix(tp_cost=0, tn_cost=0, fp_cost=incentive, fn_cost=clv)',
        result_note='Named symbols allow costs to vary per customer.',
        link_text='Defining costs',
        link_ref='defining_costs',
    ),
    Step(
        label='Measure',
        title='Turn it into a metric',
        blurb='A Metric wraps the cost matrix into a scikit-learn scoring function for cross-validation and grid search.',
        code="""from empulse.metrics import Cost, Metric

expected_cost = Metric(cost_matrix, Cost())

y_score = baseline.predict_proba(X_test)[:, 1]
expected_cost(y_test, y_score)
""",
        result_of='expected_cost(y_test, y_score)',
        result='10.22',
        result_note='Expected cost in euros per customer for the baseline model.',
        link_text='Measuring a model',
        link_ref='measuring',
    ),
    Step(
        label='Train',
        title='Train on the cost metric',
        blurb='Pass the metric directly as the training loss function.',
        code="""from empulse.models import CSLogitClassifier

model = CSLogitClassifier(loss=expected_cost).fit(X_train, y_train)

expected_cost(y_test, model.predict_proba(X_test)[:, 1])
""",
        result_of='expected_cost(y_test, model.predict_proba(X_test)[:, 1])',
        result='9.37',
        result_note='Expected cost drops to 9.37 euros per customer, an 8% reduction over the baseline on the same test data.',
        link_text='Training a model',
        link_ref='training',
    ),
    Step(
        label='Decide',
        title='Find the optimal threshold',
        blurb='Find the decision threshold that minimizes expected cost for the trained model.',
        code="""from empulse.models import CSThresholdClassifier

decider = CSThresholdClassifier(estimator=model, loss=expected_cost)
decider.fit(X_train, y_train)

decider.threshold_
""",
        result_of='decider.threshold_',
        result='0.048',
        result_note='The cost-optimal decision threshold is 0.048 instead of the default 0.5.',
        link_text='Making a decision',
        link_ref='deciding',
    ),
)

# --- The closing row -------------------------------------------------------------------------


class Destination(NamedTuple):
    """One of the four places a reader can go next from the bottom of the homepage.

    ``icon`` is a Font Awesome class; the theme already loads Font Awesome for its icon links.
    """

    icon: str
    title: str
    body: str
    ref: str


DESTINATIONS: Final[tuple[Destination, ...]] = (
    Destination(
        icon='fa-solid fa-rocket',
        title='Getting Started',
        body='Install Empulse and get a working cost-sensitive model in five minutes.',
        ref='getting_started',
    ),
    Destination(
        icon='fa-solid fa-book-open',
        title='Tutorial',
        body='A step-by-step churn example comparing baseline models and cost-sensitive pipelines.',
        ref='tutorial',
    ),
    Destination(
        icon='fa-solid fa-screwdriver-wrench',
        title='User Guide',
        body='Detailed guides for models, metrics, samplers, and datasets.',
        ref='guide',
    ),
    Destination(
        icon='fa-solid fa-code',
        title='API Reference',
        body='The full class and function reference, with every parameter documented.',
        ref='api',
    ),
)
