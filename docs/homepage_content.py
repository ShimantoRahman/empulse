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

Headings mark their one emphasised word with asterisks (``*value*``). The extension turns that
into the landing page's emphasis style, so the template never has to know which word it is.
"""

from __future__ import annotations

from typing import Final, NamedTuple

# --- Hero ----------------------------------------------------------------------------------

EYEBROW: Final[str] = 'Cost-sensitive machine learning for scikit-learn'

# `|` marks where the motto breaks on a wide screen; on a phone it flows as one sentence.
MOTTO: Final[str] = 'Optimize|for *value*,|not stats.'

LEAD: Final[str] = (
    'Accuracy, F1 and AUC price every mistake the same. Your business does not. Write down what '
    'each outcome costs, then <strong>evaluate</strong>, <strong>train</strong> and '
    '<strong>threshold</strong> your models for profit, with the scikit-learn API you already know.'
)

INSTALL_COMMAND: Final[str] = 'pip install empulse'

# The signature beside the motto. Its numbers come from `docs/_static/data/homepage_hero.json`,
# which `scripts/figures/homepage_data.py` computes with Empulse; only the words live here.
SIGNATURE_KICKER: Final[str] = 'Churn retention'
SIGNATURE_TITLE: Final[str] = 'Same data, same features. More profit.'
SIGNATURE_FOOTNOTE: Final[str] = (
    "Profit is the retention cost saved compared with contacting nobody, using each customer's own "
    'costs. Thresholds are chosen on the training split.'
)

# --- The code walkthrough --------------------------------------------------------------------

TOUR_TITLE: Final[str] = 'A minute with *empulse*.'
TOUR_LEAD: Final[str] = 'Four scikit-learn steps from a business question to a decision that pays.'

# Not displayed. It gives the walkthrough the names an ordinary scikit-learn script would already
# have bound, so each step below can stay short enough to read in one glance. The caption under
# the code panel tells the reader this much, so nothing on the page appears out of thin air.
SETUP: Final[str] = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
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
        blurb='Pass the metric directly as the training loss, so the model minimizes the cost you defined.',
        code="""from empulse.models import CSLogitClassifier

model = CSLogitClassifier(loss=expected_cost).fit(X_train, y_train)

expected_cost(y_test, model.predict_proba(X_test)[:, 1])
""",
        result_of='expected_cost(y_test, model.predict_proba(X_test)[:, 1])',
        result='9.37',
        result_note='Expected cost drops to 9.37 euros per customer, an 8% reduction over the baseline.',
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

# --- Everything you need ------------------------------------------------------------------------

FEATURES_TITLE: Final[str] = 'Put a *price* on every outcome.'
FEATURES_LEAD: Final[str] = (
    'One cost matrix carries through evaluation, training and the final decision, so the model you '
    'ship is judged by the number your business reports.'
)


class MetricResult(NamedTuple):
    """One result tile on the metrics card: a number the quick tour's code produces.

    ``value`` is what the tile prints and ``value_of`` the expression it is the value of, evaluated
    once the walkthrough has run; ``tests/test_homepage_snippets.py`` checks the two against each
    other, so a tile cannot drift away from the code. Costs are totals over the test customers
    rather than per customer, so they read on the same scale as the €200 in the matrix above them
    and the hero's profit figures. ``sklearn_target`` names the model's intersphinx target when it
    is a scikit-learn class rather than an Empulse one.
    """

    label: str
    model: str
    value: str
    value_of: str
    sklearn_target: str | None = None


# The walkthrough's `train_test_split` leaves this many test customers; the test checks it.
TOUR_TEST_CUSTOMERS: Final[int] = 2814

METRIC_RESULTS_CAPTION: Final[str] = (
    f"The same matrix, put to work on the quick tour's {TOUR_TEST_CUSTOMERS:,} test customers:"
)
METRIC_RESULTS: Final[tuple[MetricResult, ...]] = (
    MetricResult(
        'Total expected cost',
        'LogisticRegression',
        '€28,765',
        'expected_cost(y_test, baseline.predict_proba(X_test)[:, 1]) * len(y_test)',
        sklearn_target='sklearn.linear_model.LogisticRegression',
    ),
    MetricResult(
        'Total expected cost',
        'CSLogitClassifier',
        '€26,379',
        'expected_cost(y_test, model.predict_proba(X_test)[:, 1]) * len(y_test)',
    ),
    MetricResult('Optimal threshold', 'CSThresholdClassifier', '0.048', 'decider.threshold_'),
)

# The metrics card's snippet: the matrix drawn above it, written down. It is the walkthrough's
# first step without the defaults, and runs after that step, where `CostMatrix` is imported.
COST_MATRIX_SNIPPET: Final[str] = """CostMatrix().add_fp_cost('incentive').add_fn_cost('clv')
"""

# The estimators named on the training card. The count beside them is read from
# `empulse.models.__all__` at build time, so adding a model never leaves the number behind.
ESTIMATOR_HIGHLIGHTS: Final[tuple[str, ...]] = (
    'CSLogitClassifier',
    'CSBoostClassifier',
    'B2BoostClassifier',
    'CSTreeClassifier',
    'CSForestClassifier',
    'ProfTreeClassifier',
    'ProfLogitClassifier',
)

# The scikit-learn card's snippet. It runs after the walkthrough's steps, in the same namespace, so
# `expected_cost`, `X_train` and `y_train` are already bound and `cross_val_score` is imported by
# `SETUP`. The scikit-learn names in the card's text link into scikit-learn's own documentation and
# are examples, so the sentence around them says so.
SKLEARN_LEAD: Final[str] = (
    "Estimators, metrics and samplers follow scikit-learn's conventions, so they slot into the tools you "
    'already use:'
)
SKLEARN_TAIL: Final[str] = 'and the rest of the ecosystem.'
SKLEARN_TOOLS: Final[tuple[tuple[str, str, str], ...]] = (
    ('Pipeline', 'py:class', 'sklearn.pipeline.Pipeline'),
    ('GridSearchCV', 'py:class', 'sklearn.model_selection.GridSearchCV'),
    ('cross_val_score', 'py:function', 'sklearn.model_selection.cross_val_score'),
)

PIPELINE_SNIPPET: Final[str] = """pipe = make_pipeline(
    StandardScaler(),
    CSLogitClassifier(loss=expected_cost),
)
cross_val_score(pipe, X_train, y_train)
"""


class DatasetHighlight(NamedTuple):
    """One row of the datasets card: a few loaders, not the full list, which grows each release."""

    name: str
    problem: str
    loader: str
    source: str
    ref: str


DATASET_HIGHLIGHTS: Final[tuple[DatasetHighlight, ...]] = (
    DatasetHighlight('TV subscriptions', 'Churn', 'load_churn_tv_subscriptions', 'Bundled', 'churn_tv_subscriptions'),
    DatasetHighlight(
        'Bank telemarketing', 'Upsell', 'load_upsell_bank_telemarketing', 'Bundled', 'upsell_bank_telemarketing'
    ),
    DatasetHighlight('PAKDD', 'Credit scoring', 'load_credit_scoring_pakdd', 'Bundled', 'credit_scoring_pakdd'),
    DatasetHighlight('Credit card fraud', 'Fraud', 'fetch_credit_card_fraud', 'Downloaded', 'credit_card_fraud'),
    DatasetHighlight('KDD Cup 1998', 'Direct mailing', 'fetch_kdd98', 'Downloaded', 'kdd98'),
)

# --- Built on research ---------------------------------------------------------------------------

RESEARCH_TITLE: Final[str] = 'Methods from *peer-reviewed* papers.'
RESEARCH_LEAD: Final[str] = (
    'Every estimator and metric implements a published method, with the reference in its '
    'docstring. Empulse puts them behind one consistent API.'
)


class Paper(NamedTuple):
    """A paper behind one of the package's estimators or metrics, as cited in its docstring."""

    year: int
    title: str
    authors: str
    venue: str
    implements: tuple[str, ...]


PAPERS: Final[tuple[Paper, ...]] = (
    Paper(
        2013,
        'A novel profit maximizing metric for measuring classification performance of customer churn '
        'prediction models',
        'Verbraken, Verbeke & Baesens',
        'IEEE Transactions on Knowledge and Data Engineering',
        ('empc_score', 'mpc_score'),
    ),
    Paper(
        2014,
        'Development and application of consumer credit scoring models using profit-based '
        'classification measures',
        'Verbraken, Bravo, Weber & Baesens',
        'European Journal of Operational Research',
        ('empcs_score', 'mpcs_score'),
    ),
    Paper(
        2015,
        'Example-dependent cost-sensitive decision trees',
        'Correa Bahnsen, Aouada & Ottersten',
        'Expert Systems with Applications',
        ('CSTreeClassifier',),
    ),
    Paper(
        2017,
        'Profit maximizing logistic model for customer churn prediction using genetic algorithms',
        'Stripling, vanden Broucke, Antonio, Baesens & Snoeck',
        'Swarm and Evolutionary Computation',
        ('ProfLogitClassifier',),
    ),
    Paper(
        2020,
        'Profit-based churn prediction based on minimax probability machines',
        'Maldonado, López & Vairetti',
        'European Journal of Operational Research',
        ('ProfMPMClassifier',),
    ),
    Paper(
        2022,
        'Instance-dependent cost-sensitive learning for detecting transfer fraud',
        'Höppner, Baesens, Verbeke & Verdonck',
        'European Journal of Operational Research',
        ('CSLogitClassifier',),
    ),
    Paper(
        2022,
        'B2Boost: instance-dependent profit-driven modelling of B2B churn',
        'Janssens, Bogaert, Bagué & Van den Poel',
        'Annals of Operations Research',
        ('B2BoostClassifier', 'empb_score'),
    ),
    Paper(
        2025,
        'Profit-driven pre-processing in B2B customer churn modeling using fairness techniques',
        'Rahman, Janssens & Bogaert',
        'Journal of Business Research',
        ('BiasResamplingClassifier', 'auepc_score'),
    ),
)

CITATION_DOI: Final[str] = '10.5281/zenodo.11185664'

# --- The closing row -------------------------------------------------------------------------

CLOSING_TITLE: Final[str] = 'Start shipping models that *pay*.'
CLOSING_BUTTON: Final[str] = 'Get started in five minutes'


class Destination(NamedTuple):
    """One of the four places a reader can go next from the bottom of the homepage."""

    title: str
    body: str
    ref: str


DESTINATIONS: Final[tuple[Destination, ...]] = (
    Destination(
        title='Getting Started',
        body='Install Empulse and fit a working cost-sensitive model.',
        ref='getting_started',
    ),
    Destination(
        title='Tutorial',
        body='A churn example comparing baseline and cost-sensitive pipelines.',
        ref='tutorial',
    ),
    Destination(
        title='User Guide',
        body='Models, metrics, samplers and datasets in depth.',
        ref='guide',
    ),
    Destination(
        title='API Reference',
        body='Every class and function, with every parameter documented.',
        ref='api',
    ),
)
