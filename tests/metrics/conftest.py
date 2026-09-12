"""
Fixtures shared across ``tests/metrics/``.

Each of these previously existed in two or three modules with a byte-identical body. The
``empirical_churn_*`` pair in particular was duplicated verbatim between
``test_auepc_strategy.py`` and ``test_empirical_max_profit_strategy.py``, which are otherwise
near-identical twins.
"""

import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from empulse.metrics import CostMatrix


@pytest.fixture(scope='module')
def y_true_and_prediction():
    """Calibrated probabilities from a logistic regression fit on its own training data."""
    X, y = make_classification(random_state=12)
    lr = LogisticRegression()
    lr.fit(X, y)
    return y, lr.predict_proba(X)[:, 1]


@pytest.fixture
def churn_cost_matrix():
    """The deterministic churn cost matrix, parameterised by ``clv``, ``d``, ``f`` and ``gamma``."""
    clv, d, f, gamma = sympy.symbols('clv d f gamma')
    return CostMatrix().add_tp_benefit(gamma * (clv - d - f)).add_tp_benefit((1 - gamma) * -f).add_fp_cost(d + f)


@pytest.fixture(scope='module')
def empirical_churn_cost_matrix():
    """
    The churn cost matrix used by the empirical strategies, with a ``Beta(6, 14)`` accept rate.

    The contact cost is incurred whenever a churner is contacted, regardless of whether they accept
    the incentive offer, so it is a separate gamma-independent term -- matching the canonical churn
    cost matrix documented on ``CostMatrix`` and used by ``B2BoostClassifier``.

    Returns the matrix together with the symbol table, since callers need the symbols to build the
    parameter dict they pass to the metric.
    """
    gamma = sympy.stats.Beta('gamma', 6, 14)
    delta, f, clv = sympy.symbols('delta f clv')
    return (
        CostMatrix()
        .add_tp_benefit(gamma * ((1 - delta) * clv - f))
        .add_tp_benefit((1 - gamma) * -f)
        .add_fp_cost(delta * clv + f),
        {'delta': delta, 'f': f, 'clv': clv},
    )


@pytest.fixture(scope='module')
def empirical_churn_dataset():
    """A 300-row instance-dependent churn problem: labels, ranking scores and per-customer CLV."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    clv = rng.gamma(2, 100, size=n)
    y_score = rng.normal(size=n) + y * 1.5
    return y, y_score, clv
