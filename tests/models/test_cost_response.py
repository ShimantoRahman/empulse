"""
Every cost-sensitive model must be steered by the costs it is given.

scikit-learn's estimator checks and the inventory suites fit each model with throwaway settings and
check that what comes back is well formed. None of them can tell a model that optimises its cost
matrix from one that ignores it, or from one that has the sign of its objective flipped. These
tests can: each fits a model under two opposite cost regimes -- a missed positive four times as
expensive as a false alarm, and the reverse -- and asserts that each fit does better on held-out
data under the regime it was trained for. A model that ignores its costs ties with itself and a
model with a flipped objective loses, so both fail.

What "does better" means depends on what the model produces. Most models make decisions, so they
are judged by the cost of their predictions. The profit-driven family trains a ``MaxProfit``
objective: it learns a ranking meant to be cut at its most profitable fraction, and predicting at a
probability of one half is not what it optimises, so it is judged by the maximum profit of its
scores instead.

``ProfMPMClassifier`` is deliberately absent. It forces both classes to share one worst-case accuracy
bound (Maldonado, Lopez & Vairetti, 2020, Section 4.2), so its objective is that bound times the
*sum* of the class weights: no ratio of false-positive to false-negative cost can favour either
class.
"""

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from empulse.metrics import CostMatrix, MaxProfit, Metric, cost_loss, expected_cost_loss_churn
from empulse.models import (
    B2BoostClassifier,
    CSBaggingClassifier,
    CSBoostClassifier,
    CSForestClassifier,
    CSLogitClassifier,
    CSRateClassifier,
    CSThresholdClassifier,
    CSTreeClassifier,
    ProfLogitClassifier,
    ProfMEMPMClassifier,
    ProfSRClassifier,
    ProfTreeClassifier,
    RobustCSClassifier,
)
from empulse.optimizers import GeneticAlgorithmOptimizer

from .estimator_inventory import estimator_id

MISSES_EXPENSIVE = {'fp_cost': 1.0, 'fn_cost': 4.0}
FALSE_ALARMS_EXPENSIVE = {'fp_cost': 4.0, 'fn_cost': 1.0}

# Unlike the inventory, these are configured to learn: a model that has not converged cannot show
# that it responds to its costs.
DECISION_MODELS = [
    CSLogitClassifier(),
    CSBoostClassifier(XGBClassifier(n_estimators=20, max_depth=2)),
    CSTreeClassifier(max_depth=3, random_state=0),
    # With three features, `max_features='sqrt'` would offer each split a single one.
    CSForestClassifier(n_estimators=10, max_depth=3, max_features=None, random_state=0),
    CSBaggingClassifier(CSTreeClassifier(max_depth=3), n_estimators=5, random_state=0),
    RobustCSClassifier(CSLogitClassifier()),
    CSThresholdClassifier(LogisticRegression(), random_state=0),
    CSRateClassifier(LogisticRegression()),
]

RANKING_MODELS = [
    ProfLogitClassifier(optimizer=GeneticAlgorithmOptimizer(max_iter=50, population_size=50, random_state=0)),
    ProfTreeClassifier(max_depth=3, max_iter=50, population_size=50, random_state=0),
    ProfMEMPMClassifier(),
    ProfSRClassifier(generations=5, population_size=100, random_state=0),
]

# The models that train on per-sample costs. The decision-rule models are left out because they
# take per-sample costs at predict time instead, which test_csdecision_rule.py covers, and the
# MaxProfit family because MaxProfit averages per-sample costs away before optimising.
PER_SAMPLE_COST_MODELS = [
    model for model in DECISION_MODELS if not isinstance(model, CSThresholdClassifier | CSRateClassifier)
]

MAX_PROFIT = Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost'), MaxProfit())


@pytest.fixture(scope='module')
def split():
    """
    ``(X_train, X_test, y_train, y_test)`` for a problem where the costs change the best ranking.

    On a generic problem every model ranks almost identically whatever its costs, and the profit
    models could not show a difference. Here the first feature marks a small group of near-certain
    positives but points the wrong way in the rest of the population: a ranking that targets only a
    few samples wants to lean on it, one that targets most samples wants to discount it. The third
    feature is pure noise, which :func:`test_per_sample_costs_steer_the_model` relies on.
    """
    rng = np.random.default_rng(42)
    n_samples = 1200
    sure_positive = rng.random(n_samples) < 0.1
    x1 = np.where(sure_positive, rng.normal(3.0, 0.5, n_samples), rng.normal(0.0, 1.0, n_samples))
    x2 = rng.normal(0.0, 1.0, n_samples)
    noise = rng.normal(0.0, 1.0, n_samples)
    y = (rng.random(n_samples) < np.where(sure_positive, 0.97, expit(2.0 * x2 - x1))).astype(int)
    X = np.column_stack([x1, x2, noise])
    return train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)


@pytest.mark.parametrize('estimator', DECISION_MODELS, ids=estimator_id)
def test_decisions_follow_the_training_costs(estimator, split):
    X_train, X_test, y_train, y_test = split
    for_misses = clone(estimator).fit(X_train, y_train, **MISSES_EXPENSIVE).predict(X_test)
    for_false_alarms = clone(estimator).fit(X_train, y_train, **FALSE_ALARMS_EXPENSIVE).predict(X_test)

    assert for_misses.mean() > for_false_alarms.mean(), 'expensive misses should mean flagging more samples'
    assert cost_loss(y_test, for_misses, **MISSES_EXPENSIVE) < cost_loss(y_test, for_false_alarms, **MISSES_EXPENSIVE)
    assert cost_loss(y_test, for_false_alarms, **FALSE_ALARMS_EXPENSIVE) < cost_loss(
        y_test, for_misses, **FALSE_ALARMS_EXPENSIVE
    )


@pytest.mark.parametrize('estimator', RANKING_MODELS, ids=estimator_id)
def test_ranking_follows_the_training_costs(estimator, split):
    X_train, X_test, y_train, y_test = split
    for_misses = clone(estimator).fit(X_train, y_train, **MISSES_EXPENSIVE).predict_proba(X_test)[:, 1]
    for_false_alarms = clone(estimator).fit(X_train, y_train, **FALSE_ALARMS_EXPENSIVE).predict_proba(X_test)[:, 1]

    assert MAX_PROFIT(y_test, for_misses, **MISSES_EXPENSIVE) > MAX_PROFIT(y_test, for_false_alarms, **MISSES_EXPENSIVE)
    assert MAX_PROFIT(y_test, for_false_alarms, **FALSE_ALARMS_EXPENSIVE) > MAX_PROFIT(
        y_test, for_misses, **FALSE_ALARMS_EXPENSIVE
    )


def test_b2boost_decisions_follow_the_churn_economics(split):
    """B2Boost takes the churn parameters rather than the four costs, so its regimes are churn ones."""
    X_train, X_test, y_train, y_test = split
    # Valuable customers who are cheap to contact are worth targeting; the reverse are not.
    worth_retaining = {'clv': 500.0, 'contact_cost': 1.0, 'incentive_fraction': 0.05, 'accept_rate': 0.3}
    not_worth_retaining = {'clv': 100.0, 'contact_cost': 10.0, 'incentive_fraction': 0.05, 'accept_rate': 0.3}
    estimator = B2BoostClassifier(XGBClassifier(n_estimators=20, max_depth=2))
    for_retaining = clone(estimator).fit(X_train, y_train, **worth_retaining).predict(X_test)
    for_not_retaining = clone(estimator).fit(X_train, y_train, **not_worth_retaining).predict(X_test)

    # Hard 0/1 predictions make the expected cost the cost of the decisions.
    assert for_retaining.mean() > for_not_retaining.mean()
    assert expected_cost_loss_churn(y_test, for_retaining, **worth_retaining) < expected_cost_loss_churn(
        y_test, for_not_retaining, **worth_retaining
    )
    assert expected_cost_loss_churn(y_test, for_not_retaining, **not_worth_retaining) < expected_cost_loss_churn(
        y_test, for_retaining, **not_worth_retaining
    )


@pytest.mark.parametrize('estimator', PER_SAMPLE_COST_MODELS, ids=estimator_id)
def test_per_sample_costs_steer_the_model(estimator, split):
    """
    Misses are expensive only where the noise feature is positive, so that is where to flag.

    The noise feature says nothing about the label, so any difference in the positive rate between
    its two halves can only come from the costs. Shuffling the costs relative to the rows -- what a
    resampling step that loses track of them would do -- leaves no difference, which is why this
    test uses per-sample costs rather than repeating the scalar ones.
    """
    X_train, X_test, y_train, _ = split
    fn_cost = np.where(X_train[:, 2] > 0, 9.0, 1 / 9)
    y_pred = clone(estimator).fit(X_train, y_train, fp_cost=1.0, fn_cost=fn_cost).predict(X_test)

    expensive_misses = X_test[:, 2] > 0
    # At these costs the optimal threshold is 0.1 on one half and 0.9 on the other.
    assert y_pred[expensive_misses].mean() - y_pred[~expensive_misses].mean() > 0.5
