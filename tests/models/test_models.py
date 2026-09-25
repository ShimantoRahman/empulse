import inspect

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.estimator_checks import _get_check_estimator_ids, estimator_checks_generator
from xgboost import XGBClassifier

from empulse.datasets import fetch_give_me_some_credit
from empulse.metrics import cost_loss, mpc_score
from empulse.models import (
    CSBoostClassifier,
    CSForestClassifier,
    CSLogitClassifier,
    CSThresholdClassifier,
    CSTreeClassifier,
    ProfLogitClassifier,
    ProfMEMPMClassifier,
    ProfMPMClassifier,
    ProfSRClassifier,
    ProfTreeClassifier,
    RobustCSClassifier,
)
from empulse.optimizers import LBFGSBOptimizer

from .._estimator_common import iter_invalid_params
from .estimator_inventory import estimator_id, make_estimators

# The single source of truth for "every estimator, cheaply configured". Shared with
# `test_sklearn_integration.py` so the list cannot drift between the two conformance suites.
ESTIMATORS = tuple(make_estimators())


ESTIMATOR_CLASSES = {est.__class__ for est in ESTIMATORS}


def expected_failed_checks(estimator):
    if isinstance(estimator, CSThresholdClassifier):
        return {'check_decision_proba_consistency': 'CalibratedClassifierCV does not support decision_function.'}
    if isinstance(
        estimator,
        CSTreeClassifier
        | CSForestClassifier
        | CSLogitClassifier
        | RobustCSClassifier
        | ProfTreeClassifier
        | ProfLogitClassifier
        | ProfSRClassifier
        | ProfMPMClassifier
        | ProfMEMPMClassifier,
    ):
        return {
            'check_classifiers_one_label_sample_weights': 'Sklearn assumes that the estimator accepts sample weights.'
        }
    return {}


def _conformance_checks():
    for estimator in ESTIMATORS:
        yield from estimator_checks_generator(
            estimator, expected_failed_checks=expected_failed_checks(estimator), mark='xfail'
        )


def _conformance_check_id(value):
    # `parametrize_with_checks` names a case by the estimator's repr, which for the models taking an
    # optimizer includes that optimizer's memory address. The ids then differ between processes, and
    # pytest-xdist refuses to run a suite whose workers collected different tests.
    if isinstance(value, BaseEstimator):
        return estimator_id(value)
    return _get_check_estimator_ids(value)


# `parametrize_with_checks`, with ids that are the same in every process.
@pytest.mark.parametrize(('estimator', 'check'), _conformance_checks(), ids=_conformance_check_id)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)


@pytest.fixture(scope='module')
def give_me_some_credit():
    dataset = fetch_give_me_some_credit(backend=pd)
    X = dataset.data
    y = dataset.target
    fp_cost = dataset.instance_costs['fp_cost']
    fn_cost = dataset.instance_costs['cl'] * 0.75  # loss_given_default
    tp_cost = 0.0
    tn_cost = 0.0
    return X, y, tp_cost, fp_cost, tn_cost, fn_cost


@pytest.mark.slow
@pytest.mark.parametrize(
    'classifier',
    [
        CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid', random_state=42),
        CSBoostClassifier(XGBClassifier(n_estimators=30)),
        CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=10)),
        CSTreeClassifier(max_depth=2),
        CSForestClassifier(n_estimators=3, max_depth=1),
        RobustCSClassifier(estimator=CSBoostClassifier(XGBClassifier(n_estimators=30))),
    ],
    ids=estimator_id,
)
def test_cost_loss_performance(classifier, give_me_some_credit):
    X, y, tp_cost, fp_cost, tn_cost, fn_cost = give_me_some_credit

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', classifier)])

    if isinstance(classifier, CSThresholdClassifier):
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)
    else:
        pipeline.fit(
            X, y, model__tp_cost=tp_cost, model__fp_cost=fp_cost, model__tn_cost=tn_cost, model__fn_cost=fn_cost
        )
        y_pred = pipeline.predict(X)

    performance = cost_loss(
        y, y_pred, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, normalize=True
    )

    assert performance < 750, f'Performance {performance} is not better than 750'


def _invalid_param_cases():
    """One case per (estimator class, constructor parameter), so the id names the parameter."""
    for estimator_class in sorted(ESTIMATOR_CLASSES, key=lambda c: c.__name__):
        for name, invalid in iter_invalid_params(estimator_class):
            yield pytest.param(estimator_class, name, invalid, id=f'{estimator_class.__name__}-{name}')


@pytest.mark.parametrize(('estimator_class', 'param_name', 'invalid_params'), _invalid_param_cases())
def test_invalid_params(estimator_class, param_name, invalid_params, cost_dataset):
    """Every constructor parameter must be rejected by scikit-learn's parameter validation."""
    X, y, _, _ = cost_dataset
    parameters = inspect.signature(estimator_class.__init__).parameters
    # Supply a valid value for the parameters the estimator cannot be constructed without, unless
    # that parameter is itself the one under test.
    defaults = {}
    if 'estimator' in parameters and param_name != 'estimator':
        defaults['estimator'] = LogisticRegression()
    if 'loss' in parameters and param_name != 'loss' and estimator_class in {ProfLogitClassifier, ProfTreeClassifier}:
        defaults['loss'] = mpc_score

    model = estimator_class(**defaults, **invalid_params)
    with pytest.raises(InvalidParameterError):
        model.fit(X, y)
