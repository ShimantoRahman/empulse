import inspect

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import InvalidParameterError
from xgboost import XGBClassifier

from empulse.datasets import load_give_me_some_credit
from empulse.metrics import cost_loss
from empulse.models import (
    B2BoostClassifier,
    BiasRelabelingClassifier,
    BiasResamplingClassifier,
    BiasReweighingClassifier,
    CSBaggingClassifier,
    CSBoostClassifier,
    CSForestClassifier,
    CSLogitClassifier,
    CSThresholdClassifier,
    CSTreeClassifier,
    ProfLogitClassifier,
    RobustCSClassifier,
)
from empulse.utils._sklearn_compat import parametrize_with_checks

ESTIMATORS = (
    B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1)),
    ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10}),
    BiasReweighingClassifier(estimator=LogisticRegression(max_iter=2)),
    BiasResamplingClassifier(estimator=LogisticRegression(max_iter=2)),
    BiasRelabelingClassifier(estimator=LogisticRegression(max_iter=2)),
    CSBoostClassifier(XGBClassifier(n_estimators=2, max_depth=1), fp_cost=1, fn_cost=1),
    CSLogitClassifier(fp_cost=1, fn_cost=1),
    CSTreeClassifier(max_depth=2, fp_cost=1, fn_cost=1),
    CSForestClassifier(n_estimators=2, max_depth=2, fp_cost=1, fn_cost=1),
    CSBaggingClassifier(
        estimator=CSLogitClassifier(optimizer_params={'max_iter': 2}), n_estimators=2, fp_cost=1, fn_cost=1
    ),
    RobustCSClassifier(estimator=CSLogitClassifier(optimizer_params={'max_iter': 2}), fp_cost=1, fn_cost=1),
    CSThresholdClassifier(estimator=LogisticRegression(max_iter=2), random_state=42, fp_cost=1, fn_cost=1),
)

ESTIMATOR_CLASSES = {est.__class__ for est in ESTIMATORS}


def expected_failed_checks(estimator):
    if isinstance(estimator, ProfLogitClassifier):
        return {
            'check_classifier_data_not_an_array': 'Sklearn does not set random_state properly in the test. '
            'Tested internally.',
            'check_fit_idempotent': 'Sklearn does not set random_state properly in the test. Tested internally.',
            'check_supervised_y_2d': 'Sklearn does not set random_state properly in the test. Tested internally.',
        }
    if isinstance(estimator, CSThresholdClassifier):
        return {'check_decision_proba_consistency': 'CalibratedClassifierCV does not support decision_function.'}
    return {}


@parametrize_with_checks(list(ESTIMATORS), expected_failed_checks=expected_failed_checks)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)


def test_proflogit_classifier_data_not_an_array():
    """Monkey patch the check_classifier_data_not_an_array to set the random state of the estimator."""

    def set_random_state(estimator, random_state=None):
        """Set the random state of an estimator, including optimizer_params if present."""
        if hasattr(estimator, 'optimizer_params'):
            if random_state is None:
                random_state = np.random.RandomState()
            estimator.optimizer_params['random_state'] = random_state

    import sklearn.utils
    from sklearn.utils.estimator_checks import set_random_state as sklearn_set_random_state

    sklearn.utils.set_random_state = set_random_state
    from sklearn.utils.estimator_checks import check_classifier_data_not_an_array

    check_classifier_data_not_an_array(
        ProfLogitClassifier.__name__, ProfLogitClassifier(optimizer_params={'max_iter': 3, 'random_state': 42})
    )
    sklearn.utils.set_random_state = sklearn_set_random_state


def test_proflogit_fit_idempotent():
    """Monkey patch the check_fit_idempotent to set the random state of the estimator."""

    def set_random_state(estimator, random_state=None):
        """Set the random state of an estimator, including optimizer_params if present."""
        if hasattr(estimator, 'optimizer_params'):
            if random_state is None:
                random_state = np.random.RandomState()
            estimator.optimizer_params['random_state'] = random_state

    import sklearn.utils
    from sklearn.utils.estimator_checks import set_random_state as sklearn_set_random_state

    sklearn.utils.set_random_state = set_random_state
    from sklearn.utils.estimator_checks import check_fit_idempotent

    check_fit_idempotent(
        ProfLogitClassifier.__name__, ProfLogitClassifier(optimizer_params={'max_iter': 3, 'random_state': 42})
    )
    sklearn.utils.set_random_state = sklearn_set_random_state


def test_proflogit_supervised_y_2d():
    """Monkey patch the check_supervised_y_2d to set the random state of the estimator."""

    def set_random_state(estimator, random_state=None):
        """Set the random state of an estimator, including optimizer_params if present."""
        if hasattr(estimator, 'optimizer_params'):
            if random_state is None:
                random_state = np.random.RandomState()
            estimator.optimizer_params['random_state'] = random_state

    import sklearn.utils
    from sklearn.utils.estimator_checks import set_random_state as sklearn_set_random_state

    sklearn.utils.set_random_state = set_random_state
    from sklearn.utils.estimator_checks import check_supervised_y_2d

    check_supervised_y_2d(
        ProfLogitClassifier.__name__, ProfLogitClassifier(optimizer_params={'max_iter': 3, 'random_state': 42})
    )
    sklearn.utils.set_random_state = sklearn_set_random_state


@pytest.fixture(scope='module')
def data():
    return load_give_me_some_credit(return_X_y_costs=True, as_frame=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    'classifier',
    [
        CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid', random_state=42),
        CSBoostClassifier(),
        CSLogitClassifier(optimizer_params={'max_iter': 10}),
        CSTreeClassifier(max_depth=2),
        CSForestClassifier(n_estimators=3, max_depth=1),
        RobustCSClassifier(estimator=CSBoostClassifier()),
    ],
)
def test_cost_loss_performance(classifier, data):
    X, y, tp_cost, fp_cost, tn_cost, fn_cost = data

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


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    fn_cost = np.random.rand(y.size)
    fp_cost = 5
    return X, y, fn_cost, fp_cost


class InvalidParameter:
    pass


def generate_invalid_params(estimator_class):
    parameters = inspect.signature(estimator_class.__init__).parameters
    takes_estimator = 'estimator' in parameters
    return [{param: InvalidParameter()} for param in parameters if param != 'self'], takes_estimator


@pytest.mark.parametrize('estimator_class', ESTIMATOR_CLASSES)
def test_invalid_params(estimator_class, dataset):
    X, y, _, _ = dataset
    invalid_params_list, takes_estimator = generate_invalid_params(estimator_class)
    for invalid_params in invalid_params_list:
        if (
            takes_estimator
            and 'estimator' in invalid_params
            and isinstance(invalid_params['estimator'], InvalidParameter)
        ):
            model = estimator_class(**invalid_params)
        elif takes_estimator:
            model = estimator_class(estimator=LogisticRegression(), **invalid_params)
        else:
            model = estimator_class(**invalid_params)
        with pytest.raises(InvalidParameterError):
            model.fit(X, y)
