import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from empulse.datasets import load_give_me_some_credit
from empulse.metrics import cost_loss
from empulse.models import (
    B2BoostClassifier,
    ProfLogitClassifier,
    BiasReweighingClassifier,
    BiasResamplingClassifier,
    BiasRelabelingClassifier,
    CSBoostClassifier,
    CSLogitClassifier,
    RobustCSClassifier,
    CSThresholdClassifier
)

ESTIMATORS = (
    B2BoostClassifier(),
    ProfLogitClassifier(optimizer_params={'max_iter': 2}),
    BiasReweighingClassifier(estimator=LogisticRegression()),
    BiasResamplingClassifier(estimator=LogisticRegression()),
    BiasRelabelingClassifier(estimator=LogisticRegression()),
    CSBoostClassifier(fp_cost=1, fn_cost=1),
    CSLogitClassifier(fp_cost=1, fn_cost=1),
    RobustCSClassifier(estimator=CSLogitClassifier(), fp_cost=1, fn_cost=1),
    CSThresholdClassifier(estimator=LogisticRegression(), random_state=42, fp_cost=1, fn_cost=1),
)


def expected_failed_checks(estimator):
    if isinstance(estimator, ProfLogitClassifier):
        return {
            'check_classifier_data_not_an_array': 'Sklearn does not set random_state properly in the test. Tested internally.',
            'check_fit_idempotent': 'Sklearn does not set random_state properly in the test. Tested internally.',
            'check_supervised_y_2d': 'Sklearn does not set random_state properly in the test. Tested internally.'
        }
    if isinstance(estimator, CSThresholdClassifier):
        return {
            'check_decision_proba_consistency': 'CalibratedClassifierCV does not support decision_function.'
        }
    return {}


@parametrize_with_checks([est for est in ESTIMATORS], expected_failed_checks=expected_failed_checks)
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

    from sklearn.utils.estimator_checks import set_random_state as sklearn_set_random_state
    import sklearn.utils
    sklearn.utils.set_random_state = set_random_state
    from sklearn.utils.estimator_checks import check_classifier_data_not_an_array

    check_classifier_data_not_an_array(
        ProfLogitClassifier.__name__,
        ProfLogitClassifier(optimizer_params={'max_iter': 3, 'random_state': 42})
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

    from sklearn.utils.estimator_checks import set_random_state as sklearn_set_random_state
    import sklearn.utils
    sklearn.utils.set_random_state = set_random_state
    from sklearn.utils.estimator_checks import check_fit_idempotent

    check_fit_idempotent(
        ProfLogitClassifier.__name__,
        ProfLogitClassifier(optimizer_params={'max_iter': 3, 'random_state': 42})
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

    from sklearn.utils.estimator_checks import set_random_state as sklearn_set_random_state
    import sklearn.utils
    sklearn.utils.set_random_state = set_random_state
    from sklearn.utils.estimator_checks import check_supervised_y_2d

    check_supervised_y_2d(
        ProfLogitClassifier.__name__,
        ProfLogitClassifier(optimizer_params={'max_iter': 3, 'random_state': 42})
    )
    sklearn.utils.set_random_state = sklearn_set_random_state


@pytest.mark.parametrize("classifier", [
    CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid', random_state=42),
    CSBoostClassifier(),
    CSLogitClassifier(),
    RobustCSClassifier(estimator=CSLogitClassifier()),
])
def test_cost_loss_performance(classifier):
    X, y, tp_cost, fp_cost, tn_cost, fn_cost = load_give_me_some_credit(return_X_y_costs=True, as_frame=True)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', classifier)
    ])

    if isinstance(classifier, CSThresholdClassifier):
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X, tp_cost=tp_cost, fp_cost=fp_cost, tn_cost=tn_cost, fn_cost=fn_cost)
    else:
        pipeline.fit(X, y, model__tp_cost=tp_cost, model__fp_cost=fp_cost,
                     model__tn_cost=tn_cost, model__fn_cost=fn_cost)
        y_pred = pipeline.predict(X)

    performance = cost_loss(y, y_pred, tp_cost=tp_cost, tn_cost=tn_cost,
                            fn_cost=fn_cost, fp_cost=fp_cost, normalize=True)

    assert performance < 750, f"Performance {performance} is not better than 750"
