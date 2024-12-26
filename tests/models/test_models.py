from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
import numpy as np


from empulse.models import (
    B2BoostClassifier,
    ProfLogitClassifier,
    BiasReweighingClassifier,
    BiasResamplingClassifier,
    BiasRelabelingClassifier,
    CSBoostClassifier,
    CSLogitClassifier,
    RobustCSClassifier,
    CostThresholdClassifier
)

ESTIMATORS = (
    B2BoostClassifier(),
    ProfLogitClassifier(optimizer_params={'max_iter': 2}),
    BiasReweighingClassifier(estimator=LogisticRegression()),
    BiasResamplingClassifier(estimator=LogisticRegression()),
    BiasRelabelingClassifier(estimator=LogisticRegression()),
    CSBoostClassifier(),
    CSLogitClassifier(),
    RobustCSClassifier(estimator=CSLogitClassifier()),
    CostThresholdClassifier(estimator=LogisticRegression())
)


def expected_failed_checks(estimator):
    if isinstance(estimator, ProfLogitClassifier):
        return {
            'check_classifier_data_not_an_array': 'Sklearn does not set random_state properly in the test. Tested internally.',
            'check_fit_idempotent': 'Sklearn does not set random_state properly in the test. Tested internally.',
            'check_supervised_y_2d': 'Sklearn does not set random_state properly in the test. Tested internally.'
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
