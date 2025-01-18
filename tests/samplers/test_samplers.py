import inspect

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.estimator_checks import parametrize_with_checks

from empulse.samplers import BiasRelabler, BiasResampler, CostSensitiveSampler

from .sampler_checks import parametrize_with_checks_samplers

ESTIMATORS = (
    BiasResampler(random_state=42),
    BiasRelabler(estimator=LogisticRegression()),
    CostSensitiveSampler(method='rejection sampling', random_state=42),
    CostSensitiveSampler(method='oversampling', random_state=42),
)
ESTIMATOR_CLASSES = {est.__class__ for est in ESTIMATORS}

FIT_PARAMS = (
    {'sensitive_feature': np.append(np.zeros(500), np.ones(500))},
    {'sensitive_feature': np.append(np.zeros(500), np.ones(500))},
    {'fp_cost': np.ones(1000) * 10, 'fn_cost': np.ones(1000)},
    {'fp_cost': np.ones(1000) * 10, 'fn_cost': np.ones(1000)},
)


@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)


@parametrize_with_checks_samplers(ESTIMATORS, FIT_PARAMS)
def test_samplers(estimator, check):
    """Check the compatibility with imbalanced-learn API"""
    check(estimator)


class InvalidParameter:
    pass


def generate_invalid_params(estimator_class):
    parameters = inspect.signature(estimator_class.__init__).parameters
    takes_estimator = 'estimator' in parameters
    return [{param: InvalidParameter()} for param in parameters if param != 'self'], takes_estimator


@pytest.mark.parametrize('estimator_class', ESTIMATOR_CLASSES)
def test_invalid_params(estimator_class):
    X, y = 1, 1
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
            model.fit_resample(X, y)
