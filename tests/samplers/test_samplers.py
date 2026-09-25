import inspect

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.estimator_checks import parametrize_with_checks

from empulse.samplers import BiasRelabler, BiasResampler, CostSensitiveSampler

from .._estimator_common import iter_invalid_params
from .sampler_checks import parametrize_with_checks_samplers

ESTIMATORS = (
    BiasResampler(random_state=42),
    BiasRelabler(estimator=LogisticRegression()),
    CostSensitiveSampler(method='rejection sampling', random_state=42),
    CostSensitiveSampler(method='oversampling', random_state=42),
)

FIT_PARAMS = (
    {'sensitive_feature': np.append(np.zeros(500), np.ones(500))},
    {'sensitive_feature': np.append(np.zeros(500), np.ones(500))},
    {'fp_cost': np.ones(1000) * 10, 'fn_cost': np.ones(1000)},
    {'fp_cost': np.ones(1000) * 10, 'fn_cost': np.ones(1000)},
)


@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API."""
    check(estimator)


@parametrize_with_checks_samplers(ESTIMATORS, FIT_PARAMS)
def test_samplers(estimator, check):
    """Check the compatibility with imbalanced-learn API."""
    check(estimator)


def _invalid_param_cases():
    """One case per (sampler class, constructor parameter), so the id names the parameter.

    Sorted, because a set of classes iterates in an order that differs between processes, and
    pytest-xdist refuses to run a suite whose workers collected different tests.
    """
    for estimator_class in sorted({type(est) for est in ESTIMATORS}, key=lambda c: c.__name__):
        for name, invalid in iter_invalid_params(estimator_class):
            yield pytest.param(estimator_class, name, invalid, id=f'{estimator_class.__name__}-{name}')


@pytest.mark.parametrize(('estimator_class', 'param_name', 'invalid_params'), _invalid_param_cases())
def test_invalid_params(estimator_class, param_name, invalid_params):
    """Every constructor parameter must be rejected by scikit-learn's parameter validation."""
    parameters = inspect.signature(estimator_class.__init__).parameters
    # Supply a valid estimator when the sampler needs one, unless that parameter is the one under test.
    defaults = {'estimator': LogisticRegression()} if 'estimator' in parameters and param_name != 'estimator' else {}
    model = estimator_class(**defaults, **invalid_params)
    with pytest.raises(InvalidParameterError):
        model.fit_resample(1, 1)
