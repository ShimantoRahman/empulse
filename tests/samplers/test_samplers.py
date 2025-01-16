import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from empulse.samplers import BiasRelabler, BiasResampler, CostSensitiveSampler

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
    """Check the compatibility with scikit-learn API"""
    check(estimator)


@parametrize_with_checks_samplers(ESTIMATORS, FIT_PARAMS)
def test_samplers(estimator, check):
    """Check the compatibility with imbalanced-learn API"""
    check(estimator)
