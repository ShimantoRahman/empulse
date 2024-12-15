from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
# from imblearn.utils.estimator_checks import parametrize_with_checks

from empulse.samplers import BiasResampler, BiasRelabler


ESTIMATORS = (
    BiasResampler(),
    BiasRelabler(estimator=LogisticRegression()),
)


# @parametrize_with_checks(ESTIMATORS)
# def test_samplers(sampler, check):
#     check(sampler)

# TODO: change to parametrize_with_checks from imbalanced-learn once it supports scikit-learn 1.6.0's Testing Framework
@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
