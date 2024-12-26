from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
# from imblearn.utils.estimator_checks import parametrize_with_checks

from empulse.samplers import BiasResampler, BiasRelabler, CostSensitiveSampler

ESTIMATORS = (
    BiasResampler(),
    BiasRelabler(estimator=LogisticRegression()),
    CostSensitiveSampler(),
)


@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
