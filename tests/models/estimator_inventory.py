"""
The inventory of estimators under test, shared by the model test modules.

``test_models.py`` (scikit-learn conformance) and ``test_sklearn_integration.py`` (clone, pickle,
Pipeline, cross-validation, ensembling) both need the same list of cheaply-configured estimators, so
it lives here rather than being written out twice.

Every estimator is deliberately configured to fit fast -- ``max_iter=2``, ``n_estimators=2`` -- since
these suites check plumbing, not learning quality.
"""

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from empulse.models import (
    B2BoostClassifier,
    BiasRelabelingClassifier,
    BiasResamplingClassifier,
    BiasReweighingClassifier,
    CSBaggingClassifier,
    CSBoostClassifier,
    CSForestClassifier,
    CSLogitClassifier,
    CSRateClassifier,
    CSThresholdClassifier,
    CSTreeClassifier,
    ProfLogitClassifier,
    ProfMEMPMClassifier,
    ProfMPMClassifier,
    ProfSRClassifier,
    ProfTreeClassifier,
    RobustCSClassifier,
)
from empulse.optimizers import GeneticAlgorithmOptimizer, LBFGSBOptimizer


def make_estimators():
    """
    A fresh instance of every estimator in ``empulse.models.__all__``.

    Returns instances rather than a module-level tuple so that a test which fits one cannot leak
    fitted state into another.
    """
    return [
        BiasReweighingClassifier(estimator=LogisticRegression(max_iter=2)),
        BiasResamplingClassifier(estimator=LogisticRegression(max_iter=2)),
        BiasRelabelingClassifier(estimator=LogisticRegression(max_iter=2)),
        B2BoostClassifier(XGBClassifier(n_estimators=2, max_depth=1)),
        ProfLogitClassifier(
            tp_cost=-1, fp_cost=1, optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42)
        ),
        ProfTreeClassifier(max_iter=2, population_size=10, random_state=42),
        CSBoostClassifier(XGBClassifier(n_estimators=2, max_depth=1), fp_cost=1, fn_cost=1),
        CSLogitClassifier(fp_cost=1, fn_cost=1),
        CSTreeClassifier(max_depth=2, fp_cost=1, fn_cost=1, random_state=42),
        CSForestClassifier(n_estimators=2, max_depth=2, fp_cost=1, fn_cost=1, random_state=42),
        CSBaggingClassifier(
            estimator=CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=2)),
            n_estimators=2,
            fp_cost=1,
            fn_cost=1,
            random_state=42,
        ),
        RobustCSClassifier(estimator=CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=2)), fp_cost=1, fn_cost=1),
        CSThresholdClassifier(estimator=LogisticRegression(max_iter=2), random_state=42, fp_cost=1, fn_cost=1),
        CSRateClassifier(estimator=LogisticRegression(max_iter=2), fp_cost=1, fn_cost=1),
        ProfMPMClassifier(tp_cost=-1, fp_cost=1),
        ProfMEMPMClassifier(tp_cost=-1, fp_cost=1),
        ProfSRClassifier(tp_cost=-1, fp_cost=1, generations=2, population_size=20, random_state=42),
    ]


# The bias-mitigation classifiers are the only ones that need a fit-time argument beyond X and y.
NEEDS_SENSITIVE_FEATURE = (
    BiasRelabelingClassifier,
    BiasResamplingClassifier,
    BiasReweighingClassifier,
)


def estimator_id(estimator):
    """A short, stable test id: the class name, plus the inner estimator's when there is one."""
    inner = estimator.get_params(deep=False).get('estimator')
    if inner is not None:
        return f'{type(estimator).__name__}({type(inner).__name__})'
    return type(estimator).__name__
