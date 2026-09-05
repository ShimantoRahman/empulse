import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from sklearn.utils.validation import NotFittedError, check_is_fitted

from empulse.models import ProfLogitClassifier
from empulse.optimizers import GeneticAlgorithmOptimizer


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


@pytest.fixture(scope='module')
def clf(X, y):
    clf = ProfLogitClassifier(
        tp_cost=-1, fp_cost=1, optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42)
    )
    clf.fit(X, y)
    return clf


def test_proflogit_with_different_parameters():
    clf = ProfLogitClassifier(
        tp_cost=-1,
        fp_cost=1,
        C=0.5,
        fit_intercept=False,
        soft_threshold=False,
        l1_ratio=0.5,
    )
    assert clf.C == 0.5
    assert clf.fit_intercept is False
    assert clf.soft_threshold is False
    assert clf.l1_ratio == 0.5


def test_proflogit_fit(clf):
    assert isinstance(clf.result_, OptimizeResult)


def test_proflogit_fit_no_intercept(X, y):
    clf = ProfLogitClassifier(
        tp_cost=-1, fp_cost=1, fit_intercept=False, optimizer=GeneticAlgorithmOptimizer(max_iter=2)
    )
    clf.fit(X, y)
    try:
        check_is_fitted(clf)
    except NotFittedError:
        pytest.fail('ProfLogitClassifier is not fitted')
    assert isinstance(clf.result_, OptimizeResult)


def test_one_variable(y):
    X = np.arange(10).reshape(10, 1)
    clf = ProfLogitClassifier(
        tp_cost=-1,
        fp_cost=1,
        fit_intercept=False,
        optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42),
    )
    clf.fit(X, y)
    assert clf.result_.x.shape == (1,)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.message == 'Maximum number of iterations reached.'


class TestDefaultOptimizer:
    """`_optimize` is implemented once on `BaseLogitClassifier`, driven by the
    `_default_optimizer` ClassVar each subclass sets.
    """

    def test_default_optimizer_is_genetic_algorithm(self):
        assert ProfLogitClassifier._default_optimizer is GeneticAlgorithmOptimizer

    def test_none_optimizer_falls_back_to_default(self, X, y):
        clf = ProfLogitClassifier(tp_cost=-1, fp_cost=1, optimizer=None)
        clf.fit(X, y)
        assert isinstance(clf.result_, OptimizeResult)
