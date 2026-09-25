"""Tests for the two logistic models, CSLogitClassifier and ProfLogitClassifier."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from sklearn.utils.validation import NotFittedError, check_is_fitted

from empulse.metrics import CostMatrix, Metric, Savings
from empulse.models import CSLogitClassifier, ProfLogitClassifier
from empulse.optimizers import GeneticAlgorithmOptimizer, LBFGSBOptimizer


@pytest.fixture(scope='module')
def proflogit(X, y):
    clf = ProfLogitClassifier(
        tp_cost=-1, fp_cost=1, optimizer=GeneticAlgorithmOptimizer(max_iter=2, population_size=10, random_state=42)
    )
    clf.fit(X, y)
    return clf


@pytest.mark.parametrize(
    ('classifier', 'default'),
    [(CSLogitClassifier, LBFGSBOptimizer), (ProfLogitClassifier, GeneticAlgorithmOptimizer)],
    ids=['CSLogitClassifier', 'ProfLogitClassifier'],
)
class TestDefaultOptimizer:
    """`_resolve_optimizer` is implemented once on `BaseLogitClassifier`, driven by the
    `_default_optimizer` ClassVar each subclass sets.

    These check the resolved optimizer rather than fitting with it: the default genetic algorithm
    runs at least 250 generations, and `_fit_estimator` calls `_resolve_optimizer` for every fit.
    """

    def test_default_optimizer(self, classifier, default):
        assert classifier._default_optimizer is default

    def test_none_optimizer_falls_back_to_default(self, classifier, default):
        assert isinstance(classifier(optimizer=None)._resolve_optimizer(), default)

    def test_explicit_optimizer_overrides_default(self, classifier, default):
        optimizer = default(max_iter=3)
        assert classifier(optimizer=optimizer)._resolve_optimizer() is optimizer


class TestCSLogit:
    def test_works_with_different_loss(self, X, y):
        clf = CSLogitClassifier(loss=Metric(CostMatrix().add_fp_cost('fp').add_fn_cost('fn'), Savings()))
        clf.fit(X, y, fp=10, fn=1)
        assert clf.result_.x.shape == (3,)
        assert isinstance(clf.result_, OptimizeResult)
        assert clf.result_.success is True

    def test_explicit_optimizer_is_used_for_fitting(self, X, y):
        clf = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=3))
        clf.fit(X, y, fp_cost=1.0, fn_cost=1.0)
        assert clf.n_iter_ <= 3

    @pytest.mark.parametrize('fit_intercept', [True, False])
    def test_fits_read_only_data(self, X, y, fit_intercept):
        """Such as the memory-mapped arrays joblib hands to parallel workers, e.g. in GridSearchCV."""
        expected = CSLogitClassifier(fit_intercept=fit_intercept).fit(X, y, fp_cost=1.0, fn_cost=5.0).result_.x
        X = np.array(X, dtype=np.float64)
        X.flags.writeable = False
        clf = CSLogitClassifier(fit_intercept=fit_intercept).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        np.testing.assert_array_equal(clf.result_.x, expected)


class TestProfLogit:
    def test_stores_its_parameters(self):
        clf = ProfLogitClassifier(tp_cost=-1, fp_cost=1, C=0.5, fit_intercept=False, l1_ratio=0.5)
        assert clf.C == 0.5
        assert clf.fit_intercept is False
        assert clf.l1_ratio == 0.5

    def test_fit(self, proflogit):
        assert isinstance(proflogit.result_, OptimizeResult)

    def test_fit_no_intercept(self, X, y):
        clf = ProfLogitClassifier(
            tp_cost=-1, fp_cost=1, fit_intercept=False, optimizer=GeneticAlgorithmOptimizer(max_iter=2, random_state=42)
        )
        clf.fit(X, y)
        try:
            check_is_fitted(clf)
        except NotFittedError:
            pytest.fail('ProfLogitClassifier is not fitted')
        assert isinstance(clf.result_, OptimizeResult)

    def test_one_variable(self, y):
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
