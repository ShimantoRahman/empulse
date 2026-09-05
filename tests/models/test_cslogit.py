import numpy as np
import pytest

from empulse.metrics import CostMatrix, Metric, Savings
from empulse.models import CSLogitClassifier
from empulse.optimizers import LBFGSBOptimizer


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


def test_works_with_different_loss(X, y):
    from scipy.optimize import OptimizeResult

    clf = CSLogitClassifier(loss=Metric(CostMatrix().add_fp_cost('fp').add_fn_cost('fn'), Savings()))
    clf.fit(X, y, fp=10, fn=1)
    assert clf.result_.x.shape == (3,)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.success is True


class TestDefaultOptimizer:
    """`_optimize` is implemented once on `BaseLogitClassifier`, driven by the
    `_default_optimizer` ClassVar each subclass sets.
    """

    def test_default_optimizer_is_lbfgsb(self):
        assert CSLogitClassifier._default_optimizer is LBFGSBOptimizer

    def test_none_optimizer_falls_back_to_default(self, X, y):
        clf = CSLogitClassifier(optimizer=None)
        clf.fit(X, y, fp_cost=1.0, fn_cost=1.0)
        assert hasattr(clf, 'result_')

    def test_explicit_optimizer_overrides_default(self, X, y):
        clf = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=3))
        clf.fit(X, y, fp_cost=1.0, fn_cost=1.0)
        assert clf.n_iter_ <= 3
