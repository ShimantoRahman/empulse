import numpy as np
import pytest

from empulse.metrics import CostMatrix, Metric, Savings
from empulse.models import CSLogitClassifier


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
