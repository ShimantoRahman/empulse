import numpy as np
import pytest

from empulse.models import CSLogitClassifier


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


def test_works_with_time_stopping_condition(X, y):
    from empulse.optimizers import Generation
    from scipy.optimize import OptimizeResult
    from time import perf_counter

    def optimize(objective, X, max_time=1, **kwargs) -> OptimizeResult:
        rga = Generation(**kwargs)
        assert max_time == 0.1
        bounds = [(-10, 10)] * X.shape[1]

        def _objective(x):
            score, _ = objective(x)
            return score

        start = perf_counter()
        for _ in rga.optimize(_objective, bounds):
            if perf_counter() - start > max_time:
                rga.result.message = 'Maximum time reached.'
                rga.result.success = True
                break
        return rga.result

    cslogit = CSLogitClassifier(optimize_fn=optimize, optimizer_params={'max_time': 0.1})

    cslogit.fit(X, y, fp_cost=1, fn_cost=1)

    assert isinstance(cslogit.result_, OptimizeResult)
    assert cslogit.result_.message == 'Maximum time reached.'
    assert cslogit.result_.success is True
    assert cslogit.result_.x.shape == (3,)


def test_works_with_different_optimizers_bfgs(X, y):
    from scipy.optimize import minimize, OptimizeResult
    import numpy as np

    def optimize(objective, X, max_iter=10000, **kwargs) -> OptimizeResult:
        assert max_iter == 2
        initial_guess = np.zeros(X.shape[1], order='F', dtype=X.dtype)
        result = minimize(
            objective,  # inverse objective function
            initial_guess,
            jac=True,
            method='BFGS',
            options={
                'maxiter': max_iter,
                'gtol': 1e-4,
            },
            **kwargs,
        )
        return result

    cslogit = CSLogitClassifier(optimize_fn=optimize, optimizer_params={'max_iter': 2})

    cslogit.fit(X, y, fp_cost=1, fn_cost=1)

    assert isinstance(cslogit.result_, OptimizeResult)
    assert cslogit.result_.x.shape == (3,)


def test_works_with_different_loss(X, y):
    from empulse.metrics import expected_savings_score
    from scipy.optimize import OptimizeResult

    clf = CSLogitClassifier(loss=expected_savings_score)
    clf.fit(X, y, fp_cost=1, fn_cost=1)
    assert clf.result_.x.shape == (3,)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.success is True
