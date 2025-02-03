from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from sklearn.utils.validation import NotFittedError, check_is_fitted

from empulse.models import ProfLogitClassifier


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([0, 1] * 5)


@pytest.fixture(scope='module')
def clf(X, y):
    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10})
    clf.fit(X, y)
    return clf


def test_proflogit_init():
    clf = ProfLogitClassifier()
    assert clf.C == 1.0
    assert clf.fit_intercept is True
    assert clf.soft_threshold is False
    assert clf.l1_ratio == 1.0
    assert clf.n_jobs is None


def test_proflogit_with_different_parameters():
    clf = ProfLogitClassifier(
        C=0.5,
        fit_intercept=False,
        soft_threshold=False,
        l1_ratio=0.5,
        n_jobs=1,
    )
    assert clf.C == 0.5
    assert clf.fit_intercept is False
    assert clf.soft_threshold is False
    assert clf.l1_ratio == 0.5
    assert clf.n_jobs == 1


def test_proflogit_fit(clf):
    assert isinstance(clf.result_, OptimizeResult)


def test_proflogit_fit_no_intercept(X, y):
    clf = ProfLogitClassifier(fit_intercept=False, optimizer_params={'max_iter': 2})
    clf.fit(X, y)
    try:
        check_is_fitted(clf)
    except NotFittedError:
        pytest.fail('ProfLogitClassifier is not fitted')
    assert isinstance(clf.result_, OptimizeResult)


def test_proflogit_predict_proba(clf, X):
    y_pred = clf.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_proflogit_predict(clf, X):
    y_pred = clf.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_proflogit_score(clf, X, y):
    score = clf.score(X, y)
    assert isinstance(score, float)


def test_proflogit_with_missing_values(X, y):
    X = np.array(X, dtype=float)
    # Introduce missing values
    X[0, 0] = np.nan
    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10})
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_proflogit_with_different_bounds(clf, X, y):
    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10, 'bounds': (-1, 1)})
    clf.fit(X, y)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.message == 'Maximum number of iterations reached.'


def test_cloneable_by_sklearn():
    from sklearn.base import clone

    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2})
    clf_clone = clone(clf)
    assert isinstance(clf_clone, ProfLogitClassifier)
    assert clf.get_params() == clf_clone.get_params()


def test_works_in_cross_validation(X, y):
    from sklearn.model_selection import cross_val_score

    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10})
    scores = cross_val_score(clf, X, y, cv=2)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(scores.astype(np.float64) == scores)


def test_works_in_pipeline(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10})
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X, y)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['clf'], ProfLogitClassifier)
    assert isinstance(pipe.score(X, y), float)
    assert isinstance(pipe.predict(X), np.ndarray)


def test_works_in_ensemble(X, y):
    from sklearn.ensemble import BaggingClassifier

    clf = ProfLogitClassifier(optimizer_params={'max_iter': 2, 'population_size': 10})
    bagging = BaggingClassifier(clf, n_estimators=2, random_state=42)
    bagging.fit(X, y)
    assert isinstance(bagging.estimators_[0], ProfLogitClassifier)
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)


def test_works_with_time_stopping_condition(X, y):
    from time import perf_counter

    from scipy.optimize import OptimizeResult

    from empulse.optimizers import Generation

    def optimize(objective: Callable, X: NDArray, max_time: float = 0.01, **kwargs: Any) -> OptimizeResult:
        rga = Generation(**kwargs)
        bounds = [(-5, 5)] * X.shape[1]

        start = perf_counter()
        for _ in rga.optimize(objective, bounds):
            if perf_counter() - start > max_time:
                rga.result.message = 'Maximum time reached.'
                rga.result.success = True
                break
        return rga.result

    proflogit = ProfLogitClassifier(optimize_fn=optimize)

    proflogit.fit(X, y)

    assert isinstance(proflogit.result_, OptimizeResult)
    assert proflogit.result_.message == 'Maximum time reached.'
    assert proflogit.result_.success is True
    assert proflogit.result_.x.shape == (3,)


def test_works_with_different_optimizers_bfgs(X, y):
    import numpy as np
    from scipy.optimize import OptimizeResult, minimize

    def optimize(objective: Callable, X: NDArray, **kwargs: Any) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='BFGS',
            options={'maxiter': 2},
            **kwargs,
        )
        return result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, soft_threshold=True)

    proflogit.fit(X, y)

    assert isinstance(proflogit.result_, OptimizeResult)
    assert proflogit.result_.x.shape == (3,)


def test_works_with_different_optimizers_lbfgsb(X, y):
    import numpy as np
    from scipy.optimize import OptimizeResult, minimize

    def optimize(objective: Callable, X: NDArray, max_iter: int = 10000, **kwargs: Any) -> OptimizeResult:
        initial_guess = np.zeros(X.shape[1])
        bounds = [(-5, 5)] * X.shape[1]
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,
            },
            **kwargs,
        )
        return result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, optimizer_params={'max_iter': 2})

    proflogit.fit(X, y)

    assert isinstance(proflogit.result_, OptimizeResult)
    assert proflogit.result_.x.shape == (3,)


def test_works_with_different_loss_empa(X, y):
    from empulse.metrics import empa_score

    clf = ProfLogitClassifier(loss=empa_score, optimizer_params={'max_iter': 2, 'population_size': 10})
    clf.fit(X, y)
    assert clf.result_.x.shape == (3,)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.message == 'Maximum number of iterations reached.'


def test_works_with_different_loss_auc(X, y):
    from sklearn.metrics import roc_auc_score

    clf = ProfLogitClassifier(loss=roc_auc_score, optimizer_params={'max_iter': 2, 'population_size': 10})
    clf.fit(X, y)
    assert clf.result_.x.shape == (3,)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.message == 'Maximum number of iterations reached.'


def test_one_variable(y):
    X = np.arange(10).reshape(10, 1)
    clf = ProfLogitClassifier(fit_intercept=False, optimizer_params={'max_iter': 2, 'population_size': 10})
    clf.fit(X, y)
    assert clf.result_.x.shape == (1,)
    assert isinstance(clf.result_, OptimizeResult)
    assert clf.result_.message == 'Maximum number of iterations reached.'
