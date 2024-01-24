import pytest
import numpy as np
from scipy.optimize import OptimizeResult

from empulse.models import ProfLogitClassifier


@pytest.fixture(scope='module')
def clf():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(max_iter=10)
    clf.fit(X, y)
    return clf


def test_proflogit_init():
    clf = ProfLogitClassifier()
    assert clf.C == 1.0
    assert clf.fit_intercept is True
    assert clf.soft_threshold is True
    assert clf.l1_ratio == 1.0
    assert clf.n_jobs is None
    assert clf.default_bounds == (-3, 3)


def test_proflogit_with_different_parameters():
    clf = ProfLogitClassifier(
        C=0.5,
        fit_intercept=False,
        soft_threshold=False,
        l1_ratio=0.5,
        n_jobs=1,
        default_bounds=(-1, 1)
    )
    assert clf.C == 0.5
    assert clf.fit_intercept is False
    assert clf.soft_threshold is False
    assert clf.l1_ratio == 0.5
    assert clf.n_jobs == 1
    assert clf.default_bounds == (-1, 1)


def test_proflogit_fit(clf):
    assert clf.n_dim == 3
    assert isinstance(clf.result, OptimizeResult)


def test_proflogit_fit_no_intercept():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(fit_intercept=False, max_iter=10)
    clf.fit(X, y)
    assert clf.n_dim == 2
    assert isinstance(clf.result, OptimizeResult)


def test_proflogit_predict_proba(clf):
    X = np.random.rand(10, 2)
    y_pred = clf.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_proflogit_predict(clf):
    X = np.random.rand(10, 2)
    y_pred = clf.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_proflogit_score(clf):
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    score = clf.score(X, y)
    assert isinstance(score, float)


def test_proflogit_with_missing_values():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    # Introduce missing values
    X[0, 0] = np.nan
    clf = ProfLogitClassifier(max_iter=10)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_proflogit_with_different_bounds(clf):
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(bounds=[(-1, 1)] * 3, max_iter=10)
    clf.fit(X, y)
    assert clf.n_dim == 3
    assert isinstance(clf.result, OptimizeResult)
    assert clf.result.message == "Maximum number of iterations reached."


def test_cloneable_by_sklearn():
    from sklearn.base import clone
    clf = ProfLogitClassifier(max_iter=10)
    clf_clone = clone(clf)
    assert isinstance(clf_clone, ProfLogitClassifier)
    assert clf.get_params() == clf_clone.get_params()


def test_works_in_cross_validation():
    from sklearn.model_selection import cross_val_score
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(max_iter=2)
    scores = cross_val_score(clf, X, y, cv=2)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(scores.astype(np.float64) == scores)


def test_works_in_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(max_iter=2)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X, y)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['clf'], ProfLogitClassifier)
    assert isinstance(pipe.score(X, y), float)
    assert isinstance(pipe.predict(X), np.ndarray)


def test_works_in_ensemble():
    from sklearn.ensemble import BaggingClassifier
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(max_iter=2)
    bagging = BaggingClassifier(clf, n_estimators=2)
    bagging.fit(X, y)
    assert isinstance(bagging.estimators_[0], ProfLogitClassifier)
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)


def test_works_with_time_stopping_condition():
    from empulse.models.optimizers import RGA
    from scipy.optimize import OptimizeResult
    from time import perf_counter

    def optimize(objective, bounds, max_time=5, **kwargs) -> OptimizeResult:
        rga = RGA(**kwargs)

        start = perf_counter()
        for _ in rga.optimize(objective, bounds):
            if perf_counter() - start > max_time:
                rga.result.message = "Maximum time reached."
                rga.result.success = True
                break
        return rga.result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, max_time=0.1)

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    proflogit.fit(X, y)

    assert isinstance(proflogit.result, OptimizeResult)
    assert proflogit.result.message == "Maximum time reached."
    assert proflogit.result.success is True
    assert proflogit.result.x.shape == (3,)


def test_works_with_different_optimizers_bfgs():
    from scipy.optimize import minimize, OptimizeResult
    import numpy as np

    def optimize(objective, bounds, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(len(bounds))
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='BFGS',
            **kwargs
        )
        return result

    proflogit = ProfLogitClassifier(optimize_fn=optimize)

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    proflogit.fit(X, y)

    assert isinstance(proflogit.result, OptimizeResult)
    assert proflogit.result.x.shape == (3,)


def test_works_with_different_optimizers_lbfgsb():
    from scipy.optimize import minimize, OptimizeResult
    import numpy as np

    def optimize(objective, bounds, max_iter=10000, **kwargs) -> OptimizeResult:
        initial_guess = np.zeros(len(bounds))
        result = minimize(
            lambda x: -objective(x),  # inverse objective function
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,
            },
            **kwargs
        )
        return result

    proflogit = ProfLogitClassifier(optimize_fn=optimize, max_iter=10)

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    proflogit.fit(X, y)

    assert isinstance(proflogit.result, OptimizeResult)
    assert proflogit.result.x.shape == (3,)


def test_works_with_different_loss_empa():
    from empulse.metrics import empa_score
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(loss_fn=empa_score, max_iter=10)
    clf.fit(X, y)
    assert clf.result.x.shape == (3,)
    assert isinstance(clf.result, OptimizeResult)
    assert clf.result.message == "Maximum number of iterations reached."


def test_works_with_different_loss_auc():
    from sklearn.metrics import roc_auc_score
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(loss_fn=roc_auc_score, max_iter=10)
    clf.fit(X, y)
    assert clf.result.x.shape == (3,)
    assert isinstance(clf.result, OptimizeResult)
    assert clf.result.message == "Maximum number of iterations reached."


def test_one_variable():
    X = np.random.rand(10, 1)
    y = np.random.randint(0, 2, 10)
    clf = ProfLogitClassifier(fit_intercept=False, max_iter=10)
    clf.fit(X, y)
    assert clf.result.x.shape == (1,)
    assert isinstance(clf.result, OptimizeResult)
    assert clf.result.message == "Maximum number of iterations reached."
