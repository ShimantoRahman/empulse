import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import NotFittedError, check_is_fitted

from empulse.models import BiasReweighingClassifier
from empulse.models.bias_mitigation.bias_reweighing import _independent_sample_weights


def test_independent_sample_weights():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    weights = _independent_sample_weights(y_true, protected_attr)
    assert np.allclose(
        weights,
        np.array([
            0.375,
            0.375,
            0.375,
            0.375,
            1,
            0.33333333,
            0.33333333,
            0.75,
            0.33333333,
            0.75,
        ]),  # divided by 2 because of the normalization
    )


@pytest.fixture(scope='module')
def X():
    return np.arange(20).reshape(10, 2)


@pytest.fixture(scope='module')
def y():
    return np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])


@pytest.fixture(scope='module')
def sensitive_feature():
    return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


@pytest.fixture(scope='module')
def clf(X, y, sensitive_feature):
    clf = BiasReweighingClassifier(estimator=LogisticRegression(), strategy='statistical parity')
    clf.fit(X, y, sensitive_feature=sensitive_feature)
    return clf


def test_reweighing_init():
    clf = BiasReweighingClassifier(estimator=LogisticRegression(), strategy='statistical parity')
    assert isinstance(clf.estimator, LogisticRegression)
    assert clf.transform_feature is None
    assert clf.strategy == 'statistical parity'


def test_reweighing_with_different_parameters():
    clf = BiasReweighingClassifier(
        estimator=LogisticRegression(), strategy='demographic parity', transform_feature=lambda x: x
    )
    assert isinstance(clf.estimator, LogisticRegression)
    assert clf.transform_feature is not None
    assert isinstance(clf.transform_feature, type(lambda x: x))
    assert clf.strategy == 'demographic parity'


def test_reweighing_fit(X, y, sensitive_feature):
    clf = BiasReweighingClassifier(estimator=LogisticRegression())
    clf.fit(X, y, sensitive_feature=sensitive_feature)
    assert clf.classes_ is not None
    try:
        check_is_fitted(clf.estimator_)
    except NotFittedError:
        pytest.fail('BiasReweighingClassifier is not fitted')


def test_reweighing_predict_proba(clf, X):
    y_pred = clf.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_reweighing_predict(clf, X):
    y_pred = clf.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_reweighing_score(clf, X, y):
    score = clf.score(X, y)
    assert isinstance(score, float)


def test_cloneable_by_sklearn():
    from sklearn.base import clone

    clf = BiasReweighingClassifier(estimator=LogisticRegression())
    clf_clone = clone(clf)
    assert isinstance(clf_clone, BiasReweighingClassifier)
    cloned_params = clf_clone.get_params()
    for key, value in clf.get_params().items():
        if key == 'estimator':
            assert value.__class__ == cloned_params[key].__class__
        else:
            assert value == cloned_params[key]


def test_works_in_cross_validation(X, y):
    from sklearn.model_selection import cross_val_score

    clf = BiasReweighingClassifier(estimator=LogisticRegression())
    scores = cross_val_score(clf, X, y, cv=2)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(scores.astype(np.float64) == scores)


def test_works_in_pipeline(X, y, sensitive_feature):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = BiasReweighingClassifier(estimator=LogisticRegression())
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X, y, clf__sensitive_feature=sensitive_feature)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['clf'], BiasReweighingClassifier)
    assert isinstance(pipe.score(X, y), float)
    assert isinstance((pred := pipe.predict(X)), np.ndarray)
    assert np.all(~np.isnan(pred))


def test_works_in_ensemble(X, y):
    from sklearn.ensemble import BaggingClassifier

    clf = BiasReweighingClassifier(estimator=LogisticRegression())
    bagging = BaggingClassifier(clf, n_estimators=2, random_state=42)
    bagging.fit(X, y)  # no option to pass protected_attr
    assert isinstance(bagging.estimators_[0], BiasReweighingClassifier)
    assert isinstance(bagging.score(X, y), float)
    assert isinstance(bagging.predict(X), np.ndarray)


def test_metadatarouting(X, y, sensitive_feature):
    from sklearn import config_context
    from sklearn.model_selection import GridSearchCV

    param_grid = {'estimator__C': [1, 2]}

    with config_context(enable_metadata_routing=True):
        model = BiasReweighingClassifier(estimator=LogisticRegression())
        model.set_fit_request(sensitive_feature=True)
        search = GridSearchCV(model, param_grid=param_grid, cv=2)
        search.fit(X, y, sensitive_feature=sensitive_feature)
        try:
            check_is_fitted(search)
        except NotFittedError:
            pytest.fail('GridSearchCV is not fitted')
        assert isinstance(search.score(X, y), float)
        assert isinstance(search.predict(X), np.ndarray)
