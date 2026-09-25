"""
The three bias-mitigation classifiers, tested once and parametrised over the estimator.

This replaces ``test_bias_relabeling.py``, ``test_bias_resampling.py`` and
``test_bias_reweighing.py``, which were the same 148-line file three times over: the relabeling and
resampling versions differed by six lines, all of them the class name. One of the copies had drifted
-- the relabeling file reported ``'BiasReweighingClassifier is not fitted'`` on failure.

``BiasReweighingClassifier`` needs an imbalanced target for its weights to be interesting, so the
target is chosen per classifier by the ``target`` fixture rather than shared.
"""

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import NotFittedError, check_is_fitted

from empulse.models import BiasRelabelingClassifier, BiasResamplingClassifier, BiasReweighingClassifier
from empulse.models.bias_mitigation.bias_reweighing import _independent_sample_weights

CLASSIFIERS = [BiasRelabelingClassifier, BiasResamplingClassifier, BiasReweighingClassifier]

# The warning each classifier emits when the protected attribute is degenerate. GridSearchCV folds
# can legitimately contain a single group, so the metadata-routing test has to tolerate it.
DEGENERATE_GROUP_WARNINGS = [
    'ignore:sensitive_feature only contains one class, no relabeling is performed.',
    'ignore:sensitive_feature only contains one class, no resampling is performed.',
]


@pytest.fixture(params=CLASSIFIERS, ids=lambda cls: cls.__name__)
def classifier_cls(request):
    return request.param


@pytest.fixture
def target(classifier_cls, y, imbalanced_y):
    """Reweighing needs an imbalanced target; the other two work on the balanced one."""
    return imbalanced_y if classifier_cls is BiasReweighingClassifier else y


@pytest.fixture(scope='session')
def imbalanced_y():
    return np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])


@pytest.fixture
def fitted(classifier_cls, X, target, sensitive_feature):
    clf = classifier_cls(estimator=LogisticRegression(), strategy='statistical parity')
    clf.fit(X, target, sensitive_feature=sensitive_feature)
    return clf


def test_init_stores_parameters_unchanged(classifier_cls):
    clf = classifier_cls(estimator=LogisticRegression(), strategy='statistical parity')
    assert isinstance(clf.estimator, LogisticRegression)
    assert clf.transform_feature is None
    assert clf.strategy == 'statistical parity'


def test_init_accepts_a_transform_feature(classifier_cls):
    clf = classifier_cls(estimator=LogisticRegression(), strategy='demographic parity', transform_feature=lambda x: x)
    assert isinstance(clf.estimator, LogisticRegression)
    assert callable(clf.transform_feature)
    assert clf.strategy == 'demographic parity'


def test_fit_fits_the_wrapped_estimator(classifier_cls, X, target, sensitive_feature):
    clf = classifier_cls(estimator=LogisticRegression())
    clf.fit(X, target, sensitive_feature=sensitive_feature)
    assert clf.classes_ is not None
    try:
        check_is_fitted(clf.estimator_)
    except NotFittedError:
        pytest.fail(f'{classifier_cls.__name__} did not fit its inner estimator')


def test_predict_proba_returns_probabilities(fitted, X):
    y_pred = fitted.predict_proba(X)
    assert y_pred.shape == (10, 2)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_predict_returns_binary_labels(fitted, X):
    y_pred = fitted.predict(X)
    assert y_pred.shape == (10,)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_score_returns_a_float(fitted, X, target):
    assert isinstance(fitted.score(X, target), float)


@pytest.mark.filterwarnings(*DEGENERATE_GROUP_WARNINGS)
def test_metadata_routing_through_grid_search(classifier_cls, X, target, sensitive_feature):
    """``sensitive_feature`` must survive being routed through GridSearchCV's inner folds."""
    from sklearn import config_context
    from sklearn.model_selection import GridSearchCV

    with config_context(enable_metadata_routing=True):
        model = classifier_cls(estimator=LogisticRegression())
        model.set_fit_request(sensitive_feature=True)
        search = GridSearchCV(model, param_grid={'estimator__C': [1, 2]}, cv=2)
        search.fit(X, target, sensitive_feature=sensitive_feature)
        try:
            check_is_fitted(search)
        except NotFittedError:
            pytest.fail('GridSearchCV is not fitted')
        assert isinstance(search.score(X, target), float)
        assert isinstance(search.predict(X), np.ndarray)


def test_sensitive_feature_length_mismatch_raises(classifier_cls, X, target, sensitive_feature):
    """Regression test: a sensitive_feature of the wrong length used to be silently accepted."""
    model = classifier_cls(estimator=LogisticRegression())
    with pytest.raises(ValueError, match='sensitive_feature must have the same length as y'):
        model.fit(X, target, sensitive_feature=sensitive_feature[:-1])


def test_independent_sample_weights():
    """``BiasReweighingClassifier``'s weighting helper, which has no counterpart on the other two."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    weights = _independent_sample_weights(y_true, protected_attr)
    # Halved relative to the raw group ratios because the weights are normalised.
    assert np.allclose(
        weights,
        np.array([0.375, 0.375, 0.375, 0.375, 1, 0.33333333, 0.33333333, 0.75, 0.33333333, 0.75]),
    )


@pytest.mark.parametrize('strategy', ['statistical parity', 'demographic parity'])
def test_mitigation_narrows_the_positive_rate_gap(classifier_cls, strategy):
    """
    The point of all three: the model flags the two groups at more similar rates than it would unmitigated.

    The labels depend on the protected attribute, which is also a feature, so a plain logistic
    regression flags one group far more often than the other (a gap of 0.34 on this data).
    """
    rng = np.random.default_rng(42)
    n_samples = 1200
    group = (rng.random(n_samples) < 0.5).astype(int)
    features = rng.normal(0.0, 1.0, (n_samples, 2))
    y = (rng.random(n_samples) < expit(1.5 * features[:, 0] + 1.5 * group - 0.75)).astype(int)
    X = np.column_stack([features, group])

    def positive_rate_gap(y_pred):
        return abs(y_pred[group == 1].mean() - y_pred[group == 0].mean())

    unmitigated = positive_rate_gap(LogisticRegression().fit(X, y).predict(X))
    model = classifier_cls(LogisticRegression(), strategy=strategy).fit(X, y, sensitive_feature=group)
    assert positive_rate_gap(model.predict(X)) < unmitigated / 2


NON_ZERO_ONE_LABELS = (np.array([-1, 1]), np.array(['no', 'yes']), np.array([2, 5]))


class TestNonZeroOneLabels:
    """Regression tests: relabeling and reweighing assumed the target was encoded as 0/1.

    ``BiasRelabler`` chose candidates by comparing the raw labels to 0 and 1 and relabelled them
    with the literal values 0 and 1, so ``-1``/``1`` labels gained a third class ``0`` and
    ``predict`` raised an ``IndexError``. ``BiasReweighingClassifier`` computed its group priors from
    the raw labels' mean, so ``-1``/``1`` labels got different weights and strings raised.
    """

    @pytest.fixture
    def biased_data(self):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=400, random_state=0)
        sensitive_feature = (np.random.default_rng(0).random(400) < 0.3 + 0.4 * y).astype(int)
        return X, y, sensitive_feature

    @pytest.mark.parametrize('classifier_cls', [BiasRelabelingClassifier, BiasReweighingClassifier])
    @pytest.mark.parametrize('labels', NON_ZERO_ONE_LABELS, ids=['minus_one_one', 'strings', 'two_five'])
    def test_same_model_as_with_zero_one_labels(self, biased_data, classifier_cls, labels):
        X, y, sensitive_feature = biased_data
        reference = classifier_cls(LogisticRegression()).fit(X, y, sensitive_feature=sensitive_feature)
        model = classifier_cls(LogisticRegression()).fit(X, labels[y], sensitive_feature=sensitive_feature)

        np.testing.assert_array_equal(model.classes_, labels)
        np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X))
        np.testing.assert_array_equal(model.predict(X), labels[reference.predict(X)])
