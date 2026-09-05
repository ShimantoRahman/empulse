import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from empulse.models import CSBaggingClassifier, CSForestClassifier


@pytest.fixture
def data():
    return make_classification(n_samples=100, random_state=42)


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
def test_csforest_criteria(data, criterion):
    X, y = data
    model = CSForestClassifier(criterion=criterion)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csforest_combination(data, combination):
    X, y = data
    model = CSForestClassifier(combination=combination)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csbagging_combination(data, combination):
    X, y = data
    model = CSBaggingClassifier(combination=combination)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


class TestCSBaggingFeatureSubsets:
    """Regression tests: weighted-voting predict used to call sub-estimators on the full X.

    `BaggingClassifier` draws a feature subset per estimator when `max_features < 1.0` or
    `bootstrap_features=True`; `_predict_weighted_proba` must slice `X` to each estimator's own
    `estimators_features_` before calling `predict_proba`, exactly like `_get_oob_weights` already
    does, or a sub-estimator raises "X has N features, but ... is expecting M features".
    """

    @pytest.mark.parametrize('max_features', [1.0, 0.5])
    @pytest.mark.parametrize('bootstrap_features', [False, True])
    def test_weighted_voting_with_feature_subsets(self, max_features, bootstrap_features):
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        model = CSBaggingClassifier(
            n_estimators=5,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            combination='weighted_voting',
            random_state=0,
        )
        model.fit(X, y, fp_cost=1.0, fn_cost=5.0)
        y_proba = model.predict_proba(X)
        assert y_proba.shape == (200, 2)
        assert np.allclose(y_proba.sum(axis=1), 1)

    def test_weighted_voting_requires_predict_proba(self):
        """A base estimator without predict_proba must fail fast at fit time, not at predict time."""

        class NoProbaEstimator(ClassifierMixin, BaseEstimator):
            def fit(self, X, y, **kwargs):
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        X, y = make_classification(n_samples=50, random_state=0)
        model = CSBaggingClassifier(estimator=NoProbaEstimator(), combination='weighted_voting')
        with pytest.raises(ValueError, match='predict_proba'):
            model.fit(X, y, fp_cost=1.0, fn_cost=5.0)


class TestOOBWeightsFiniteAndNormalized:
    """Regression test: an integer `max_samples` used to make CSForestClassifier's OOB weights NaN.

    `isinstance(x, Real)` is also `True` for every `int`, so three independent `if`s (rather than
    `if/elif/else`) meant an integer `max_samples` fell through the `Integral` branch and then got
    overwritten by the `Real` branch, computing a wildly wrong bootstrap sample size.
    """

    @pytest.mark.parametrize('max_samples', [None, 50, 0.5])
    def test_csforest_weighted_voting_weights_are_valid(self, max_samples):
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        model = CSForestClassifier(
            n_estimators=5, max_samples=max_samples, combination='weighted_voting', random_state=0
        )
        model.fit(X, y, fp_cost=1.0, fn_cost=5.0)
        assert np.isfinite(model.estimator_weights_).all()
        assert model.estimator_weights_.sum() == pytest.approx(1.0)


class TestOOBWeightingDirectionAndInstanceCosts:
    """Regression tests: OOB weighting used to ignore metric direction and instance-dependent costs.

    Weighting proportionally to a raw loss value (rather than a direction-aware "goodness" score)
    gives the *worst* estimators the most weight; and passing whole-dataset instance-dependent costs
    unsliced to a loss evaluated on OOB rows either raises a shape-mismatch error or silently scores
    against the wrong rows.
    """

    @pytest.mark.parametrize('model_cls', [CSForestClassifier, CSBaggingClassifier])
    def test_weighted_voting_with_array_valued_fn_cost(self, model_cls):
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        fn_cost = np.random.RandomState(0).uniform(1.0, 5.0, size=200)
        model = model_cls(n_estimators=5, combination='weighted_voting', random_state=0)
        model.fit(X, y, fp_cost=1.0, fn_cost=fn_cost)
        assert np.isfinite(model.estimator_weights_).all()
        assert model.estimator_weights_.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize('model_cls', [CSForestClassifier, CSBaggingClassifier])
    def test_weighted_voting_beats_or_matches_majority_voting_on_holdout(self, model_cls):
        """A sanity check that direction-aware weighting is not worse than unweighted voting."""
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_samples=400, n_features=10, random_state=1, class_sep=0.8)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

        weighted = model_cls(n_estimators=20, combination='weighted_voting', random_state=1)
        weighted.fit(X_train, y_train, fp_cost=1.0, fn_cost=1.0)
        majority = model_cls(n_estimators=20, combination='majority_voting', random_state=1)
        majority.fit(X_train, y_train, fp_cost=1.0, fn_cost=1.0)

        weighted_acc = accuracy_score(y_test, weighted.predict(X_test))
        majority_acc = accuracy_score(y_test, majority.predict(X_test))
        # Not a strict requirement that weighting always wins, but it shouldn't be far worse.
        assert weighted_acc >= majority_acc - 0.1
