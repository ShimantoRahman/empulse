"""CSForestClassifier and CSBaggingClassifier, and the out-of-bag weighting helpers they share."""

import threading

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from empulse.models import CSBaggingClassifier, CSForestClassifier
from empulse.models._base.ensemble_weighting import (
    accumulate_weighted_prediction,
    goodness_weights,
    subset_loss_params,
)


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
def test_csforest_criteria(classification_data, criterion):
    X, y = classification_data
    model = CSForestClassifier(n_estimators=5, max_depth=3, criterion=criterion)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csforest_combination(classification_data, combination):
    X, y = classification_data
    model = CSForestClassifier(n_estimators=5, max_depth=3, combination=combination)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csbagging_combination(classification_data, combination):
    X, y = classification_data
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
        fn_cost = np.random.default_rng(0).uniform(1.0, 5.0, size=200)
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


def test_csforest_rejects_a_metric_as_criterion(classification_data):
    """Regression test: parameter validation accepted a ``BaseMetric`` as ``criterion``.

    ``fit`` could not use one and failed later with a confusing "Unknown criterion" error. The
    criterion is now validated like :class:`~empulse.models.CSTreeClassifier`'s.
    """
    from sklearn.utils._param_validation import InvalidParameterError

    from empulse.metrics import expected_cost_loss

    X, y = classification_data
    with pytest.raises(InvalidParameterError, match="The 'criterion' parameter"):
        CSForestClassifier(criterion=expected_cost_loss, n_estimators=2).fit(X, y, fn_cost=5.0, fp_cost=1.0)


def test_csforest_accepts_a_cost_impurity_instance(classification_data):
    from empulse.models.tree._impurity import GiniCostImpurity

    X, y = classification_data
    criterion = GiniCostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    model = CSForestClassifier(criterion=criterion, n_estimators=2, random_state=0).fit(X, y, fn_cost=5.0, fp_cost=1.0)
    assert isinstance(model.criterion_, GiniCostImpurity)


# --- The out-of-bag weighting helpers ------------------------------------------------------------


class TestGoodnessWeights:
    """Regression tests for OOB weighting giving more weight to worse estimators.

    `goodness_weights` takes values from `BaseMetric._loss`, which is always minimized, so a lower
    value must win the higher weight. It must also not blow up when normalizing signed values.
    """

    def test_lower_loss_gets_more_weight(self):
        losses = np.array([1.0, 5.0, 0.5, 10.0])
        weights = goodness_weights(losses)
        assert weights.argmax() == 2  # lowest loss
        assert weights.argmin() == 3  # highest loss

    def test_negated_score_ranks_the_same_way(self):
        """A MAXIMIZE metric reaches this via `_loss`, i.e. negated; the best score must still win."""
        scores = np.array([1.0, 5.0, 0.5, 10.0])
        weights = goodness_weights(-scores)
        assert weights.argmax() == 3  # highest score
        assert weights.argmin() == 2  # lowest score

    def test_weights_are_non_negative_and_sum_to_one(self):
        weights = goodness_weights(np.array([-3.0, 0.5, 2.0, -1.0]))
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_signed_values_do_not_produce_negative_weights(self):
        """Normalizing signed values by their raw sum can go negative; goodness-shifting must not."""
        weights = goodness_weights(np.array([-10.0, -5.0, 3.0, 8.0]))
        assert (weights >= 0).all()

    def test_identical_values_fall_back_to_uniform(self):
        weights = goodness_weights(np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(weights, np.full(3, 1 / 3))

    def test_non_finite_values_fall_back_to_uniform(self):
        weights = goodness_weights(np.array([1.0, np.nan, 3.0]))
        np.testing.assert_allclose(weights, np.full(3, 1 / 3))


class TestSubsetLossParams:
    """Regression tests for instance-dependent loss params not being subset to OOB rows."""

    def test_array_matching_n_samples_is_subset_by_index(self):
        params = {'fn_cost': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        index = np.array([0, 2, 4])
        result = subset_loss_params(params, index, n_samples=5)
        np.testing.assert_array_equal(result['fn_cost'], [1.0, 3.0, 5.0])

    def test_array_matching_n_samples_is_subset_by_boolean_mask(self):
        params = {'fn_cost': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        mask = np.array([True, False, True, False, True])
        result = subset_loss_params(params, mask, n_samples=5)
        np.testing.assert_array_equal(result['fn_cost'], [1.0, 3.0, 5.0])

    def test_scalar_passes_through_unchanged(self):
        params = {'fn_cost': 5.0}
        result = subset_loss_params(params, np.array([0, 2]), n_samples=5)
        assert result['fn_cost'] == 5.0

    def test_array_not_matching_n_samples_passes_through_unchanged(self):
        """An array parameter unrelated to the per-sample axis (e.g. a fixed-length constant) is untouched."""
        params = {'class_priors': np.array([0.3, 0.7])}
        result = subset_loss_params(params, np.array([0, 2]), n_samples=5)
        np.testing.assert_array_equal(result['class_priors'], [0.3, 0.7])

    def test_mixed_params(self):
        params = {'fn_cost': np.arange(5, dtype=np.float64), 'fp_cost': 1.0}
        index = np.array([1, 3])
        result = subset_loss_params(params, index, n_samples=5)
        np.testing.assert_array_equal(result['fn_cost'], [1.0, 3.0])
        assert result['fp_cost'] == 1.0


def test_accumulate_weighted_prediction():
    out = np.zeros((3, 2))
    lock = threading.Lock()

    def predict(X):
        return np.full((3, 2), 2.0)

    accumulate_weighted_prediction(predict, np.zeros((3, 1)), out, weight=0.5, lock=lock)
    np.testing.assert_allclose(out, np.full((3, 2), 1.0))
