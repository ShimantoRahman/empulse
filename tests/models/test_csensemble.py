"""CSForestClassifier and CSBaggingClassifier, and the out-of-bag weighting helpers they share."""

import threading

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from empulse.models import CSBaggingClassifier, CSForestClassifier, CSTreeClassifier
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
    assert len(model.estimators_) == 5
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
def test_csforest_combination(classification_data, combination):
    X, y = classification_data
    model = CSForestClassifier(n_estimators=5, max_depth=3, combination=combination)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert len(model.estimators_) == 5
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
    """Weighted-voting predict with per-estimator feature subsets.

    `BaggingClassifier` draws a feature subset per estimator when `max_features < 1.0` or
    `bootstrap_features=True`; `_predict_weighted_proba` must slice `X` to each estimator's own
    `estimators_features_` before calling `predict_proba`, exactly like `_get_oob_weights`, or a
    sub-estimator raises "X has N features, but ... is expecting M features".
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
    """An integer `max_samples` gives finite, normalized OOB weights for CSForestClassifier.

    `isinstance(x, Real)` is also `True` for every `int`, so the bootstrap sample size must be chosen
    with `if/elif/else`: three independent `if`s would let an integer `max_samples` fall through the
    `Integral` branch and then be overwritten by the `Real` branch.
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
    """OOB weighting respects metric direction and instance-dependent costs.

    Weights must follow a direction-aware "goodness" score: proportional to a raw loss value, the
    *worst* estimators would get the most weight. Whole-dataset instance-dependent costs must be
    sliced to the OOB rows before a loss is evaluated on them; unsliced, they either raise a
    shape-mismatch error or silently score against the wrong rows.
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
    """Parameter validation rejects a ``BaseMetric`` as ``criterion``.

    ``fit`` cannot use one and would fail later with a confusing "Unknown criterion" error. The
    criterion is validated like :class:`~empulse.models.CSTreeClassifier`'s.
    """
    from sklearn.utils._param_validation import InvalidParameterError

    from empulse.metrics import expected_cost_loss

    X, y = classification_data
    with pytest.raises(InvalidParameterError, match="The 'criterion' parameter"):
        CSForestClassifier(criterion=expected_cost_loss, n_estimators=2).fit(X, y, fn_cost=5.0, fp_cost=1.0)


class TestCSForestWarmStart:
    """``warm_start`` keeps the trees already grown instead of rebuilding the inner forest on every fit."""

    def test_adding_trees_keeps_the_old_ones(self, make_data):
        X, y = make_data(n_samples=300)
        model = CSForestClassifier(n_estimators=4, warm_start=True, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        first = list(model.estimators_)
        model.set_params(n_estimators=7).fit(X, y, fp_cost=1.0, fn_cost=5.0)

        assert len(model.estimators_) == 7
        assert all(a is b for a, b in zip(first, model.estimators_[:4], strict=False))

    def test_matches_growing_all_trees_at_once(self, make_data):
        X, y = make_data(n_samples=300)
        warm = CSForestClassifier(n_estimators=4, warm_start=True, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        warm.set_params(n_estimators=7).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        cold = CSForestClassifier(n_estimators=7, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)

        np.testing.assert_array_equal(warm.predict_proba(X), cold.predict_proba(X))

    def test_fewer_trees_raises(self, make_data):
        X, y = make_data(n_samples=100)
        model = CSForestClassifier(n_estimators=4, warm_start=True, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        with pytest.raises(ValueError, match='must be larger or equal'):
            model.set_params(n_estimators=2).fit(X, y, fp_cost=1.0, fn_cost=5.0)


class TestCSForestOutOfBag:
    def test_oob_decision_function_averages_the_trees_that_left_each_sample_out(self, make_data):
        X, y = make_data(n_samples=300)
        model = CSForestClassifier(n_estimators=15, oob_score=True, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)

        total = np.zeros((y.size, 2))
        count = np.zeros(y.size)
        for tree, drawn in zip(model.estimators_, model.estimators_samples_, strict=True):
            out = np.bincount(drawn, minlength=y.size) == 0
            total[out] += tree.predict_proba(X[out])
            count[out] += 1
        expected = total / np.maximum(count, 1)[:, None]

        np.testing.assert_allclose(model.oob_decision_function_, expected)
        assert model.oob_score_ == pytest.approx(np.mean(expected.argmax(axis=1) == y))

    def test_callable_oob_score(self, make_data):
        from sklearn.metrics import balanced_accuracy_score

        X, y = make_data(n_samples=300)
        model = CSForestClassifier(n_estimators=25, oob_score=balanced_accuracy_score, random_state=0)
        model.fit(X, y, fp_cost=1.0, fn_cost=5.0)
        assert model.oob_score_ == pytest.approx(
            balanced_accuracy_score(y, model.oob_decision_function_.argmax(axis=1))
        )

    def test_oob_requires_bootstrap(self, make_data):
        X, y = make_data(n_samples=100)
        with pytest.raises(ValueError, match='Out of bag'):
            CSForestClassifier(bootstrap=False, oob_score=True).fit(X, y, fp_cost=1.0, fn_cost=5.0)


def test_csforest_balanced_subsample_weighs_each_bootstrap_sample(make_data):
    X, y = make_data(n_samples=400, weights=[0.8])
    model = CSForestClassifier(n_estimators=3, class_weight='balanced_subsample', max_depth=2, random_state=0)
    model.fit(X, y, fp_cost=1.0, fn_cost=1.0)
    # Balanced classes weigh the same at the root of every tree.
    for tree in model.estimators_:
        np.testing.assert_allclose(tree.tree_.value[0, 0], [0.5, 0.5])


# --- The out-of-bag weighting helpers ------------------------------------------------------------


class TestGoodnessWeights:
    """OOB weighting gives more weight to better estimators.

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
    """Instance-dependent loss params are subset to the OOB rows."""

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


class TestPredictVotesCheapestClasses:
    """Each tree votes for the class that costs least on the samples it drew into the leaf."""

    @staticmethod
    def _data(seed=0):
        X, y = make_classification(n_samples=800, n_features=6, weights=[0.85], flip_y=0.05, random_state=seed)
        rng = np.random.default_rng(seed)
        return X, y, rng.uniform(0.5, 2, y.size), rng.uniform(5, 30, y.size)

    @staticmethod
    def _tree_votes(leaves, y, counts, fp_cost, fn_cost):
        """Reference: per leaf, sum each decision's cost over the drawn samples, one sample at a time."""
        votes = np.zeros(leaves.size)
        for leaf in np.unique(leaves):
            in_leaf = leaves == leaf
            w, positive = counts[in_leaf], y[in_leaf] == 1
            cost_positive = np.sum(w * np.where(positive, 0.0, fp_cost[in_leaf]))
            cost_negative = np.sum(w * np.where(positive, fn_cost[in_leaf], 0.0))
            votes[in_leaf] = cost_positive < cost_negative
        return votes

    @pytest.mark.parametrize('bootstrap', [True, False])
    def test_forest_majority_vote_of_the_cheapest_classes(self, bootstrap):
        X, y, fp_cost, fn_cost = self._data()
        model = CSForestClassifier(n_estimators=9, max_depth=4, bootstrap=bootstrap, random_state=0)
        model.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        leaves = model.apply(X)
        votes = sum(
            self._tree_votes(leaves[:, i], y, np.bincount(drawn, minlength=y.size), fp_cost, fn_cost)
            for i, drawn in enumerate(model.estimators_samples_)
        )
        np.testing.assert_array_equal(model.predict(X), (votes > 9 / 2).astype(int))

    def test_forest_weighted_vote_uses_the_out_of_bag_weights(self):
        X, y, fp_cost, fn_cost = self._data()
        model = CSForestClassifier(n_estimators=9, max_depth=4, combination='weighted_voting', random_state=0)
        model.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        leaves = model.apply(X)
        votes = sum(
            weight * self._tree_votes(leaves[:, i], y, np.bincount(drawn, minlength=y.size), fp_cost, fn_cost)
            for i, (drawn, weight) in enumerate(zip(model.estimators_samples_, model.estimator_weights_, strict=True))
        )
        np.testing.assert_array_equal(model.predict(X), (votes > model.estimator_weights_.sum() / 2).astype(int))

    def test_forest_decisions_cost_less_than_the_majority(self):
        X, y, fp_cost, fn_cost = self._data()
        model = CSForestClassifier(n_estimators=25, max_depth=6, random_state=0).fit(
            X, y, fp_cost=fp_cost, fn_cost=fn_cost
        )

        def training_cost(y_pred):
            return np.sum(np.where(y == 1, np.where(y_pred == 1, 0, fn_cost), np.where(y_pred == 1, fp_cost, 0)))

        assert training_cost(model.predict(X)) < training_cost(model.predict_proba(X).argmax(axis=1))

    @pytest.mark.parametrize('combination', ['majority_voting', 'weighted_voting'])
    def test_bagging_votes_its_trees_decisions(self, combination):
        X, y, fp_cost, fn_cost = self._data()
        model = CSBaggingClassifier(n_estimators=9, combination=combination, random_state=0)
        model.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        weights = model.estimator_weights_ if combination == 'weighted_voting' else np.ones(9)
        votes = sum(
            weight * tree.predict(X[:, features])
            for tree, features, weight in zip(
                model.estimator_.estimators_, model.estimator_.estimators_features_, weights, strict=True
            )
        )
        np.testing.assert_array_equal(model.predict(X), (votes > weights.sum() / 2).astype(int))

    def test_bagging_keeps_fully_grown_trees(self):
        X, y, fp_cost, fn_cost = self._data()
        model = CSBaggingClassifier(n_estimators=3, random_state=0).fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        assert all(tree.min_impurity_decrease_ == 0.0 for tree in model.estimator_.estimators_)

    def test_bagging_an_explicit_cost_sensitive_tree_votes_like_the_default(self):
        X, y, fp_cost, fn_cost = self._data()
        default = CSBaggingClassifier(n_estimators=5, random_state=0).fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)
        explicit = CSBaggingClassifier(CSTreeClassifier(min_impurity_decrease=0.0), n_estimators=5, random_state=0)
        explicit.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        np.testing.assert_array_equal(explicit.predict(X), default.predict(X))

    def test_bagging_a_custom_estimator_still_averages_probabilities(self):
        from empulse.models import CSLogitClassifier

        X, y, fp_cost, fn_cost = self._data()
        model = CSBaggingClassifier(CSLogitClassifier(), n_estimators=5, random_state=0)
        model.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        np.testing.assert_array_equal(model.predict(X), model.predict_proba(X).argmax(axis=1))
