import numpy as np
import pytest
import sympy
import sympy.stats

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.models import CSTreeClassifier


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
def test_cstree_criteria(classification_data, criterion):
    X, y = classification_data
    model = CSTreeClassifier(criterion=criterion)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'tree_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


def test_criterion_must_be_a_name(classification_data):
    """Only the named criteria are accepted; there is no criterion object to pass in."""
    from sklearn.utils._param_validation import InvalidParameterError

    from empulse.metrics import expected_cost_loss

    X, y = classification_data
    with pytest.raises(InvalidParameterError, match="The 'criterion' parameter"):
        CSTreeClassifier(criterion=expected_cost_loss).fit(X, y, fp_cost=1.0, fn_cost=5.0)


def test_monotonic_cst_is_not_a_parameter():
    with pytest.raises(TypeError, match='monotonic_cst'):
        CSTreeClassifier(monotonic_cst=[1, 0])


def test_cstree_with_stochastic_maxprofit_metric(classification_data):
    """Regression test: a MaxProfit metric with a stochastic variable used to raise, not reduce to its mean.

    `_fit` used to call `self.loss._evaluate_costs(**loss_params)` without `replace_stochastic=True`,
    so a `sympy.stats` random variable in the cost/benefit expression raised deep inside
    `_evaluate_expression` instead of being reduced to its mean - the same reduction
    `_prepare_class_costs`/`ProfTreeClassifier` already apply for exactly this situation.
    """
    X, y = classification_data
    clv = sympy.stats.Beta('clv', 2, 5)
    contact_cost = sympy.symbols('contact_cost')
    metric = Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(contact_cost), MaxProfit())

    model = CSTreeClassifier(loss=metric).fit(X, y, contact_cost=1.0)
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (100, 2)


class TestInspection:
    """The introspection API of scikit-learn's DecisionTreeClassifier, served by the fitted ``tree_``."""

    @pytest.fixture
    def model(self, classification_data):
        X, y = classification_data
        return CSTreeClassifier(max_depth=3, random_state=42).fit(X, y, fp_cost=1.0, fn_cost=1.0)

    def test_feature_importances(self, model, classification_data):
        X, _ = classification_data
        importances = model.feature_importances_
        assert importances.shape == (X.shape[1],)
        assert np.isclose(importances.sum(), 1.0)
        np.testing.assert_array_equal(importances, model.tree_.compute_feature_importances())

    def test_feature_importances_are_the_weighted_impurity_decreases(self, model):
        tree = model.tree_
        internal = np.flatnonzero(tree.children_left != -1)
        weighted = tree.weighted_n_node_samples * tree.impurity
        decrease = weighted[internal] - weighted[tree.children_left[internal]] - weighted[tree.children_right[internal]]
        expected = np.bincount(tree.feature[internal], weights=decrease, minlength=model.n_features_in_)
        np.testing.assert_allclose(model.feature_importances_, expected / expected.sum())

    def test_max_features(self, model):
        assert model.max_features_ == model.n_features_in_

    def test_n_classes(self, model):
        assert model.n_classes_ == 2

    def test_n_outputs(self, model):
        assert model.n_outputs_ == 1

    def test_tree_(self, model):
        from empulse.models.tree._cstree import CostTree

        assert isinstance(model.tree_, CostTree)

    def test_get_depth(self, model):
        assert model.get_depth() == model.tree_.max_depth
        assert model.get_depth() <= 3

    def test_get_n_leaves(self, model):
        assert model.get_n_leaves() == np.sum(model.tree_.children_left == -1)
        assert model.get_n_leaves() > 0

    def test_apply(self, model, classification_data):
        X, _ = classification_data
        leaves = model.apply(X)
        assert leaves.shape == (X.shape[0],)
        assert np.all(model.tree_.children_left[leaves] == -1)

    def test_cost_complexity_pruning_path(self, model, classification_data):
        X, y = classification_data
        path = model.cost_complexity_pruning_path(X, y, fp_cost=1.0, fn_cost=1.0)
        assert 'ccp_alphas' in path
        assert 'impurities' in path
        assert len(path.ccp_alphas) == len(path.impurities)

    def test_decision_path(self, model, classification_data):
        X, _ = classification_data
        indicator = model.decision_path(X).toarray()
        assert indicator.shape == (X.shape[0], model.tree_.node_count)
        # Every path runs from the root to the leaf the sample lands in.
        assert np.all(indicator[:, 0] == 1)
        np.testing.assert_array_equal(indicator[np.arange(X.shape[0]), model.apply(X)], 1)
        assert np.all(indicator.sum(axis=1) <= model.get_depth() + 1)

    def test_not_fitted_raises(self, classification_data):
        from sklearn.exceptions import NotFittedError

        model = CSTreeClassifier()
        with pytest.raises(NotFittedError):
            _ = model.feature_importances_
        with pytest.raises(NotFittedError):
            model.get_depth()
        with pytest.raises(NotFittedError):
            model.get_n_leaves()
        with pytest.raises(NotFittedError):
            model.predict(classification_data[0])


@pytest.mark.parametrize(
    'labels',
    [np.array(['no', 'yes']), np.array([-1, 1]), np.array([2, 5])],
    ids=['strings', 'minus_one_one', 'two_five'],
)
def test_predict_returns_original_labels(classification_data, labels):
    """Regression test: ``predict`` returned the inner tree's 0/1-encoded classes.

    ``CostSensitiveClassifier.fit`` encodes the target as 0/1 before fitting the inner tree, so its
    predictions have to be mapped back through ``classes_``.
    """
    X, y = classification_data
    y_labels = labels[y]
    model = CSTreeClassifier(max_depth=3, random_state=0).fit(X, y_labels, fn_cost=5.0, fp_cost=1.0)

    y_pred = model.predict(X)

    np.testing.assert_array_equal(model.classes_, labels)
    np.testing.assert_array_equal(y_pred, labels[np.argmax(model.predict_proba(X), axis=1)])


def _split_gains(model):
    """The decrease of the weighted impurity at every internal node of a fitted tree."""
    tree = model.tree_
    internal = np.flatnonzero(tree.children_left != -1)
    left, right = tree.children_left[internal], tree.children_right[internal]
    weight = tree.weighted_n_node_samples
    return (
        weight[internal] * tree.impurity[internal]
        - weight[left] * tree.impurity[left]
        - weight[right] * tree.impurity[right]
    )


class TestSplitsMustLowerTheCost:
    """With the cost criterion, a split that leaves the training cost unchanged is not made by default.

    The cost impurity is the cost of the node's best decision, so a split whose children both keep that
    decision has no gain; making such splits anyway grew trees until every leaf held one class.
    """

    @staticmethod
    def _fit(make_data, seeded_rng, **params):
        X, y = make_data(n_samples=2000, n_features=10, weights=[0.9], flip_y=0.1)
        fn_cost = seeded_rng.uniform(2, 20, y.size)
        return CSTreeClassifier(random_state=0, **params).fit(X, y, fp_cost=1.0, fn_cost=fn_cost)

    def test_every_split_lowers_the_cost_by_default(self, make_data, seeded_rng):
        model = self._fit(make_data, seeded_rng)
        cost_scale = model.tree_.impurity[0] * model.tree_.weighted_n_node_samples[0]

        assert np.all(_split_gains(model) > 1e-9 * cost_scale)

    def test_zero_restores_splits_that_leave_the_cost_unchanged(self, make_data, seeded_rng):
        default = self._fit(make_data, seeded_rng)
        unrestricted = self._fit(make_data, seeded_rng, min_impurity_decrease=0.0)
        cost_scale = unrestricted.tree_.impurity[0] * unrestricted.tree_.weighted_n_node_samples[0]

        assert np.any(np.abs(_split_gains(unrestricted)) <= 1e-9 * cost_scale)
        assert default.get_depth() < unrestricted.get_depth()
        assert default.get_n_leaves() < unrestricted.get_n_leaves()

    def test_threshold_follows_the_scale_of_the_costs(self, make_data, seeded_rng):
        cheap = self._fit(make_data, np.random.default_rng(0))
        X, y = make_data(n_samples=2000, n_features=10, weights=[0.9], flip_y=0.1)
        fn_cost = np.random.default_rng(0).uniform(2, 20, y.size)
        expensive = CSTreeClassifier(random_state=0).fit(X, y, fp_cost=1000.0, fn_cost=1000.0 * fn_cost)

        assert expensive.min_impurity_decrease_ == pytest.approx(1000 * cheap.min_impurity_decrease_)
        # Scaling every cost by the same factor does not change which splits lower the cost.
        assert expensive.get_depth() == cheap.get_depth()
        assert expensive.get_n_leaves() == cheap.get_n_leaves()

    @pytest.mark.parametrize('criterion', ['gini', 'entropy'])
    def test_other_criteria_keep_scikit_learns_default(self, make_data, seeded_rng, criterion):
        model = self._fit(make_data, seeded_rng, criterion=criterion)
        assert model.min_impurity_decrease_ == 0.0

    def test_an_explicit_value_is_used_as_is(self, make_data, seeded_rng):
        model = self._fit(make_data, seeded_rng, min_impurity_decrease=0.01)
        assert model.min_impurity_decrease_ == 0.01


def _cheapest_class_per_leaf(leaves, y, weights, tp, tn, fp, fn):
    """Reference: sum each decision's cost over every training sample in each leaf, one sample at a time."""
    decisions = {}
    for leaf in np.unique(leaves):
        in_leaf = leaves == leaf
        w, positive = weights[in_leaf], y[in_leaf] == 1
        cost_positive = np.sum(w * np.where(positive, tp[in_leaf], fp[in_leaf]))
        cost_negative = np.sum(w * np.where(positive, fn[in_leaf], tn[in_leaf]))
        if np.isclose(cost_positive, cost_negative):
            decisions[leaf] = int(np.sum(w[positive]) > np.sum(w[~positive]))
        else:
            decisions[leaf] = int(cost_positive < cost_negative)
    return decisions


class TestPredictDecidesByCost:
    """Each leaf predicts the class that costs least on its training samples, not its majority class."""

    @staticmethod
    def _data(make_data, seeded_rng):
        X, y = make_data(n_samples=1500, n_features=6, weights=[0.85], flip_y=0.05)
        fn_cost = seeded_rng.uniform(5, 30, y.size)
        fp_cost = seeded_rng.uniform(0.5, 2, y.size)
        return X, y, fp_cost, fn_cost

    def test_each_leaf_predicts_its_cheapest_class(self, make_data, seeded_rng):
        X, y, fp_cost, fn_cost = self._data(make_data, seeded_rng)
        model = CSTreeClassifier(max_depth=5, random_state=0).fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        zeros = np.zeros(y.size)
        leaves = model.apply(X)
        expected = _cheapest_class_per_leaf(leaves, y, np.ones(y.size), zeros, zeros, fp_cost, fn_cost)
        np.testing.assert_array_equal(model.predict(X), [expected[leaf] for leaf in leaves])

    def test_predicts_the_costly_class_where_it_is_the_minority(self, make_data, seeded_rng):
        X, y, fp_cost, fn_cost = self._data(make_data, seeded_rng)
        model = CSTreeClassifier(max_depth=5, random_state=0).fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        majority = model.predict_proba(X).argmax(axis=1)
        # False negatives cost far more, so some leaves with a negative majority predict positive.
        assert np.any((model.predict(X) == 1) & (majority == 0))

        # And the decisions cost less on the training data than the majority's.
        def training_cost(y_pred):
            return np.sum(np.where(y == 1, np.where(y_pred == 1, 0, fn_cost), np.where(y_pred == 1, fp_cost, 0)))

        assert training_cost(model.predict(X)) < training_cost(majority)

    def test_decisions_follow_the_class_weights_the_tree_was_grown_with(self, make_data, seeded_rng):
        from sklearn.utils.class_weight import compute_sample_weight

        X, y, fp_cost, fn_cost = self._data(make_data, seeded_rng)
        class_weight = {0: 3.0, 1: 0.5}
        model = CSTreeClassifier(max_depth=5, random_state=0, class_weight=class_weight).fit(
            X, y, fp_cost=fp_cost, fn_cost=fn_cost
        )

        zeros = np.zeros(y.size)
        leaves = model.apply(X)
        expected = _cheapest_class_per_leaf(
            leaves, y, compute_sample_weight(class_weight, y), zeros, zeros, fp_cost, fn_cost
        )
        np.testing.assert_array_equal(model.predict(X), [expected[leaf] for leaf in leaves])

    def test_leaves_whose_classes_cost_the_same_predict_their_majority(self, make_data):
        X, y = make_data(n_samples=500, n_features=4)
        # Every outcome costs the same whatever is predicted, so the costs prefer neither class.
        model = CSTreeClassifier(max_depth=4, random_state=0).fit(X, y, tp_cost=2, fn_cost=2, fp_cost=3, tn_cost=3)

        np.testing.assert_array_equal(model.predict(X), model.predict_proba(X).argmax(axis=1))

    def test_predicts_the_original_labels(self, make_data, seeded_rng):
        X, y, fp_cost, fn_cost = self._data(make_data, seeded_rng)
        labels = np.where(y == 1, 'yes', 'no')
        model = CSTreeClassifier(max_depth=5, random_state=0).fit(X, labels, fp_cost=fp_cost, fn_cost=fn_cost)
        encoded = CSTreeClassifier(max_depth=5, random_state=0).fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        np.testing.assert_array_equal(model.predict(X), np.where(encoded.predict(X) == 1, 'yes', 'no'))
