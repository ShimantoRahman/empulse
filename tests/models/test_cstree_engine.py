"""The compiled tree engine behind CSTreeClassifier and CSForestClassifier (``empulse.models.tree._cstree``)."""

import pickle
from itertools import pairwise

import numpy as np
import pytest

from empulse.models import CSTreeClassifier
from empulse.models.tree._cstree import CostTree, Splitter, build_tree, cost_records, criterion_kind


def _weighted_impurity(criterion, pos_cost, neg_cost, pos_weight, neg_weight):
    """``weight * impurity`` of a node, from its definition (Correa Bahnsen et al., 2015)."""
    weight = pos_weight + neg_weight
    if criterion == 'cost':
        return min(pos_cost, neg_cost)
    if criterion == 'gini':
        return min(pos_cost * pos_weight**2 / weight, neg_cost * neg_weight**2 / weight)
    pos_entropy = np.log2(pos_weight / weight) if pos_weight > 0 else 0.0
    neg_entropy = np.log2(neg_weight / weight) if neg_weight > 0 else 0.0
    return min(-pos_cost * pos_entropy, -neg_cost * neg_entropy)


def _best_root_split(X, y, a, b, criterion):
    """Reference: try every threshold of every feature, one at a time, and keep the best."""
    X = X.astype(np.float32)
    best = (np.inf, None, None)
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for low, high in pairwise(values):
            threshold = np.float64(low) / 2.0 + np.float64(high) / 2.0
            child_impurity = 0.0
            for side in (X[:, feature] <= threshold, X[:, feature] > threshold):
                child_impurity += _weighted_impurity(
                    criterion, a[side].sum(), b[side].sum(), np.sum(y[side] == 1), np.sum(y[side] == 0)
                )
            if child_impurity < best[0]:
                best = (child_impurity, feature, threshold)
    return best[1], best[2]


class TestBestSplit:
    """The root split is the one that lowers the impurity most, by brute force."""

    @pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
    # Nodes of at least 512 samples are radix sorted, smaller ones by introsort.
    @pytest.mark.parametrize('n_samples', [150, 1000], ids=['introsort', 'radix_sort'])
    def test_root_split_matches_brute_force(self, make_data, seeded_rng, criterion, n_samples):
        X, y = make_data(n_samples=n_samples, n_features=4, n_informative=3, n_redundant=0, flip_y=0.2)
        fp_cost = seeded_rng.uniform(0.5, 2, y.size)
        fn_cost = seeded_rng.uniform(1, 10, y.size)
        model = CSTreeClassifier(criterion=criterion, max_depth=1, min_impurity_decrease=0.0, random_state=0)
        model.fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost)

        a = np.where(y == 1, 0.0, fp_cost)  # the cost of predicting each sample positive
        b = np.where(y == 1, fn_cost, 0.0)  # and negative
        feature, threshold = _best_root_split(X, y, a, b, criterion)
        assert model.tree_.feature[0] == feature
        assert model.tree_.threshold[0] == threshold


class TestCostRecords:
    def test_layout(self):
        y = np.array([0, 1, 1])
        records = cost_records(y, tp_cost=0.5, tn_cost=0.25, fn_cost=np.array([1.0, 2.0, 3.0]), fp_cost=4.0)
        # Per sample: cost if predicted positive, cost if predicted negative, the class, padding.
        np.testing.assert_array_equal(records, [[4.0, 0.25, 0, 0], [0.5, 2.0, 1, 0], [0.5, 3.0, 1, 0]])

    def test_negative_costs_are_shifted_to_zero(self):
        y = np.array([0, 1])
        records = cost_records(y, tp_cost=-3.0, tn_cost=0.0, fn_cost=4.0, fp_cost=1.0)
        np.testing.assert_array_equal(records[:, :2], [[4.0, 3.0], [0.0, 7.0]])

    def test_mismatched_array_shape_raises(self):
        with pytest.raises(ValueError, match='fn_cost has shape'):
            cost_records(np.zeros(10), tp_cost=1.0, tn_cost=0.0, fn_cost=np.ones(5), fp_cost=1.0)

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match='Unknown criterion'):
            criterion_kind('not-a-criterion')


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
@pytest.mark.parametrize('instance_dependent', [False, True], ids=['class_dependent', 'instance_dependent'])
def test_sample_weights_act_like_repeated_rows(make_data, seeded_rng, criterion, instance_dependent):
    """
    A tree fitted with integer weights matches one fitted on the rows repeated that many times.

    Forests pass their bootstrap draws as weights. Node by node both trees see the same costs; two
    split candidates can still tie exactly and be broken differently, as the sums run in a different
    order, so predictions are compared loosely.
    """
    X, y = make_data(n_samples=300, flip_y=0.2)
    counts = np.bincount(seeded_rng.integers(0, 300, 300), minlength=300).astype(np.float64)
    fn_cost = seeded_rng.random(300) * 10 if instance_dependent else 5.0
    repeated = np.repeat(np.arange(300), counts.astype(int))

    def grow(X, y, fn_cost, weight):
        records = cost_records(y, tp_cost=0.0, tn_cost=0.0, fn_cost=fn_cost, fp_cost=1.0)
        X = np.asfortranarray(X, dtype=np.float32)
        splitter = Splitter(X, records, weight, criterion_kind(criterion), False, X.shape[1], 1, 0.0, 0, False)
        tree = CostTree(X.shape[1])
        build_tree(tree, splitter, 2, 1, 0.0, 4, -1, 0.0, False)
        return tree

    weighted = grow(X, y, fn_cost, counts)
    materialised = grow(X[repeated], y[repeated], fn_cost[repeated] if instance_dependent else fn_cost, None)

    np.testing.assert_allclose(weighted.impurity, materialised.impurity)
    x32 = np.asarray(X, dtype=np.float32)
    assert np.mean(weighted.predict_positive(x32) == materialised.predict_positive(x32)) > 0.95


@pytest.mark.parametrize('criterion', ['entropy', 'log_loss'])
def test_entropy_criterion_grows_beyond_a_stump(make_data, criterion):
    """Negative child impurities must not turn every entropy tree into a stump."""
    X, y = make_data(n_samples=500)
    tree = CSTreeClassifier(criterion=criterion, max_depth=6, random_state=0).fit(X, y, fn_cost=5.0, fp_cost=1.0)

    assert tree.get_depth() > 1
    assert np.all(tree.tree_.impurity >= 0)


def test_cost_bound_only_skips_nodes_no_split_improves(make_data, seeded_rng):
    """Nodes whose samples all prefer the same decision become leaves without a split search."""
    X, y = make_data(n_samples=2000, n_features=6, flip_y=0.1)
    # Instance-dependent costs, under which many impure nodes are already decided.
    fn_cost = seeded_rng.uniform(0.1, 3, y.size)
    fp_cost = seeded_rng.uniform(0.1, 3, y.size)
    min_impurity_decrease = CSTreeClassifier().fit(X, y, fp_cost=fp_cost, fn_cost=fn_cost).min_impurity_decrease_
    assert min_impurity_decrease > 0
    records = cost_records(y, tp_cost=0.0, tn_cost=0.0, fn_cost=fn_cost, fp_cost=fp_cost)
    x_fortran = np.asfortranarray(X, dtype=np.float32)

    def grow(cost_bound):
        splitter = Splitter(x_fortran, records, None, criterion_kind('cost'), False, X.shape[1], 1, 0.0, 0, cost_bound)
        tree = CostTree(X.shape[1])
        build_tree(tree, splitter, 2, 1, 0.0, 2**31 - 1, -1, min_impurity_decrease, cost_bound)
        return tree

    bounded, searched = grow(True), grow(False)
    np.testing.assert_array_equal(bounded.feature, searched.feature)
    np.testing.assert_array_equal(bounded.threshold, searched.threshold)
    np.testing.assert_array_equal(bounded.positive, searched.positive)


class TestCostTree:
    @pytest.fixture
    def model(self, make_data):
        X, y = make_data(n_samples=400, n_features=5)
        return CSTreeClassifier(max_depth=4, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0), X

    def test_pickle_round_trip(self, model):
        model, X = model
        restored = pickle.loads(pickle.dumps(model.tree_))
        for name in ('children_left', 'children_right', 'feature', 'threshold', 'impurity', 'value', 'positive'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(model.tree_, name))
        assert restored.max_depth == model.tree_.max_depth
        x32 = np.asarray(X, dtype=np.float32)
        np.testing.assert_array_equal(restored.apply(x32), model.tree_.apply(x32))

    def test_scikit_learn_attributes(self, model):
        model, _ = model
        tree = model.tree_
        n = tree.node_count
        assert tree.n_outputs == 1
        np.testing.assert_array_equal(tree.n_classes, [2])
        assert tree.value.shape == (n, 1, 2)
        np.testing.assert_allclose(tree.value.sum(axis=2), 1.0)
        is_leaf = tree.children_left == -1
        assert np.all(tree.children_right[is_leaf] == -1)
        assert np.all(tree.feature[is_leaf] == -2)
        assert tree.n_leaves == model.get_n_leaves() == is_leaf.sum()
        np.testing.assert_array_equal(tree.n_node_samples[0], 400)

    def test_export_graphviz_accepts_the_classifier(self, model):
        from sklearn.tree import export_graphviz

        model, _ = model
        assert export_graphviz(model).startswith('digraph Tree {')

    def test_children_partition_their_parent(self, model):
        model, _ = model
        tree = model.tree_
        internal = np.flatnonzero(tree.children_left != -1)
        children = (
            tree.n_node_samples[tree.children_left[internal]] + tree.n_node_samples[tree.children_right[internal]]
        )
        np.testing.assert_array_equal(children, tree.n_node_samples[internal])


class TestGrowth:
    def test_max_leaf_nodes(self, make_data):
        X, y = make_data(n_samples=1000)
        model = CSTreeClassifier(max_leaf_nodes=7, min_impurity_decrease=0.0, random_state=0)
        assert model.fit(X, y, fp_cost=1.0, fn_cost=5.0).get_n_leaves() == 7

    def test_random_splitter(self, make_data):
        X, y = make_data(n_samples=500)
        model = CSTreeClassifier(splitter='random', max_depth=5, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        assert 1 < model.get_n_leaves() <= 32

    def test_pruning_path(self, make_data):
        X, y = make_data(n_samples=800)
        model = CSTreeClassifier(random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        path = model.cost_complexity_pruning_path(X, y, fp_cost=1.0, fn_cost=5.0)

        assert np.all(np.diff(path.ccp_alphas) >= 0)
        assert np.all(np.diff(path.impurities) >= -1e-12)
        # With the cost criterion, the leaves' summed impurity is the training cost per sample.
        training_cost = np.sum(np.where(y == 1, np.where(model.predict(X) == 1, 0, 5.0), model.predict(X) == 1))
        assert path.impurities[0] == pytest.approx(training_cost / y.size)

    def test_ccp_alpha_prunes(self, make_data):
        X, y = make_data(n_samples=800)
        full = CSTreeClassifier(random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        pruned = CSTreeClassifier(ccp_alpha=0.005, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        assert 1 <= pruned.get_n_leaves() < full.get_n_leaves()
