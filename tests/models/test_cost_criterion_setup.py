import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from empulse.models.cost_sensitive._impurity import (
    CostImpurity,
    EntropyCostImpurity,
    GiniCostImpurity,
    build_cost_criterion,
)

# CostImpurity's cost state lives in C-level (cdef, non-public) attributes with no Python getters,
# so these tests verify behavior through the public API (type dispatch, error handling, and
# actually fitting a tree) rather than inspecting internal state directly.


@pytest.mark.parametrize(
    'criterion_name, expected_type',
    [
        ('cost', CostImpurity),
        ('gini', GiniCostImpurity),
        ('entropy', EntropyCostImpurity),
        ('log_loss', EntropyCostImpurity),
    ],
)
def test_build_cost_criterion_dispatches_on_string(criterion_name, expected_type):
    criterion = build_cost_criterion(criterion_name, tp_cost=1.0, tn_cost=0.0, fn_cost=5.0, fp_cost=1.0, n_samples=10)
    assert type(criterion) is expected_type


def test_unknown_criterion_raises():
    with pytest.raises(ValueError, match='Unknown criterion'):
        build_cost_criterion('not-a-criterion', tp_cost=1.0, tn_cost=0.0, fn_cost=5.0, fp_cost=1.0, n_samples=10)


def test_mismatched_array_shape_raises():
    with pytest.raises(ValueError, match='fn_cost has shape'):
        build_cost_criterion('cost', tp_cost=1.0, tn_cost=0.0, fn_cost=np.ones(5), fp_cost=1.0, n_samples=10)


def test_custom_criterion_instance_gets_a_fresh_copy_not_reused():
    """Passing a CostImpurity instance must not reuse (or mutate) that exact object."""
    original = CostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    criterion = build_cost_criterion(original, tp_cost=1.0, tn_cost=0.0, fn_cost=5.0, fp_cost=1.0, n_samples=10)
    assert criterion is not original
    assert type(criterion) is CostImpurity


def test_custom_criterion_subclass_preserves_type():
    original = GiniCostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
    criterion = build_cost_criterion(original, tp_cost=1.0, tn_cost=0.0, fn_cost=5.0, fp_cost=1.0, n_samples=10)
    assert type(criterion) is GiniCostImpurity


def test_criterion_usable_by_decision_tree_classifier_with_negative_costs():
    """A negative cost (e.g. tp_cost=-200, a benefit) must be offset without crashing or erroring.

    sklearn's tree builder requires node_impurity() >= 0; build_cost_criterion offsets all costs
    by the negated minimum to guarantee this, exactly reproducing what cstree.py/csforest.py did
    inline before being deduplicated into this helper.
    """
    X = np.array([[0.0], [0.0], [1.0], [1.0]] * 5)
    y = np.array([0, 0, 1, 1] * 5)
    criterion = build_cost_criterion('cost', tp_cost=-200.0, tn_cost=0.0, fn_cost=5.0, fp_cost=1.0, n_samples=len(y))
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    y_pred = tree.predict(X)
    assert y_pred.shape == y.shape


def test_criterion_usable_with_array_valued_costs():
    X = np.array([[0.0], [0.0], [1.0], [1.0]] * 5)
    y = np.array([0, 0, 1, 1] * 5)
    fn_cost = np.random.default_rng(0).uniform(1.0, 5.0, size=len(y))
    criterion = build_cost_criterion('cost', tp_cost=0.0, tn_cost=0.0, fn_cost=fn_cost, fp_cost=1.0, n_samples=len(y))
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    y_pred = tree.predict(X)
    assert y_pred.shape == y.shape
