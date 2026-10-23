"""``CSForestClassifier``'s default ``min_impurity_decrease=None`` resolves its threshold like ``CSTreeClassifier``."""

import pytest

from empulse.models import CSForestClassifier, CSTreeClassifier


@pytest.fixture
def imbalanced_costs(make_data, seeded_rng):
    X, y = make_data(n_samples=500, n_features=10, weights=[0.9], flip_y=0.1)
    fn_cost = seeded_rng.uniform(2, 20, y.size)
    return X, y, fn_cost


def test_none_resolves_to_the_trees_threshold(imbalanced_costs):
    X, y, fn_cost = imbalanced_costs
    forest = CSForestClassifier(n_estimators=5, min_impurity_decrease=None, random_state=0)
    forest.fit(X, y, fp_cost=1.0, fn_cost=fn_cost)
    tree = CSTreeClassifier(min_impurity_decrease=None, random_state=0).fit(X, y, fp_cost=1.0, fn_cost=fn_cost)

    assert tree.min_impurity_decrease_ > 0
    assert all(estimator.min_impurity_decrease_ == tree.min_impurity_decrease_ for estimator in forest.estimators_)
    assert all(estimator.min_impurity_decrease is None for estimator in forest.estimators_)


def test_none_skips_splits_that_leave_the_cost_unchanged(imbalanced_costs):
    X, y, fn_cost = imbalanced_costs
    forests = {
        value: CSForestClassifier(n_estimators=5, min_impurity_decrease=value, random_state=0).fit(
            X, y, fp_cost=1.0, fn_cost=fn_cost
        )
        for value in (None, 0.0)
    }

    def n_leaves(forest):
        return sum(estimator.get_n_leaves() for estimator in forest.estimators_)

    assert n_leaves(forests[None]) < n_leaves(forests[0.0])


@pytest.mark.parametrize('criterion', ['gini', 'entropy'])
def test_none_is_zero_for_other_criteria(imbalanced_costs, criterion):
    X, y, fn_cost = imbalanced_costs
    forest = CSForestClassifier(n_estimators=3, criterion=criterion, min_impurity_decrease=None, random_state=0)
    forest.fit(X, y, fp_cost=1.0, fn_cost=fn_cost)

    assert all(estimator.min_impurity_decrease_ == 0.0 for estimator in forest.estimators_)


def test_none_is_the_default():
    assert CSForestClassifier().min_impurity_decrease is None
