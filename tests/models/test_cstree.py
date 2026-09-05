import numpy as np
import pytest
import sympy
import sympy.stats
from sklearn.datasets import make_classification

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.models import CSTreeClassifier
from empulse.models.cost_sensitive._impurity import CostImpurity, GiniCostImpurity


@pytest.fixture
def data():
    return make_classification(n_samples=100, random_state=42)


@pytest.mark.parametrize('criterion', ['cost', 'gini', 'entropy'])
def test_cstree_criteria(data, criterion):
    X, y = data
    model = CSTreeClassifier(criterion=criterion)
    model.fit(X, y, fp_cost=1, fn_cost=1)
    y_proba = model.predict_proba(X)
    assert hasattr(model, 'estimator_')
    assert y_proba.shape == (100, 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


class TestCustomCriterionNotMutated:
    """Regression tests: a user-supplied CostImpurity instance used to be stored (and mutated) directly.

    `_fit` calls `set_costs()`/`set_array_costs()` on whatever ends up in `self.criterion_`; storing
    the user's instance there directly meant fitting mutated an `__init__` parameter in place,
    corrupting sklearn's clone()/get_params() contract and silently sharing state between any two
    estimators (or two `fit()` calls) given the same criterion instance.
    """

    @pytest.mark.parametrize('criterion_cls', [CostImpurity, GiniCostImpurity])
    def test_shared_criterion_instance_across_two_estimators(self, data, criterion_cls):
        X, y = data
        shared_criterion = criterion_cls(n_outputs=1, n_classes=np.array([2], dtype=np.intp))

        model1 = CSTreeClassifier(criterion=shared_criterion)
        model1.fit(X, y, fp_cost=1.0, fn_cost=5.0)
        model2 = CSTreeClassifier(criterion=shared_criterion)
        model2.fit(X, y, fp_cost=10.0, fn_cost=1.0)

        # The __init__ parameter itself must be untouched (sklearn's clone()/get_params()
        # contract): both models still reference the exact same original object.
        assert model1.criterion is shared_criterion
        assert model2.criterion is shared_criterion
        # But the object actually used for fitting must be an independent instance per model,
        # not the shared original and not shared between the two models.
        assert model1.criterion_ is not shared_criterion
        assert model2.criterion_ is not shared_criterion
        assert model1.criterion_ is not model2.criterion_

        # Both fits must succeed independently without raising or corrupting each other.
        assert model1.predict_proba(X).shape == (100, 2)
        assert model2.predict_proba(X).shape == (100, 2)

    def test_fresh_uninitialized_criterion_does_not_crash(self, data):
        """A freshly constructed (never fit) CostImpurity must be usable without crashing.

        `CostImpurity`'s C-level cost buffers are uninitialized until `set_array_costs()` is
        called; a naive `copy.deepcopy()` of the user-supplied instance dereferences those
        buffers unconditionally and crashes (compiled with `initializedcheck=False`), so the fix
        must not deep-copy an instance in this state.
        """
        X, y = data
        criterion = CostImpurity(n_outputs=1, n_classes=np.array([2], dtype=np.intp))
        model = CSTreeClassifier(criterion=criterion).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        y_proba = model.predict_proba(X)
        assert y_proba.shape == (100, 2)


def test_cstree_with_stochastic_maxprofit_metric(data):
    """Regression test: a MaxProfit metric with a stochastic variable used to raise, not reduce to its mean.

    `_fit` used to call `self.loss._evaluate_costs(**loss_params)` without `replace_stochastic=True`,
    so a `sympy.stats` random variable in the cost/benefit expression raised deep inside
    `_evaluate_expression` instead of being reduced to its mean - the same reduction
    `_prepare_class_costs`/`ProfTreeClassifier` already apply for exactly this situation.
    """
    X, y = data
    clv = sympy.stats.Beta('clv', 2, 5)
    contact_cost = sympy.symbols('contact_cost')
    metric = Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(contact_cost), MaxProfit())

    model = CSTreeClassifier(loss=metric).fit(X, y, contact_cost=1.0)
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (100, 2)
