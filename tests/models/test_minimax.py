import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.models import ProfMEMPMClassifier, ProfMPMClassifier

MODEL_CLASSES = [ProfMPMClassifier, ProfMEMPMClassifier]


@pytest.fixture
def data():
    return make_classification(n_samples=200, n_features=5, random_state=42)


@pytest.mark.parametrize('model_cls', MODEL_CLASSES)
def test_fit_predict_proba(model_cls, data):
    X, y = data
    model = model_cls().fit(X, y, tp_cost=-200, fp_cost=10)
    y_proba = model.predict_proba(X)

    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    assert model.coef_.shape == (X.shape[1],)
    assert y_proba.shape == (X.shape[0], 2)
    assert np.allclose(y_proba.sum(axis=1), 1)


@pytest.mark.parametrize('model_cls', MODEL_CLASSES)
@pytest.mark.parametrize('penalty', ['l1', 'l2'])
def test_regularized_fit(model_cls, penalty, data):
    X, y = data
    model = model_cls(lambda_reg=0.5, penalty=penalty).fit(X, y, tp_cost=-200, fp_cost=10)
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (X.shape[0], 2)
    assert np.isfinite(model.coef_).all()


@pytest.mark.parametrize('model_cls', MODEL_CLASSES)
def test_fit_with_maxprofit_metric(model_cls, data):
    """Only BaseMetric instances built with MaxProfit are supported."""
    X, y = data
    clv, d = sympy.symbols('clv d')
    metric = Metric(CostMatrix().add_tp_benefit(clv).add_fp_cost(d), MaxProfit())
    model = model_cls(loss=metric).fit(X, y, clv=100.0, d=10.0)
    y_proba = model.predict_proba(X)
    assert y_proba.shape == (X.shape[0], 2)


@pytest.mark.parametrize('model_cls', MODEL_CLASSES)
def test_instance_dependent_costs_are_aggregated_to_mean(model_cls, data):
    """Per the docstring, array-like costs are aggregated to their mean before fitting."""
    X, y = data
    rng = np.random.default_rng(0)
    fn_cost = rng.uniform(1.0, 5.0, size=len(y))
    model = model_cls().fit(X, y, fn_cost=fn_cost, fp_cost=1.0)
    y_proba = model.predict_proba(X)
    assert np.isfinite(y_proba).all()


class TestWorstCaseAccuracies:
    """Regression tests for the shared-vs-per-class worst-case accuracy bound distinction.

    This is the one real behavioral difference between ProfMPMClassifier and
    ProfMEMPMClassifier that survived extracting their shared BaseMinimaxProbabilityMachine base.
    """

    def test_profmpm_shares_one_alpha_across_classes(self):
        model = ProfMPMClassifier()
        alpha_1, alpha_0 = model._worst_case_accuracies(k_1=2.0, k_0=0.5)
        assert alpha_1 == alpha_0

    def test_profmempm_computes_alpha_per_class(self):
        model = ProfMEMPMClassifier()
        alpha_1, alpha_0 = model._worst_case_accuracies(k_1=2.0, k_0=0.5)
        assert alpha_1 != alpha_0
        # alpha_1 (k_1=2.0) should be higher than alpha_0 (k_0=0.5): k^2/(1+k^2) is increasing in k.
        assert alpha_1 > alpha_0

    @pytest.mark.parametrize('k_1, k_0', [(-1.0, 2.0), (2.0, -1.0), (-1.0, -1.0)])
    def test_non_positive_kappa_gives_zero_accuracy(self, k_1, k_0):
        """A non-positive kappa means the Chebyshev-Cantelli bound is vacuous (alpha=0)."""
        mpm_alpha_1, mpm_alpha_0 = ProfMPMClassifier()._worst_case_accuracies(k_1, k_0)
        mempm_alpha_1, mempm_alpha_0 = ProfMEMPMClassifier()._worst_case_accuracies(k_1, k_0)
        if k_1 <= 0:
            assert mempm_alpha_1 == 0.0
        if k_0 <= 0:
            assert mempm_alpha_0 == 0.0
        if min(k_1, k_0) <= 0:
            assert mpm_alpha_1 == mpm_alpha_0 == 0.0


class TestConstraints:
    """Regression tests for the unit-norm-vs-unconstrained distinction between the two models.

    ProfMEMPMClassifier constrains ||w||=1 when unregularized (dropped when lambda_reg > 0);
    ProfMPMClassifier has never applied any constraint (see MODELS_OPTIMIZERS_REVIEW.md item 14 -
    intentionally preserved as-is by this refactor, not fixed here).
    """

    def test_profmpm_has_no_constraints(self):
        model = ProfMPMClassifier()
        assert model._build_constraints(regularized=False) == ()
        assert model._build_constraints(regularized=True) == ()

    def test_profmempm_constrains_unit_norm_when_unregularized(self):
        model = ProfMEMPMClassifier()
        constraints = model._build_constraints(regularized=False)
        assert constraints != ()
        assert constraints['type'] == 'eq'

    def test_profmempm_drops_constraint_when_regularized(self):
        model = ProfMEMPMClassifier()
        assert model._build_constraints(regularized=True) == ()

    def test_profmempm_unregularized_coef_has_unit_norm(self, data):
        X, y = data
        model = ProfMEMPMClassifier(lambda_reg=0.0).fit(X, y, tp_cost=-200, fp_cost=10)
        assert np.linalg.norm(model.coef_) == pytest.approx(1.0, abs=1e-4)
