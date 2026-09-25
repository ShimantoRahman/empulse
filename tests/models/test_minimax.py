import numpy as np
import pytest
import sympy

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.models import ProfMEMPMClassifier, ProfMPMClassifier

MODEL_CLASSES = [ProfMPMClassifier, ProfMEMPMClassifier]


@pytest.fixture(scope='module')
def data(make_data):
    return make_data(n_samples=200, n_features=5)


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

    ``_worst_case_accuracies`` is the one real behavioral difference between ProfMPMClassifier
    and ProfMEMPMClassifier that survived extracting their shared BaseMinimaxProbabilityMachine
    base, and every ``_solve_*`` method on that base defers to it for its final
    ``(alpha_1, alpha_0)``.
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

    def test_unregularized_mempm_beta_max_uses_true_kappa_supremum(self, seeded_rng):
        """The search bound for alpha_0 must be the true supremum of kappa_0 over the feasible
        set (sqrt(delta_mu^T Sigma_0^-1 delta_mu)), not kappa_0 at the unrelated MPM solution.

        With anisotropic, differently-shaped covariances, the MPM solution direction is a poor
        proxy for the direction that maximizes kappa_0, so bounding the search by it leaves most
        of the feasible (0, 1) range for alpha_0 unexplored and can cost double-digit percentages
        of worst-case profit.
        """
        n = 2000
        sigma_1 = np.array([[4.0, 0.0], [0.0, 0.05]])
        sigma_0 = np.array([[0.05, 0.0], [0.0, 4.0]])
        X_pos = seeded_rng.multivariate_normal([1.0, 1.0], sigma_1, size=n)
        X_neg = seeded_rng.multivariate_normal([0.0, 0.0], sigma_0, size=n)
        X = np.vstack([X_pos, X_neg])
        y = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]

        # Heavily favors specificity, so the optimum sits where alpha_0 is large.
        model = ProfMEMPMClassifier().fit(X, y, tp_cost=-1.0, fp_cost=1000.0)

        # The bug capped alpha_0 near 0.59 on data shaped like this; the true achievable
        # bound is close to 0.95.
        assert model.alpha_0_ > 0.8


class TestConstraints:
    """Tests for scale constraints and boundary conditions from Maldonado et al. (2020)."""

    @pytest.mark.parametrize('model_cls', MODEL_CLASSES)
    def test_unregularized_scale_constraint(self, model_cls, data):
        """Unregularized models satisfy w^T(mu_1 - mu_0) = 1 (Eq. 7 and Eq. 14)."""
        X, y = data
        model = model_cls(lambda_reg=0.0).fit(X, y, tp_cost=-200, fp_cost=10)
        mu_1 = np.mean(X[y == 1], axis=0)
        mu_0 = np.mean(X[y == 0], axis=0)
        assert float(model.coef_ @ (mu_1 - mu_0)) == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize('model_cls', MODEL_CLASSES)
    def test_unregularized_intercept_boundary_condition(self, model_cls, data):
        """Intercept b* satisfies tight boundary condition (Eq. 28 and Eq. 29)."""
        X, y = data
        model = model_cls(lambda_reg=0.0).fit(X, y, tp_cost=-200, fp_cost=10)
        mu_1 = np.mean(X[y == 1], axis=0)
        mu_0 = np.mean(X[y == 0], axis=0)
        ridge = np.eye(X.shape[1]) * model.ridge_penalty
        sigma_1 = np.cov(X[y == 1], rowvar=False) + ridge
        sigma_0 = np.cov(X[y == 0], rowvar=False) + ridge

        w = model.coef_
        d1 = float(np.sqrt(w @ sigma_1 @ w))
        d0 = float(np.sqrt(w @ sigma_0 @ w))
        k1 = np.sqrt(model.alpha_1_ / (1.0 - model.alpha_1_))
        k0 = np.sqrt(model.alpha_0_ / (1.0 - model.alpha_0_))

        b1 = -float(w @ mu_1) + k1 * d1
        b0 = -float(w @ mu_0) - k0 * d0
        assert b1 == pytest.approx(b0, abs=1e-3)
        assert model.intercept_ == pytest.approx(b1, abs=1e-3)

    @pytest.mark.parametrize('model_cls', MODEL_CLASSES)
    @pytest.mark.parametrize('penalty', ['l1', 'l2'])
    def test_regularized_margin_constraints(self, model_cls, penalty, data):
        """Regularized models keep each class mean a unit inside its half-space (Eq. 9)."""
        X, y = data
        model = model_cls(lambda_reg=1.0, penalty=penalty).fit(X, y, tp_cost=-200, fp_cost=10)
        mu_1 = np.mean(X[y == 1], axis=0)
        mu_0 = np.mean(X[y == 0], axis=0)
        assert float(model.coef_ @ mu_1 + model.intercept_) >= 1.0 - 1e-6
        assert -float(model.coef_ @ mu_0 + model.intercept_) >= 1.0 - 1e-6


class TestRegularization:
    """
    The penalty must change the classifier, not just the size of its weights.

    The Chebyshev constraints of Formulations (17) and (20) are unchanged by scaling ``(w, b)``, so
    without Eq. (9)'s margin constraints the penalty shrank ``w`` towards zero at no cost: every
    ``lambda_reg`` gave the same direction, worst-case accuracies and predictions, until the solution
    collapsed numerically.
    """

    FN_COST = 4.0
    FP_COST = 1.0

    @pytest.fixture(scope='class')
    def independent_data(self, make_data):
        # No redundant features: they make the optimal direction non-unique, so a sparse solution
        # would not need the penalty to be doing anything.
        return make_data(n_samples=400, n_features=5, n_informative=3, n_redundant=0)

    @pytest.mark.parametrize('model_cls', MODEL_CLASSES)
    def test_strong_l1_penalty_zeroes_coefficients(self, model_cls, independent_data):
        X, y = independent_data
        model = model_cls(lambda_reg=10.0, penalty='l1').fit(X, y, fn_cost=self.FN_COST, fp_cost=self.FP_COST)
        coef = np.abs(model.coef_)
        assert np.sum(coef <= 1e-6 * coef.max()) >= 1

    @pytest.mark.parametrize('penalty', ['l1', 'l2'])
    def test_penalty_is_paid_for_in_worst_case_profit(self, penalty, independent_data):
        """ProfMEMPM maximises c_1 alpha_1 + c_0 alpha_0 minus the penalty, so more penalty, less of it."""
        X, y = independent_data
        c_1 = np.mean(y == 1) * self.FN_COST
        c_0 = np.mean(y == 0) * self.FP_COST

        def worst_case_profit(lambda_reg):
            model = ProfMEMPMClassifier(lambda_reg=lambda_reg, penalty=penalty)
            model.fit(X, y, fn_cost=self.FN_COST, fp_cost=self.FP_COST)
            return c_1 * model.alpha_1_ + c_0 * model.alpha_0_

        assert worst_case_profit(1.0) < worst_case_profit(1e-3) - 0.01
