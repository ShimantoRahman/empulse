"""Regularization and L1-solver behaviour of :class:`~empulse.models.CSLogitClassifier`.

The elastic-net penalty used to be added unnormalized to a sample-averaged data loss, which made it
roughly ``n_samples`` times stronger than scikit-learn's at the same ``C``. Because the expected
cost is linear in the predicted probability, its gradient is bounded by ``0.25 * max|c1 - c2|``, so
with mirrored unit costs ``w = 0`` genuinely satisfied the L1 optimality condition: the model
returned all-zero coefficients and predicted exactly 0.5 for every sample. On top of that, L-BFGS-B
cannot minimize a non-smooth L1 objective at all, since ``sign(0) == 0`` makes the origin look
stationary.
"""

import warnings
from typing import ClassVar

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from empulse.metrics import Cost, CostMatrix, LogCost, Metric
from empulse.models import CSLogitClassifier, CSTreeClassifier
from empulse.optimizers import LBFGSBOptimizer, ScipyOptimizer


@pytest.fixture(scope='module')
def standardised_data():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=12, weights=[0.7, 0.3])
    return (X - X.mean(axis=0)) / X.std(axis=0), y


def fit(X, y, **kwargs):
    return CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, **kwargs).fit(X, y)


class TestLearnsWithMirroredCosts:
    """The reported failure: mirrored unit costs and a pure L1 penalty learned nothing."""

    def test_learns_instead_of_predicting_half(self, standardised_data):
        X, y = standardised_data
        clf = fit(X, y, l1_ratio=1.0, C=1.0)

        assert clf.n_iter_ > 0
        assert clf.result_.status == 0
        assert np.any(clf.coef_ != 0)
        assert clf.predict_proba(X)[:, 1].std() > 0.1

    def test_beats_the_constant_predictor(self, standardised_data):
        X, y = standardised_data
        clf = fit(X, y, l1_ratio=1.0, C=1.0)

        proba = clf.predict_proba(X)[:, 1]
        # With fp_cost = fn_cost = 1 the expected cost of predicting 0.5 everywhere is exactly 0.5.
        expected_cost = float(np.mean(np.where(y == 1, 1 - proba, proba)))
        assert expected_cost < 0.5

    @pytest.mark.parametrize('l1_ratio', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_converges_across_the_elastic_net_range(self, standardised_data, l1_ratio):
        X, y = standardised_data
        clf = fit(X, y, l1_ratio=l1_ratio, C=1.0)

        assert clf.result_.status == 0
        assert clf.n_iter_ > 0
        assert np.any(clf.coef_ != 0)


class TestScaleInvariance:
    """``C`` is scaled by the objective magnitude, so rescaling the costs must not change the fit."""

    @pytest.mark.parametrize('C', [0.01, 0.1, 1.0])
    def test_regularisation_path_survives_rescaling_the_costs(self, standardised_data, C):
        """Rescaling every cost by a constant must select the same model.

        Only approximately at the default tolerance: the penalty scaling is exactly invariant, but
        L-BFGS-B still stops at a slightly different point, which is what
        :meth:`test_rescaling_difference_is_only_the_stopping_point` pins down.
        """
        X, y = standardised_data
        fits = {
            scale: CSLogitClassifier(fp_cost=scale, fn_cost=scale, l1_ratio=1.0, C=C).fit(X, y).coef_
            for scale in (1.0, 10.0, 1000.0)
        }
        reference = fits[10.0]
        for scale, coef in fits.items():
            np.testing.assert_allclose(
                coef, reference, atol=5e-2, err_msg=f'coefficients differ at cost magnitude {scale}'
            )

    def test_rescaling_difference_is_only_the_stopping_point(self, standardised_data):
        """Tightening the tolerance must drive the remaining difference to zero.

        If rescaling the costs changed the regularization itself, the gap would be a fixed property
        of the problem and would not shrink. It falls by roughly four orders of magnitude, which is
        what identifies it as a stopping criterion artifact.
        """
        X, y = standardised_data

        def worst_gap(tolerance: float) -> float:
            coefs = [
                CSLogitClassifier(
                    fp_cost=scale,
                    fn_cost=scale,
                    l1_ratio=1.0,
                    C=1.0,
                    optimizer=LBFGSBOptimizer(tolerance=tolerance),
                )
                .fit(X, y)
                .coef_
                for scale in (1.0, 10.0, 1000.0)
            ]
            return max(float(np.max(np.abs(coef - coefs[1]))) for coef in coefs)

        assert worst_gap(1e-8) < 1e-4
        assert worst_gap(1e-8) < worst_gap(1e-4)

    def test_coefficients_survive_rescaling_the_costs(self, standardised_data):
        X, y = standardised_data
        small = CSLogitClassifier(fp_cost=1.0, fn_cost=3.0, l1_ratio=0.0, C=1.0).fit(X, y)
        large = CSLogitClassifier(fp_cost=1000.0, fn_cost=3000.0, l1_ratio=0.0, C=1.0).fit(X, y)
        np.testing.assert_allclose(small.coef_, large.coef_, rtol=1e-3, atol=1e-5)

    def test_lambda_matches_sklearn_for_plain_log_loss(self, standardised_data):
        """For plain log loss the objective scale is 1, so lambda is scikit-learn's 1 / (C n)."""
        X, y = standardised_data
        features = np.hstack([np.ones((X.shape[0], 1)), X])
        metric = Metric(CostMatrix().add_tp_benefit('b').add_tn_benefit('b'), LogCost())
        objective = metric._logit_objective(features=features, y_true=y, C=2.0, l1_ratio=1.0, fit_intercept=True, b=1.0)
        assert objective.penalty.lambda_ == pytest.approx(1 / (2.0 * len(y)))

    @pytest.mark.parametrize('C', [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_reproduces_sklearn_logistic_regression(self, standardised_data, C):
        """The whole point of the new scaling: at the same ``C``, plain log loss must give
        scikit-learn's model.

        A :class:`~empulse.metrics.LogCost` metric whose only terms are a true-positive and
        true-negative benefit of 1 *is* the log loss, so an L2-penalized fit has to land on
        :class:`~sklearn:sklearn.linear_model.LogisticRegression`. Both solvers are driven to a
        tight optimum, because at their default tolerances they stop ~5e-4 apart -- which is the
        stopping point, not the objective.
        """
        X, y = standardised_data
        metric = Metric(CostMatrix().add_tp_benefit('b').add_tn_benefit('b'), LogCost())
        ours = CSLogitClassifier(
            loss=metric, l1_ratio=0.0, C=C, optimizer=LBFGSBOptimizer(tolerance=1e-9, max_iter=5000)
        ).fit(X, y, b=1.0)
        theirs = LogisticRegression(C=C, tol=1e-12, max_iter=5000).fit(X, y)

        np.testing.assert_allclose(ours.coef_, theirs.coef_[0], atol=1e-7)
        np.testing.assert_allclose(ours.intercept_, theirs.intercept_[0], atol=1e-7)


class TestSparsity:
    """L-BFGS-B reaches exact zeros only through the split-variable reformulation."""

    def test_l1_produces_exact_zeros(self, standardised_data):
        X, y = standardised_data
        clf = fit(X, y, l1_ratio=1.0, C=0.1)

        zeros = clf.coef_ == 0.0
        assert zeros.any()
        assert not zeros.all()

    def test_stronger_regularisation_is_at_least_as_sparse(self, standardised_data):
        X, y = standardised_data
        counts = [np.count_nonzero(fit(X, y, l1_ratio=1.0, C=C).coef_) for C in (0.01, 0.1, 1.0, 10.0)]
        assert counts == sorted(counts), counts

    def test_intercept_is_not_penalised(self, standardised_data):
        X, y = standardised_data
        # Heavy regularization zeroes every coefficient, but the intercept must still be free to
        # move to the base rate.
        clf = fit(X, y, l1_ratio=1.0, C=1e-4)
        assert np.all(clf.coef_ == 0.0)
        assert clf.intercept_ != 0.0

    def test_matches_a_reference_proximal_solver(self, standardised_data):
        """FISTA on the same objective, as an independent check of the reformulation."""
        X, y = standardised_data
        clf = fit(X, y, l1_ratio=0.5, C=0.5)

        features = np.hstack([np.ones((X.shape[0], 1)), X])
        metric = Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost'), Cost())
        objective = metric._logit_objective(
            features=features, y_true=y, C=0.5, l1_ratio=0.5, fit_intercept=True, fp_cost=1.0, fn_cost=1.0
        )
        penalty = objective.penalty

        w = np.zeros(features.shape[1])
        z, t, step = w.copy(), 1.0, 20.0
        for _ in range(20000):
            _, gradient = objective.data_loss_gradient(z)
            nxt = penalty.prox(z - gradient * step, step)
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = nxt + ((t - 1) / t_next) * (nxt - w)
            w, t = nxt, t_next

        fitted = np.r_[clf.intercept_, clf.coef_]
        # The two solvers agree on the support exactly and on the objective value to near machine
        # precision; the coefficients themselves only to the reference loop's own convergence.
        np.testing.assert_array_equal(fitted == 0.0, w == 0.0)
        assert objective.logit_loss(fitted) == pytest.approx(objective.logit_loss(w), abs=1e-6)
        np.testing.assert_allclose(fitted, w, atol=5e-2)

    def test_split_variable_can_be_disabled(self, standardised_data):
        X, y = standardised_data
        plain = fit(X, y, l1_ratio=1.0, C=1.0, optimizer=LBFGSBOptimizer(split_variable=False))
        split = fit(X, y, l1_ratio=1.0, C=1.0, optimizer=LBFGSBOptimizer(split_variable=True))

        # Subgradient descent cannot reach exact zeros; the reformulation can.
        assert np.count_nonzero(split.coef_) < np.count_nonzero(plain.coef_)

    def test_split_variable_true_rejects_a_smooth_penalty(self, standardised_data):
        X, y = standardised_data
        with pytest.raises(ValueError, match='non-smooth'):
            fit(X, y, l1_ratio=0.0, optimizer=LBFGSBOptimizer(split_variable=True))

    def test_scipy_optimizer_does_not_reformulate(self, standardised_data):
        X, y = standardised_data
        generic = fit(X, y, l1_ratio=1.0, C=1.0, optimizer=ScipyOptimizer(max_iter=200))
        specific = fit(X, y, l1_ratio=1.0, C=1.0, optimizer=LBFGSBOptimizer(max_iter=200))
        assert np.count_nonzero(specific.coef_) < np.count_nonzero(generic.coef_)

    def test_explicit_bounds_fall_back_with_a_warning(self, standardised_data):
        X, y = standardised_data
        n_params = X.shape[1] + 1
        optimizer = LBFGSBOptimizer()
        metric = Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost'), Cost())
        features = np.hstack([np.ones((X.shape[0], 1)), X])
        objective = metric._logit_objective(
            features=features, y_true=y, C=1.0, l1_ratio=1.0, fit_intercept=True, fp_cost=1.0, fn_cost=1.0
        )
        with pytest.warns(UserWarning, match='bounds cannot be combined'):
            optimizer(objective=objective, X=features, bounds=[(-1.0, 1.0)] * n_params)


class TestGradientCorrectness:
    """The analytic gradient must be the gradient of the loss that is reported alongside it."""

    @pytest.mark.parametrize('strategy', [Cost, LogCost], ids=['Cost', 'LogCost'])
    @pytest.mark.parametrize('l1_ratio', [0.0, 0.5])
    def test_gradient_matches_finite_differences(self, standardised_data, strategy, l1_ratio):
        X, y = standardised_data
        features = np.hstack([np.ones((X.shape[0], 1)), X])
        metric = Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost'), strategy())
        objective = metric._logit_objective(
            features=features,
            y_true=y,
            C=1.0,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            fp_cost=1.0,
            fn_cost=1.0,
        )
        # Evaluated away from the origin: an L1 penalty is not differentiable at w_j = 0.
        rng = np.random.default_rng(0)
        weights = rng.uniform(0.2, 0.6, size=features.shape[1]) * rng.choice([-1.0, 1.0], features.shape[1])

        _, analytic = objective.logit_loss_gradient(weights)
        numeric = approx_fprime(weights, objective.logit_loss, 1e-7)
        np.testing.assert_allclose(analytic, numeric, atol=1e-5)

    @pytest.mark.parametrize('strategy', [Cost, LogCost], ids=['Cost', 'LogCost'])
    def test_data_term_excludes_the_penalty(self, standardised_data, strategy):
        X, y = standardised_data
        features = np.hstack([np.ones((X.shape[0], 1)), X])
        metric = Metric(CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost'), strategy())
        objective = metric._logit_objective(
            features=features, y_true=y, C=1.0, l1_ratio=1.0, fit_intercept=True, fp_cost=1.0, fn_cost=1.0
        )
        weights = np.full(features.shape[1], 0.3)

        data_loss, data_gradient = objective.data_loss_gradient(weights)
        full_loss, full_gradient = objective.logit_loss_gradient(weights)
        penalty = objective.penalty

        assert full_loss == pytest.approx(data_loss + penalty.value(weights))
        np.testing.assert_allclose(full_gradient, data_gradient + penalty.gradient(weights), atol=1e-12)


class TestNoTrainingSignal:
    """A cost matrix with constant rows leaves a gradient-based objective nothing to descend.

    ``tp_cost == fn_cost`` and ``fp_cost == tn_cost`` means classifying a sample either way costs
    the same, so the derivative of the expected cost with respect to the predicted probability is
    zero for every sample. That is a property of the *gradient-based* objectives, not of every
    cost-sensitive model: see :meth:`test_tree_models_still_learn_and_do_not_warn`.
    """

    DEGENERATE: ClassVar[dict[str, float]] = {'tp_cost': 2.0, 'fn_cost': 2.0, 'fp_cost': 3.0, 'tn_cost': 3.0}

    def test_warns_for_gradient_based_models(self, standardised_data):
        X, y = standardised_data
        with pytest.warns(UserWarning, match='no gradient'):
            CSLogitClassifier(**self.DEGENERATE).fit(X, y)

    def test_the_gradient_really_is_zero(self, standardised_data):
        X, y = standardised_data
        features = np.hstack([np.ones((X.shape[0], 1)), X])
        matrix = CostMatrix().add_tp_cost('tp').add_fn_cost('fn').add_fp_cost('fp').add_tn_cost('tn')
        objective = Metric(matrix, Cost())._logit_objective(
            features=features, y_true=y, C=1.0, l1_ratio=0.0, fit_intercept=True, tp=2.0, fn=2.0, fp=3.0, tn=3.0
        )
        _, gradient = objective.data_loss_gradient(np.full(features.shape[1], 0.3))
        assert np.all(gradient == 0.0)

    def test_does_not_warn_for_a_usable_matrix(self, standardised_data):
        X, y = standardised_data
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            CSLogitClassifier(fp_cost=1.0, fn_cost=5.0).fit(X, y)

    @pytest.mark.parametrize('criterion', ['gini', 'entropy'])
    def test_tree_models_still_learn_and_do_not_warn(self, standardised_data, criterion):
        """Trees weight their split quality by class purity, which owes nothing to the costs.

        This is why the warning belongs on the objectives that differentiate the cost matrix rather
        than on the shared ``fit``: the very same matrix that stops a logit model dead leaves a tree
        or forest learning perfectly well.
        """
        X, y = standardised_data
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            tree = CSTreeClassifier(criterion=criterion, max_depth=4, random_state=0, **self.DEGENERATE)
            tree.fit(X, y)

        assert roc_auc_score(y, tree.predict_proba(X)[:, 1]) > 0.8
        assert len(np.unique(tree.predict(X))) > 1
