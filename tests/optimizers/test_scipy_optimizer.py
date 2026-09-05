import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from empulse.models import CSLogitClassifier
from empulse.optimizers import LBFGSBOptimizer, ScipyOptimizer
from empulse.optimizers._scipy import _check_optimize_result


@dataclass
class _StatuslessResult:
    """Mimics an OptimizeResult from a method that never sets `.status`."""

    success: bool
    message: str = 'done'


class TestCheckOptimizeResult:
    def test_warns_when_not_converged(self):
        result = OptimizeResult(status=1, success=False, message='did not converge')
        with pytest.warns(ConvergenceWarning):
            _check_optimize_result(result)

    def test_no_warning_when_converged(self):
        result = OptimizeResult(status=0, success=True, message='converged')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _check_optimize_result(result)

    def test_no_status_attribute_but_success_true_does_not_raise(self):
        """A result missing `.status` entirely must not raise AttributeError."""
        result = _StatuslessResult(success=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _check_optimize_result(result)

    def test_no_status_attribute_and_success_false_still_warns(self):
        """Missing `.status` but a failed optimisation should still warn, using a default status."""
        result = _StatuslessResult(success=False, message='did not converge')
        with pytest.warns(ConvergenceWarning, match='status=0'):
            _check_optimize_result(result)

    def test_missing_success_attribute_defaults_to_converged(self):
        """A result missing both `.status` and `.success` is assumed converged (no warning)."""

        class _BareResult:
            message = 'n/a'

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _check_optimize_result(_BareResult())

    def test_optimizer_name_in_message(self):
        result = OptimizeResult(status=2, success=False, message='boom')
        with pytest.warns(ConvergenceWarning, match='my-optimizer'):
            _check_optimize_result(result, 'my-optimizer')


@pytest.fixture
def data():
    return make_classification(n_samples=200, n_features=5, random_state=0)


class TestScipyOptimizer:
    """`ScipyOptimizer` was never exercised directly - only its default (`LBFGSBOptimizer`-like)
    settings overlapped with other tests. These cover its actual reason to exist: swapping in a
    different `scipy.optimize.minimize` method, gradient-free optimization, and passing through
    solver-specific keyword arguments.
    """

    def test_default_settings_separate_classes(self, data):
        """Default method ('L-BFGS-B', use_jacobian=True) must behave like a real optimizer."""
        X, y = data
        clf = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(max_iter=200))
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)[:, 1]
        assert roc_auc_score(y, y_proba) > 0.8

    def test_alternate_gradient_method_separates_classes(self, data):
        """A non-default gradient-based method ('CG') must still be wired up correctly."""
        X, y = data
        clf = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(method='CG', max_iter=200))
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)[:, 1]
        assert roc_auc_score(y, y_proba) > 0.8

    def test_use_jacobian_false_derivative_free_method(self, data):
        """`use_jacobian=False` must route through `logit_loss` (no gradient) instead of
        `logit_loss_gradient`, for genuinely derivative-free methods like Nelder-Mead.
        """
        X, y = data
        clf = CSLogitClassifier(
            fp_cost=1.0,
            fn_cost=1.0,
            optimizer=ScipyOptimizer(method='Nelder-Mead', use_jacobian=False, max_iter=500),
        )
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)[:, 1]
        assert roc_auc_score(y, y_proba) > 0.7

    def test_tolerance_is_forwarded(self, data):
        """A very loose `tolerance` should make the solver stop (much) earlier."""
        X, y = data
        clf_loose = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(tolerance=1.0))
        clf_tight = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(tolerance=1e-10))
        clf_loose.fit(X, y)
        clf_tight.fit(X, y)
        assert clf_loose.n_iter_ <= clf_tight.n_iter_

    def test_options_maxiter_overrides_max_iter(self):
        """`options={'maxiter': ...}` must win over the `max_iter` constructor argument, since
        `merged_options = {'maxiter': self.max_iter, **self.options}` applies `options` last.
        """
        X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        clf = CSLogitClassifier(
            fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(max_iter=1000, options={'maxiter': 2})
        )
        with pytest.warns(ConvergenceWarning):
            clf.fit(X, y)
        assert clf.n_iter_ <= 2

    def test_scipy_kwargs_bounds_are_respected(self, data):
        """Arbitrary solver kwargs (e.g. `bounds`) must reach `scipy.optimize.minimize` verbatim."""
        X, y = data
        n_params = X.shape[1] + 1  # +1 for the intercept CSLogitClassifier prepends
        clf = CSLogitClassifier(
            fp_cost=1.0,
            fn_cost=1.0,
            optimizer=ScipyOptimizer(max_iter=200, bounds=[(-0.5, 0.5)] * n_params),
        )
        clf.fit(X, y)
        assert np.all(clf.coef_ >= -0.5 - 1e-8) and np.all(clf.coef_ <= 0.5 + 1e-8)
        assert -0.5 - 1e-8 <= clf.intercept_ <= 0.5 + 1e-8

    def test_result_matches_optimizer_contract(self, data):
        """The returned OptimizeResult must expose the fields Optimizer's contract promises."""
        X, y = data
        clf = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(max_iter=50))
        clf.fit(X, y)
        result = clf.result_
        assert isinstance(result, OptimizeResult)
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'nit')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')

    def test_matches_lbfgsb_optimizer_direction(self, data):
        """ScipyOptimizer with default settings should broadly agree with LBFGSBOptimizer, since
        both ultimately drive the same 'L-BFGS-B' method with the analytic gradient - though exact
        coefficients differ slightly (LBFGSBOptimizer additionally tunes maxls/gtol/ftol).
        """
        X, y = data
        clf_generic = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=ScipyOptimizer(max_iter=200))
        clf_specific = CSLogitClassifier(fp_cost=1.0, fn_cost=1.0, optimizer=LBFGSBOptimizer(max_iter=200))
        clf_generic.fit(X, y)
        clf_specific.fit(X, y)
        # Same sign on every coefficient, and correlated predictions on the training data.
        np.testing.assert_array_equal(np.sign(clf_generic.coef_), np.sign(clf_specific.coef_))
        proba_generic = clf_generic.predict_proba(X)[:, 1]
        proba_specific = clf_specific.predict_proba(X)[:, 1]
        assert np.corrcoef(proba_generic, proba_specific)[0, 1] > 0.99
