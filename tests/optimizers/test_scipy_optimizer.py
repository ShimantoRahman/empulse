import warnings
from dataclasses import dataclass

import pytest
from scipy.optimize import OptimizeResult
from sklearn.exceptions import ConvergenceWarning

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
