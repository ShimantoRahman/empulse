import warnings
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from sklearn.exceptions import ConvergenceWarning

from .._types import FloatNDArray
from ..metrics import LogitObjective
from ._base import Optimizer


def _check_optimize_result(result: OptimizeResult, optimizer_name: str = 'scipy') -> None:
    """Warn if the optimizer did not converge.

    Not every ``scipy.optimize.minimize`` method populates ``result.status`` (some only set
    ``result.success``), so convergence is judged from ``success`` (defaulting to converged/``True``
    if even that is missing) rather than requiring ``status`` to exist. ``status`` is read
    defensively too, since it is only used for the warning message.
    """
    if getattr(result, 'success', True):
        return
    status = getattr(result, 'status', 0)
    warnings.warn(
        f'{optimizer_name} failed to converge (status={status}):\n{result.message}.\n\n'
        'Increase max_iter or scale the data as shown in:\n'
        '    https://scikit-learn.org/stable/modules/preprocessing.html',
        ConvergenceWarning,
        stacklevel=3,
    )


class LBFGSBOptimizer(Optimizer):
    """Limited-memory BFGS with box constraints (L-BFGS-B) via :func:`scipy.optimize.minimize`.

    This is the default optimizer for :class:`~empulse.models.CSLogitClassifier`.
    It is well-suited for smooth objectives and scales to thousands of features.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of L-BFGS-B iterations.
    tolerance : float, default=1e-4
        Gradient infinity-norm convergence tolerance (``gtol``).
    max_line_search_steps : int, default=50
        Maximum number of line-search steps per iteration (``maxls``).
    ftol_scale : float, default=64.0
        Function-value tolerance is set to ``ftol_scale * machine_epsilon``.

    Examples
    --------
    .. code-block:: python

        from empulse.models import CSLogitClassifier
        from empulse.optimizers import LBFGSBOptimizer

        model = CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=500, tolerance=1e-5))
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        max_line_search_steps: int = 50,
        ftol_scale: float = 64.0,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_line_search_steps = max_line_search_steps
        self.ftol_scale = ftol_scale

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """Run L-BFGS-B optimisation."""
        initial_weights = self._initial_weights(X)
        result = minimize(
            objective.logit_loss_gradient,
            initial_weights,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': self.max_iter,
                'maxls': self.max_line_search_steps,
                'gtol': self.tolerance,
                'ftol': self.ftol_scale * np.finfo(float).eps,
            },
            **kwargs,
        )
        _check_optimize_result(result, 'L-BFGS-B')
        return result


class ScipyOptimizer(Optimizer):
    """General-purpose wrapper around :func:`scipy.optimize.minimize`.

    Supports every method that :func:`scipy.optimize.minimize` accepts
    (e.g. ``'CG'``, ``'BFGS'``, ``'Newton-CG'``, ``'TNC'``, ``'SLSQP'``).

    When ``method`` requires a gradient (``use_jacobian=True``, the default),
    the Jacobian is supplied automatically via
    :meth:`~empulse.metrics.LogitObjective.logit_loss_gradient`.  For
    derivative-free methods set ``use_jacobian=False``.

    Parameters
    ----------
    method : str, default='L-BFGS-B'
        Optimisation method passed to :func:`scipy.optimize.minimize`.
    max_iter : int, default=1000
        Maximum number of iterations (passed as ``options['maxiter']``).
    tolerance : float or None, default=None
        Solver-specific convergence tolerance passed as the ``tol`` argument.
        ``None`` uses scipy's default per-method tolerance.
    use_jacobian : bool, default=True
        If ``True``, pass the analytic gradient to scipy (``jac=True``).
        Set to ``False`` for derivative-free methods.
    options : dict or None, default=None
        Extra entries merged into the ``options`` dict passed to scipy.
        ``maxiter`` is always set from *max_iter* but can be overridden here.
    **scipy_kwargs
        Additional keyword arguments forwarded verbatim to
        :func:`scipy.optimize.minimize`, e.g. ``bounds=[(min, max), ...]`` for methods that
        support bounds.

    Examples
    --------
    .. code-block:: python

        from empulse.models import CSLogitClassifier
        from empulse.optimizers import ScipyOptimizer

        # Use conjugate gradient
        model = CSLogitClassifier(optimizer=ScipyOptimizer(method='CG', max_iter=500))

        # Use Nelder–Mead (no gradient)
        model = CSLogitClassifier(
            optimizer=ScipyOptimizer(method='Nelder-Mead', use_jacobian=False, max_iter=2000)
        )
    """

    def __init__(
        self,
        method: str = 'L-BFGS-B',
        max_iter: int = 1000,
        tolerance: float | None = None,
        use_jacobian: bool = True,
        options: dict[str, Any] | None = None,
        **scipy_kwargs: Any,
    ) -> None:
        self.method = method
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.use_jacobian = use_jacobian
        self.options = options or {}
        self.scipy_kwargs = scipy_kwargs

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """Run scipy optimisation."""
        initial_weights = self._initial_weights(X)

        merged_options = {'maxiter': self.max_iter, **self.options}

        fun: Any
        jac: Any
        if self.use_jacobian:
            fun = objective.logit_loss_gradient
            jac = True
        else:
            fun = objective.logit_loss
            jac = None

        result: OptimizeResult = minimize(  # type: ignore[call-overload]
            fun,
            initial_weights,
            method=self.method,
            jac=jac,
            tol=self.tolerance,
            options=merged_options,
            **{**self.scipy_kwargs, **kwargs},
        )
        _check_optimize_result(result, self.method)
        return result
