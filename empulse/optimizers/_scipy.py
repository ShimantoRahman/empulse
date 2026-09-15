import warnings
from typing import Any, Literal

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from sklearn.exceptions import ConvergenceWarning

from .._types import FloatNDArray
from ..metrics import ElasticNetPenalty, LogitObjective
from ._base import Optimizer


def _check_optimize_result(result: OptimizeResult, optimizer_name: str = 'scipy') -> None:
    """
    Warn if the optimizer did not converge.

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


def _minimize_split_variable(
    objective: LogitObjective,
    initial_weights: FloatNDArray,
    penalty: ElasticNetPenalty,
    options: dict[str, Any],
) -> OptimizeResult:
    """
    Minimize a non-smooth elastic-net objective exactly, with L-BFGS-B.

    L-BFGS-B assumes a differentiable objective, and an L1 penalty is not differentiable at zero.
    Handing it ``sign(w)`` as a subgradient makes ``w = 0`` look stationary, so the solver stops
    immediately and never produces exact zeros.

    Splitting each penalized coefficient into a positive and a negative part, ``w = u - v`` with
    ``u, v >= 0``, turns ``|w|`` into the *linear* ``u + v``, which is smooth. The non-negativity is
    expressed through the box constraints L-BFGS-B already supports, so the reformulated problem is
    exactly the original one and the solver is used within its assumptions.

    The unpenalized intercept is not split: it stays a single free coordinate, since ``u - v`` would
    otherwise leave it unidentified.

    Parameters
    ----------
    objective : :class:`~empulse.metrics.LogitObjective`
        Objective exposing an unpenalized ``data_loss_gradient``.
    initial_weights : ndarray
        Starting coefficient vector, in the original ``w`` space.
    penalty : :class:`~empulse.metrics.ElasticNetPenalty`
        The penalty to apply.
    options : dict
        Options forwarded to :func:`scipy.optimize.minimize`.

    Returns
    -------
    result : :class:`scipy.optimize.OptimizeResult`
        The optimization result, reported in the original ``w`` space.
    """
    w0 = np.asarray(initial_weights, dtype=np.float64)
    n_params = w0.size
    start = penalty.start_coef
    n_split = n_params - start
    l1_weight = penalty.l1_weight

    def to_weights(z: FloatNDArray) -> FloatNDArray:
        return np.concatenate([z[:start], z[start : start + n_split] - z[start + n_split :]])

    def objective_and_gradient(z: FloatNDArray) -> tuple[float, FloatNDArray]:
        u = z[start : start + n_split]
        v = z[start + n_split :]
        weights = to_weights(z)
        loss, gradient = objective.data_loss_gradient(weights)
        gradient = np.asarray(gradient, dtype=np.float64)

        coef = weights[start:]
        loss += l1_weight * float(np.sum(u) + np.sum(v)) + penalty.l2_value(coef)
        l2_gradient = penalty.l2_gradient(coef)

        grad_u = gradient[start:] + l1_weight + l2_gradient
        grad_v = -gradient[start:] + l1_weight - l2_gradient
        return loss, np.concatenate([gradient[:start], grad_u, grad_v])

    z0 = np.concatenate([w0[:start], np.maximum(w0[start:], 0.0), np.maximum(-w0[start:], 0.0)])
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * start + [(0.0, None)] * (2 * n_split)

    result: OptimizeResult = minimize(  # type: ignore[call-overload]
        objective_and_gradient, z0, method='L-BFGS-B', jac=True, bounds=bounds, options=options
    )

    weights = to_weights(np.asarray(result.x, dtype=np.float64))
    # At a complementary optimum min(u_j, v_j) is zero up to solver tolerance, so a coefficient that
    # should be exactly zero comes back as noise around 1e-13. Snap it, so the reported sparsity is
    # the real one.
    if n_split:
        scale = max(1.0, float(np.max(np.abs(weights))))
        coef = weights[start:]
        coef[np.abs(coef) <= 1e-10 * scale] = 0.0
        weights[start:] = coef

    return OptimizeResult(  # type: ignore[call-arg]
        x=weights,
        fun=objective.logit_loss(weights),
        jac=objective.logit_gradient(weights),
        nit=result.nit,
        nfev=result.nfev,
        njev=getattr(result, 'njev', result.nfev),
        status=result.status,
        success=result.success,
        message=result.message,
    )


class LBFGSBOptimizer(Optimizer):
    """
    Limited-memory BFGS with box constraints (L-BFGS-B) via :func:`scipy.optimize.minimize`.

    This is the default optimizer for :class:`~empulse.models.CSLogitClassifier`.
    It is well-suited for smooth objectives and scales to thousands of features.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of L-BFGS-B iterations.
    tolerance : float, default=1e-4
        Gradient infinity-norm convergence tolerance, *relative to the objective magnitude*.
        The value handed to SciPy as ``gtol`` is this number times the objective scale, so a cost
        matrix expressed in euros and the same one expressed in cents converge to the same model.
    max_line_search_steps : int, default=50
        Maximum number of line-search steps per iteration (``maxls``).
    ftol_scale : float, default=64.0
        Function-value tolerance is set to ``ftol_scale * machine_epsilon``.
    split_variable : bool or 'auto', default='auto'
        Whether to minimize a non-smooth L1 penalty through a split-variable reformulation
        (``w = u - v`` with ``u, v >= 0``), which turns ``|w|`` into a linear term L-BFGS-B handles
        within its box constraints. ``'auto'`` applies it whenever the objective carries an
        elastic-net penalty with ``l1_ratio > 0``, and falls back to plain subgradient descent when
        explicit ``bounds`` are supplied. ``False`` always uses subgradient descent, which stops at
        the kink and does not produce exact zeros; ``True`` requires a non-smooth penalty.

    Notes
    -----
    This is the one place the two SciPy optimizers differ: :class:`ScipyOptimizer` with
    ``method='L-BFGS-B'`` never reformulates, so with ``l1_ratio > 0`` it performs plain subgradient
    descent and will generally report a different (slightly worse, non-sparse) solution.

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
        split_variable: bool | Literal['auto'] = 'auto',
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_line_search_steps = max_line_search_steps
        self.ftol_scale = ftol_scale
        self.split_variable = split_variable

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Run the L-BFGS-B optimization and return the optimization result.

        Parameters
        ----------
        objective : :class:`~empulse.metrics.LogitObjective`
            Prepared objective exposing the loss and gradient of the logit model.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix, used only to size the coefficient vector.
        **kwargs : Any
            Forwarded to the underlying solver.

        Returns
        -------
        result : :class:`scipy.optimize.OptimizeResult`
            The optimization result; ``fun`` is reported as a loss to be minimized.
        """
        initial_weights = self._initial_weights(X)
        penalty = objective.penalty
        # `gtol` bounds the gradient infinity-norm in absolute terms, but a cost-sensitive objective
        # is measured in whatever unit the cost matrix uses, so a fixed value stops early on costs
        # of 1 and late on costs of 1000. Scaling it by the objective magnitude makes convergence --
        # and therefore the selected model -- invariant to rescaling the costs.
        gtol = self.tolerance * (penalty.objective_scale if penalty is not None else 1.0)
        options = {
            'maxiter': self.max_iter,
            'maxls': self.max_line_search_steps,
            'gtol': gtol,
            'ftol': self.ftol_scale * np.finfo(float).eps,
        }

        if self.split_variable is True and (penalty is None or not penalty.is_nonsmooth):
            raise ValueError(
                'split_variable=True requires an objective with a non-smooth (l1_ratio > 0) '
                'elastic-net penalty; got '
                f'{"no penalty" if penalty is None else f"l1_ratio={penalty.l1_ratio}"}.'
            )
        use_split = self.split_variable is not False and penalty is not None and penalty.is_nonsmooth
        if use_split and 'bounds' in kwargs:
            warnings.warn(
                'Explicit bounds cannot be combined with the split-variable reformulation, so the '
                'L1 penalty is minimized by subgradient descent instead. It will not produce exact '
                'zeros and may stop early.',
                UserWarning,
                stacklevel=2,
            )
            use_split = False

        if use_split:
            assert penalty is not None
            result = _minimize_split_variable(objective, initial_weights, penalty, options)
        else:
            result = minimize(  # type: ignore[call-overload]
                objective.logit_loss_gradient,
                initial_weights,
                method='L-BFGS-B',
                jac=True,
                options=options,
                **kwargs,
            )
        _check_optimize_result(result, 'L-BFGS-B')
        return result


class ScipyOptimizer(Optimizer):
    """
    General-purpose wrapper around :func:`scipy.optimize.minimize`.

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

    Notes
    -----
    Unlike :class:`LBFGSBOptimizer`, this optimizer never reformulates a non-smooth penalty: with
    ``l1_ratio > 0`` the objective is minimized by plain subgradient descent, which does not produce
    exact zeros and may stop at the kink. That is what keeps user-supplied ``bounds`` meaningful
    here. Use :class:`LBFGSBOptimizer` when you want sparsity.

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
        """
        Run the SciPy optimization and return the optimization result.

        Parameters
        ----------
        objective : :class:`~empulse.metrics.LogitObjective`
            Prepared objective exposing the loss and gradient of the logit model.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix, used only to size the coefficient vector.
        **kwargs : Any
            Forwarded to the underlying solver.

        Returns
        -------
        result : :class:`scipy.optimize.OptimizeResult`
            The optimization result; ``fun`` is reported as a loss to be minimized.
        """
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
