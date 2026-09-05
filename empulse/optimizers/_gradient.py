from abc import abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from .._types import FloatNDArray
from ..metrics import LogitObjective
from ._base import Optimizer
from ._schedules import Schedule


def _make_result(
    weights: FloatNDArray,
    loss: float,
    gradient: FloatNDArray,
    nit: int,
    nfev: int,
    success: bool,
    message: str,
    status: int,
) -> OptimizeResult:
    return OptimizeResult(  # type: ignore[call-arg]
        x=weights.copy(),
        fun=loss,
        jac=gradient.copy(),
        nit=nit,
        nfev=nfev,
        success=success,
        message=message,
        status=status,
    )


class _IterativeGradientOptimizer(Optimizer):
    """Shared loop logic for iterative gradient-descent optimizers.

    Subclasses implement :meth:`_init_state` and :meth:`_step`.

    Convergence is declared when *either* condition holds:

    * ``||gradient||_inf < tolerance``, or
    * the loss range (max - min) over the last ``patience`` iterations is
      below ``tolerance``.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        patience: int = 20,
        lr_schedule: Schedule | None = None,
        alpha_schedule: Schedule | None = None,
        batch_size: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.patience = patience
        self.lr_schedule = lr_schedule
        self.alpha_schedule = alpha_schedule
        self.batch_size = batch_size
        self.random_state = random_state

    @abstractmethod
    def _init_state(self, weights: FloatNDArray) -> dict[str, Any]:
        """Return the initial optimizer state (moment buffers, etc.)."""

    @abstractmethod
    def _step(
        self,
        weights: FloatNDArray,
        gradient: FloatNDArray,
        state: dict[str, Any],
        t: int,
        lr: float,
    ) -> tuple[FloatNDArray, dict[str, Any]]:
        """Compute the next weight vector and updated state.

        Parameters
        ----------
        weights : ndarray
            Current weight vector.
        gradient : ndarray
            Gradient at *weights*.
        state : dict
            Optimizer state from the previous step.
        t : int
            1-based step counter (used for bias-correction in Adam).
        lr : float
            Effective learning rate for this step (may be overridden by a
            :class:`~empulse.optimizers.BaseSchedule`).

        Returns
        -------
        new_weights : ndarray
        new_state : dict
        """

    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        weights: FloatNDArray = self._initial_weights(X)
        state = self._init_state(weights)

        # Mini-batch setup
        n_samples = X.shape[0]
        rng: np.random.Generator | None = None
        effective_batch = None
        if self.batch_size is not None:
            if not hasattr(objective, 'with_indices'):
                raise TypeError(
                    f'{type(objective).__name__} does not support mini-batch training. '
                    'Override with_indices() to enable it.'
                )
            if isinstance(self.random_state, np.random.Generator):
                rng = self.random_state
            else:
                rng = np.random.default_rng(self.random_state)
            effective_batch = min(self.batch_size, n_samples)

        base_lr: float = getattr(self, 'lr', 1.0)

        loss_history: list[float] = []
        loss: float = np.inf
        gradient: FloatNDArray = np.zeros_like(weights)
        nfev = 0

        # Best-so-far iterate: the loss surface (particularly MaxProfit's) can be rugged and
        # non-convex, so the last iterate visited is not necessarily the best one.
        best_loss = np.inf
        best_weights = weights.copy()
        best_gradient = gradient.copy()

        for t in range(1, self.max_iter + 1):
            # Apply alpha schedule (no-op if objective doesn't support it)
            if self.alpha_schedule is not None and hasattr(objective, 'set_alpha'):
                objective.set_alpha(self.alpha_schedule(t - 1))

            # Compute effective learning rate
            effective_lr = self.lr_schedule(t - 1) if self.lr_schedule is not None else base_lr

            # Select full or mini-batch objective
            if rng is not None and effective_batch is not None:
                indices = rng.choice(n_samples, size=effective_batch, replace=False)
                step_objective = objective.with_indices(indices)
            else:
                step_objective = objective

            loss, gradient = step_objective.logit_loss_gradient(weights)
            nfev += 1
            loss_history.append(float(loss))

            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                best_gradient = gradient.copy()

            # Gradient-norm convergence
            if float(np.max(np.abs(gradient))) < self.tolerance:
                return _make_result(
                    best_weights,
                    best_loss,
                    best_gradient,
                    nit=t,
                    nfev=nfev,
                    success=True,
                    message='Converged: gradient norm below tolerance.',
                    status=0,
                )

            # Loss-plateau convergence (patience window): compare the window's range, not just
            # its endpoints, so an objective that oscillates back to its starting value isn't
            # mistaken for having converged.
            if len(loss_history) > self.patience:
                window = loss_history[-self.patience - 1 :]
                if (max(window) - min(window)) < self.tolerance:
                    return _make_result(
                        best_weights,
                        best_loss,
                        best_gradient,
                        nit=t,
                        nfev=nfev,
                        success=True,
                        message='Converged: loss improvement below tolerance.',
                        status=0,
                    )

            weights, state = self._step(weights, gradient, state, t, effective_lr)

        return _make_result(
            best_weights,
            best_loss,
            best_gradient,
            nit=self.max_iter,
            nfev=nfev,
            success=False,
            message='Maximum number of iterations reached.',
            status=1,
        )


class SGD(_IterativeGradientOptimizer):
    """Stochastic Gradient Descent with optional Nesterov momentum.

    Although called "stochastic", this optimizer operates on the full dataset
    (as required by cost-sensitive logit objectives) and is therefore a
    full-batch gradient descent with SGD-style update rules.

    The update rule without momentum is simply:

    .. math::

        w_{t+1} = w_t - \\text{lr} \\cdot g_t

    With momentum (``momentum > 0``):

    .. math::

        v_t &= \\text{momentum} \\cdot v_{t-1} + (1 - \\text{dampening}) \\cdot g_t \\\\
        w_{t+1} &= w_t - \\text{lr} \\cdot v_t

    With Nesterov momentum (``nesterov=True``):

    .. math::

        w_{t+1} = w_t - \\text{lr} \\cdot (g_t + \\text{momentum} \\cdot v_t)

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate (used when no ``lr_schedule`` is given).
    momentum : float, default=0.0
        Momentum factor.  ``0.0`` disables momentum.
    nesterov : bool, default=False
        If ``True``, use Nesterov momentum (requires ``momentum > 0``).
    dampening : float, default=0.0
        Dampening applied to the velocity update (only used when
        ``momentum > 0`` and ``nesterov=False``).
    lr_schedule : BaseSchedule, optional
        If given, overrides the constant ``lr`` each step.
        ``schedule(t)`` receives the 0-based step index and returns a float.
    alpha_schedule : BaseSchedule, optional
        If given, calls ``objective.set_alpha(schedule(t))`` before each
        gradient computation.  Has no effect on objectives that do not expose
        ``set_alpha``.
    batch_size : int, optional
        Number of samples per gradient step.  ``None`` (default) uses all
        samples.  The objective must support :meth:`with_indices` for
        mini-batching to work.
    random_state : int or numpy.random.Generator, optional
        Seed or random number generator used for mini-batch shuffling.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tolerance : float, default=1e-6
        Convergence tolerance on gradient infinity-norm and loss plateau.
    patience : int, default=20
        Number of consecutive steps with loss improvement smaller than
        *tolerance* before declaring convergence.

    Examples
    --------
    .. code-block:: python

        from empulse.models import CSLogitClassifier
        from empulse.optimizers import SGD, ExponentialSchedule

        lr_schedule = ExponentialSchedule(start_value=0.05, gamma=0.99, min_value=1e-5)
        model = CSLogitClassifier(optimizer=SGD(lr=0.05, momentum=0.9, nesterov=True,
                                                lr_schedule=lr_schedule))
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0,
        lr_schedule: Schedule | None = None,
        alpha_schedule: Schedule | None = None,
        batch_size: int | None = None,
        random_state: int | np.random.Generator | None = None,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        patience: int = 20,
    ) -> None:
        super().__init__(
            max_iter=max_iter,
            tolerance=tolerance,
            patience=patience,
            lr_schedule=lr_schedule,
            alpha_schedule=alpha_schedule,
            batch_size=batch_size,
            random_state=random_state,
        )
        if nesterov and momentum <= 0:
            raise ValueError('Nesterov momentum requires momentum > 0.')
        if dampening != 0.0 and nesterov:
            raise ValueError('Nesterov momentum is incompatible with dampening != 0.')
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening

    def _init_state(self, weights: FloatNDArray) -> dict[str, Any]:
        return {'velocity': np.zeros_like(weights)}

    def _step(
        self,
        weights: FloatNDArray,
        gradient: FloatNDArray,
        state: dict[str, Any],
        t: int,
        lr: float,
    ) -> tuple[FloatNDArray, dict[str, Any]]:
        v = state['velocity']

        if self.momentum > 0:
            v = self.momentum * v + (1.0 - self.dampening) * gradient
            effective_grad = gradient + self.momentum * v if self.nesterov else v
        else:
            effective_grad = gradient

        new_weights = weights - lr * effective_grad
        return new_weights, {'velocity': v}


class RMSProp(_IterativeGradientOptimizer):
    """RMSProp optimizer.

    Divides the learning rate by a running average of recent gradient
    magnitudes:

    .. math::

        v_t &= \\alpha \\cdot v_{t-1} + (1 - \\alpha) \\cdot g_t^2 \\\\
        w_{t+1} &= w_t - \\frac{\\text{lr}}{\\sqrt{v_t} + \\varepsilon} \\cdot g_t

    When ``momentum > 0``, a momentum buffer is added:

    .. math::

        b_t &= \\text{momentum} \\cdot b_{t-1}
              + \\frac{\\text{lr}}{\\sqrt{v_t} + \\varepsilon} \\cdot g_t \\\\
        w_{t+1} &= w_t - b_t

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate (used when no ``lr_schedule`` is given).
    alpha : float, default=0.99
        Smoothing constant for the squared-gradient running average.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    momentum : float, default=0.0
        Momentum factor.  ``0.0`` disables momentum.
    lr_schedule : BaseSchedule, optional
        If given, overrides the constant ``lr`` each step.
    alpha_schedule : BaseSchedule, optional
        If given, calls ``objective.set_alpha(schedule(t))`` before each
        gradient computation.
    batch_size : int, optional
        Number of samples per gradient step.  ``None`` uses all samples.
    random_state : int or numpy.random.Generator, optional
        Seed or random number generator for mini-batch shuffling.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tolerance : float, default=1e-6
        Convergence tolerance.
    patience : int, default=20
        Early-stopping patience (loss plateau window).

    Examples
    --------
    .. code-block:: python

        from empulse.models import CSLogitClassifier
        from empulse.optimizers import RMSProp

        model = CSLogitClassifier(optimizer=RMSProp(lr=0.005, alpha=0.9))
    """

    def __init__(
        self,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        lr_schedule: Schedule | None = None,
        alpha_schedule: Schedule | None = None,
        batch_size: int | None = None,
        random_state: int | np.random.Generator | None = None,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        patience: int = 20,
    ) -> None:
        super().__init__(
            max_iter=max_iter,
            tolerance=tolerance,
            patience=patience,
            lr_schedule=lr_schedule,
            alpha_schedule=alpha_schedule,
            batch_size=batch_size,
            random_state=random_state,
        )
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum

    def _init_state(self, weights: FloatNDArray) -> dict[str, Any]:
        return {
            'v': np.zeros_like(weights),
            'buf': np.zeros_like(weights),
        }

    def _step(
        self,
        weights: FloatNDArray,
        gradient: FloatNDArray,
        state: dict[str, Any],
        t: int,
        lr: float,
    ) -> tuple[FloatNDArray, dict[str, Any]]:
        v = self.alpha * state['v'] + (1.0 - self.alpha) * gradient**2
        scaled_grad = gradient / (np.sqrt(v) + self.eps)

        if self.momentum > 0:
            buf = self.momentum * state['buf'] + lr * scaled_grad
            new_weights = weights - buf
        else:
            buf = state['buf']
            new_weights = weights - lr * scaled_grad

        return new_weights, {'v': v, 'buf': buf}


class Adam(_IterativeGradientOptimizer):
    """Adam optimizer with optional AMSGrad correction.

    Combines momentum (first moment) with adaptive per-parameter learning
    rates (second moment):

    .. math::

        m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat m_t &= m_t / (1 - \\beta_1^t) \\\\
        \\hat v_t &= v_t / (1 - \\beta_2^t) \\\\
        w_{t+1} &= w_t - \\text{lr} \\cdot \\hat m_t / (\\sqrt{\\hat v_t} + \\varepsilon)

    When ``amsgrad=True`` the maximum of all past :math:`\\hat v_t` is used
    instead of :math:`\\hat v_t` (AMSGrad variant).

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate (used when no ``lr_schedule`` is given).
    beta1 : float, default=0.9
        Exponential decay rate for the first moment.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment.
    eps : float, default=1e-8
        Numerical stability term added to the denominator.
    amsgrad : bool, default=False
        If ``True``, use the AMSGrad variant of Adam.
    lr_schedule : BaseSchedule, optional
        If given, overrides the constant ``lr`` each step.
        ``schedule(t)`` receives the 0-based step index and returns a float.
    alpha_schedule : BaseSchedule, optional
        If given, calls ``objective.set_alpha(schedule(t))`` before each
        gradient computation.  Has no effect on objectives that do not expose
        ``set_alpha``.
    batch_size : int, optional
        Number of samples per gradient step.  ``None`` (default) uses all
        samples.  The objective must support :meth:`with_indices` for
        mini-batching to work.
    random_state : int or numpy.random.Generator, optional
        Seed or random number generator used for mini-batch shuffling.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tolerance : float, default=1e-6
        Convergence tolerance.
    patience : int, default=20
        Early-stopping patience (loss plateau window).

    Examples
    --------
    .. code-block:: python

        from empulse.models import CSLogitClassifier
        from empulse.optimizers import Adam, CosineAnnealingSchedule, LinearSchedule

        # Cosine-anneal learning rate and linearly grow alpha
        lr_schedule = CosineAnnealingSchedule(max_value=1e-2, min_value=1e-5, t_max=500)
        alpha_schedule = LinearSchedule(start_value=0.5, end_value=20.0, n_steps=200)
        model = CSLogitClassifier(
            optimizer=Adam(lr=1e-2, lr_schedule=lr_schedule, alpha_schedule=alpha_schedule)
        )
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        lr_schedule: Schedule | None = None,
        alpha_schedule: Schedule | None = None,
        batch_size: int | None = None,
        random_state: int | np.random.Generator | None = None,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        patience: int = 20,
    ) -> None:
        super().__init__(
            max_iter=max_iter,
            tolerance=tolerance,
            patience=patience,
            lr_schedule=lr_schedule,
            alpha_schedule=alpha_schedule,
            batch_size=batch_size,
            random_state=random_state,
        )
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad

    def _init_state(self, weights: FloatNDArray) -> dict[str, Any]:
        return {
            'm': np.zeros_like(weights),
            'v': np.zeros_like(weights),
            'v_max': np.zeros_like(weights),
        }

    def _step(
        self,
        weights: FloatNDArray,
        gradient: FloatNDArray,
        state: dict[str, Any],
        t: int,
        lr: float,
    ) -> tuple[FloatNDArray, dict[str, Any]]:
        m = self.beta1 * state['m'] + (1.0 - self.beta1) * gradient
        v = self.beta2 * state['v'] + (1.0 - self.beta2) * gradient**2

        m_hat = m / (1.0 - self.beta1**t)
        v_hat = v / (1.0 - self.beta2**t)

        if self.amsgrad:
            v_max = np.maximum(state['v_max'], v_hat)
            denom = np.sqrt(v_max) + self.eps
        else:
            v_max = state['v_max']
            denom = np.sqrt(v_hat) + self.eps

        new_weights = weights - lr * m_hat / denom
        return new_weights, {'m': m, 'v': v, 'v_max': v_max}
