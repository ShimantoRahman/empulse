"""Parameter schedules for iterative gradient optimizers.

Schedules map an epoch index (0-based) to a scalar value.  They are used to
vary the learning rate (:attr:`lr_schedule`) or the gradient-smoothing
parameter *alpha* (:attr:`alpha_schedule`) of
:class:`~empulse.optimizers.SGD`, :class:`~empulse.optimizers.Adam`, and
:class:`~empulse.optimizers.RMSProp`.

All schedule objects are **callable** – ``schedule(epoch)`` returns a
``float`` for the given 0-based epoch index.

Examples
--------
.. code-block:: python

    from empulse.optimizers import Adam, ExponentialSchedule, LinearSchedule

    # Exponentially decay the learning rate from 1e-2 to a floor of 1e-5.
    lr_schedule = ExponentialSchedule(start_value=1e-2, gamma=0.99, min_value=1e-5)

    # Linearly grow alpha from 0.5 to 20 over 200 epochs.
    alpha_schedule = LinearSchedule(start_value=0.5, end_value=20.0, n_steps=200)

    optimizer = Adam(lr=1e-3, lr_schedule=lr_schedule, alpha_schedule=alpha_schedule)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class Schedule(ABC):
    """Abstract base class for parameter schedules.

    Subclasses must implement :meth:`__call__`.
    """

    @abstractmethod
    def __call__(self, epoch: int) -> float:
        """Return the scheduled value at *epoch* (0-based).

        Parameters
        ----------
        epoch : int
            Current epoch index, starting from 0.

        Returns
        -------
        float
            Scheduled parameter value.
        """

    def __repr__(self) -> str:
        params = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{type(self).__name__}({params})'


class ConstantSchedule(Schedule):
    """Constant schedule – always returns *value*.

    Parameters
    ----------
    value : float
        The constant value to return at every epoch.

    Examples
    --------
    .. code-block:: python

        from empulse.optimizers import ConstantSchedule

        schedule = ConstantSchedule(0.01)
        assert schedule(0) == schedule(999) == 0.01
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, epoch: int) -> float:
        return float(self.value)


class LinearSchedule(Schedule):
    """Linear interpolation from *start_value* to *end_value* over *n_steps* epochs.

    After *n_steps* epochs the schedule stays at *end_value*.

    .. math::

        v_t = v_0 + \\frac{\\min(t,\\, n-1)}{n-1} \\, (v_{\\text{end}} - v_0)

    Parameters
    ----------
    start_value : float
        Value at epoch 0.
    end_value : float
        Value at epoch ``n_steps - 1`` and beyond.
    n_steps : int
        Number of epochs to interpolate over.  Must be ≥ 2.

    Examples
    --------
    .. code-block:: python

        from empulse.optimizers import LinearSchedule, Adam

        schedule = LinearSchedule(start_value=0.5, end_value=20.0, n_steps=100)
        # alpha grows linearly from 0.5 to 20.0 over 100 epochs
        optimizer = Adam(lr=1e-3, alpha_schedule=schedule)
    """

    def __init__(self, start_value: float, end_value: float, n_steps: int) -> None:
        if n_steps < 2:
            raise ValueError('n_steps must be at least 2.')
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps

    def __call__(self, epoch: int) -> float:
        fraction = min(epoch / (self.n_steps - 1), 1.0)
        return float(self.start_value + fraction * (self.end_value - self.start_value))


class ExponentialSchedule(Schedule):
    r"""Exponential schedule: ``value = start_value * gamma ** epoch``.

    Optionally clipped from below at *min_value*.

    .. math::

        v_t = \max\bigl(v_{\min},\; v_0 \cdot \gamma^t\bigr)

    Parameters
    ----------
    start_value : float
        Value at epoch 0.
    gamma : float
        Multiplicative decay factor per epoch.  Typically ``0 < gamma < 1``
        for decay; ``gamma > 1`` can be used for growth (e.g. alpha annealing).
    min_value : float, default=0.0
        Lower bound on the returned value.

    Examples
    --------
    .. code-block:: python

        from empulse.optimizers import ExponentialSchedule

        # Decaying learning rate
        lr_schedule = ExponentialSchedule(start_value=1e-2, gamma=0.99, min_value=1e-5)

        # Growing alpha schedule (smoothing parameter warm-up)
        alpha_schedule = ExponentialSchedule(start_value=0.1, gamma=1.05, min_value=0.0)
    """

    def __init__(self, start_value: float, gamma: float, min_value: float = 0.0) -> None:
        self.start_value = start_value
        self.gamma = gamma
        self.min_value = min_value

    def __call__(self, epoch: int) -> float:
        try:
            value = self.start_value * (self.gamma**epoch)
        except OverflowError:
            value = self.start_value
        return float(max(self.min_value, value))


class StepSchedule(Schedule):
    r"""Step decay: multiply by *gamma* every *step_size* epochs.

    .. math::

        v_t = \max\bigl(v_{\min},\; v_0 \cdot \gamma^{\lfloor t / s \rfloor}\bigr)

    Parameters
    ----------
    start_value : float
        Value at epoch 0.
    step_size : int
        Number of epochs between each reduction.  Must be at least 1.
    gamma : float, default=0.1
        Multiplicative factor applied at each drop.  Use ``gamma < 1`` for
        decay (LR reduction) or ``gamma > 1`` for growth (alpha warm-up).
    min_value : float, default=0.0
        Lower bound on the returned value.

    Examples
    --------
    .. code-block:: python

        from empulse.optimizers import StepSchedule, SGD

        # Halve the learning rate every 100 epochs
        lr_schedule = StepSchedule(start_value=1e-2, step_size=100, gamma=0.5)
        optimizer = SGD(lr=1e-2, lr_schedule=lr_schedule)
    """

    def __init__(
        self,
        start_value: float,
        step_size: int,
        gamma: float = 0.1,
        min_value: float = 0.0,
    ) -> None:
        if step_size < 1:
            raise ValueError('step_size must be at least 1.')
        self.start_value = start_value
        self.step_size = step_size
        self.gamma = gamma
        self.min_value = min_value

    def __call__(self, epoch: int) -> float:
        value = self.start_value * (self.gamma ** (epoch // self.step_size))
        return float(max(self.min_value, value))


class CosineAnnealingSchedule(Schedule):
    r"""Cosine annealing between *max_value* and *min_value* over *t_max* epochs.

    .. math::

        v_t = v_{\min} + \tfrac{1}{2}(v_{\max} - v_{\min})
              \bigl(1 + \cos(\pi\, t / T_{\max})\bigr)

    When ``cycle=True`` the schedule restarts after *t_max* epochs.

    Parameters
    ----------
    max_value : float
        Value at epoch 0 (and after restarts when ``cycle=True``).
    min_value : float
        Value reached at epoch *t_max*.
    t_max : int
        Half-period of the cosine curve (epochs from max to min).  Must be at least 1.
    cycle : bool, default=False
        If ``True``, restart the annealing cycle after *t_max* epochs.

    Examples
    --------
    .. code-block:: python

        from empulse.optimizers import CosineAnnealingSchedule, Adam

        # Cosine-anneal LR from 1e-2 to 1e-6 over 500 epochs, then restart
        lr_schedule = CosineAnnealingSchedule(
            max_value=1e-2, min_value=1e-6, t_max=500, cycle=True
        )
        optimizer = Adam(lr=1e-2, lr_schedule=lr_schedule)
    """

    def __init__(
        self,
        max_value: float,
        min_value: float,
        t_max: int,
        cycle: bool = False,
    ) -> None:
        if t_max < 1:
            raise ValueError('t_max must be at least 1.')
        self.max_value = max_value
        self.min_value = min_value
        self.t_max = t_max
        self.cycle = cycle

    def __call__(self, epoch: int) -> float:
        t = (epoch % self.t_max) if self.cycle else min(epoch, self.t_max)
        cos_val = math.cos(math.pi * t / self.t_max)
        return float(self.min_value + 0.5 * (self.max_value - self.min_value) * (1.0 + cos_val))


class WarmupSchedule(Schedule):
    """Linear warm-up for *warmup_steps* epochs, then delegates to *after_schedule*.

    Let ``target = after_schedule(0)``. During warm-up (epochs ``0`` to ``warmup_steps - 1``),
    the value increases linearly from ``target / warmup_steps`` at epoch 0 to the full *target*
    at epoch ``warmup_steps - 1``. From epoch ``warmup_steps`` onward, *after_schedule* is called
    with the *shifted* epoch (``epoch - warmup_steps``), so epoch ``warmup_steps`` itself also maps
    to ``after_schedule(0)`` - the same value the warm-up just reached, so the schedule holds
    steady across the boundary rather than jumping.

    Parameters
    ----------
    warmup_steps : int
        Number of epochs for the linear warm-up phase.  Must be at least 1.
    after_schedule : Schedule
        Schedule to use after the warm-up.  Its 0-based epoch counter restarts
        at the end of the warm-up.

    Examples
    --------
    .. code-block:: python

        from empulse.optimizers import WarmupSchedule, CosineAnnealingSchedule

        cosine = CosineAnnealingSchedule(max_value=1e-2, min_value=1e-5, t_max=400)
        schedule = WarmupSchedule(warmup_steps=50, after_schedule=cosine)
        # For epochs 0..49: linearly warm up to 1e-2; then cosine-anneal.
    """

    def __init__(self, warmup_steps: int, after_schedule: Schedule) -> None:
        if warmup_steps < 1:
            raise ValueError('warmup_steps must be at least 1.')
        self.warmup_steps = warmup_steps
        self.after_schedule = after_schedule

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_steps:
            target = self.after_schedule(0)
            return float(target * (epoch + 1) / self.warmup_steps)
        return float(self.after_schedule(epoch - self.warmup_steps))
