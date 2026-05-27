import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from empulse.optimizers import (
    SGD,
    Adam,
    ConstantSchedule,
    CosineAnnealingSchedule,
    ExponentialSchedule,
    LinearSchedule,
    RMSProp,
    Schedule,
    StepSchedule,
    WarmupSchedule,
)


class _QuadraticObjective:
    """loss = 0.5 * sum(w**2), gradient = w.  Minimum at w = 0.

    Only used for tests that intentionally need the minimum at the origin
    (e.g. max-iter failure tests where tolerance=0 prevents early exit).
    Most tests should prefer _ShiftedQuadraticObjective so the optimizer
    actually has to perform real gradient steps.
    """

    def __init__(self, n_features: int = 4) -> None:
        self.n_features = n_features
        self._alpha: float | None = None

    def logit_loss_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.5 * float(np.dot(weights, weights))
        grad = weights.copy()
        return loss, grad

    def logit_gradient(self, weights: np.ndarray) -> np.ndarray:
        return weights.copy()

    def logit_loss(self, weights: np.ndarray) -> float:
        return 0.5 * float(np.dot(weights, weights))

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha

    def with_indices(self, indices: np.ndarray) -> '_QuadraticObjective':
        clone = _QuadraticObjective(self.n_features)
        clone._alpha = self._alpha
        return clone


class _ShiftedQuadraticObjective:
    """loss = 0.5 * sum((w - 1)^2), gradient = w - 1.  Minimum at w = ones.

    Because weights are initialised to zero, the optimizer always starts with
    a non-zero gradient (-ones) and must actually perform gradient steps to
    reach the minimum.  Use this class in any test that should verify the
    optimizer does real work.
    """

    def __init__(self, n_features: int = 4) -> None:
        self.n_features = n_features
        self._alpha: float | None = None
        self.target = np.ones(n_features)

    def logit_loss_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray]:
        diff = weights - self.target
        loss = 0.5 * float(np.dot(diff, diff))
        return loss, diff.copy()

    def logit_gradient(self, weights: np.ndarray) -> np.ndarray:
        return (weights - self.target).copy()

    def logit_loss(self, weights: np.ndarray) -> float:
        diff = weights - self.target
        return 0.5 * float(np.dot(diff, diff))

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha

    def with_indices(self, indices: np.ndarray) -> '_ShiftedQuadraticObjective':
        clone = _ShiftedQuadraticObjective(self.n_features)
        clone._alpha = self._alpha
        return clone


class _DataObjective:
    """Least-squares objective: loss = 0.5 * mean((X @ w - 1)^2).

    - Minimum is at the OLS solution (not at zero).
    - Starting from w=0 the gradient = -mean(X_batch, axis=0) ≠ 0.
    - Different mini-batches produce different gradient directions, so two runs
      with different random seeds will diverge.
    """

    def __init__(self, X_data: np.ndarray) -> None:
        self._X = X_data
        self._y = np.ones(len(X_data))

    def logit_loss_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray]:
        residuals = self._X @ weights - self._y
        loss = 0.5 * float(np.mean(residuals**2))
        grad = self._X.T @ residuals / len(self._X)
        return loss, grad

    def with_indices(self, indices: np.ndarray) -> '_DataObjective':
        return _DataObjective(self._X[indices])


class _AlphaTrackingObjective(_ShiftedQuadraticObjective):
    """Records every alpha value received via set_alpha.

    Replaces the repeated inner _TrackingObjective pattern across alpha tests.
    """

    def __init__(self, n_features: int = 4) -> None:
        super().__init__(n_features)
        self.alphas_received: list[float] = []

    def set_alpha(self, alpha: float) -> None:
        self.alphas_received.append(alpha)


def _make_X(n_samples: int = 20, n_features: int = 4) -> np.ndarray:
    """Reproducible random feature matrix."""
    return np.random.default_rng(0).standard_normal((n_samples, n_features))


@pytest.fixture(scope='module')
def X() -> np.ndarray:
    return _make_X()


class TestConstantSchedule:
    def test_returns_constant(self):
        s = ConstantSchedule(0.05)
        assert s(0) == pytest.approx(0.05)
        assert s(999) == pytest.approx(0.05)

    def test_repr(self):
        assert 'ConstantSchedule' in repr(ConstantSchedule(1.0))

    def test_is_base_schedule(self):
        assert isinstance(ConstantSchedule(1.0), Schedule)


class TestLinearSchedule:
    def test_start(self):
        s = LinearSchedule(start_value=0.0, end_value=1.0, n_steps=11)
        assert s(0) == pytest.approx(0.0)

    def test_end(self):
        s = LinearSchedule(start_value=0.0, end_value=1.0, n_steps=11)
        assert s(10) == pytest.approx(1.0)

    def test_midpoint(self):
        s = LinearSchedule(start_value=0.0, end_value=2.0, n_steps=3)
        assert s(1) == pytest.approx(1.0)

    def test_clamps_after_n_steps(self):
        s = LinearSchedule(start_value=0.0, end_value=1.0, n_steps=5)
        assert s(100) == pytest.approx(1.0)

    def test_n_steps_too_small_raises(self):
        with pytest.raises(ValueError):
            LinearSchedule(0.0, 1.0, n_steps=1)

    def test_n_steps_2_is_minimum_valid(self):
        """n_steps=2 is the smallest legal value and should interpolate correctly."""
        s = LinearSchedule(start_value=0.0, end_value=1.0, n_steps=2)
        assert s(0) == pytest.approx(0.0)
        assert s(1) == pytest.approx(1.0)

    def test_descending(self):
        """end_value < start_value should decay correctly."""
        s = LinearSchedule(start_value=1.0, end_value=0.0, n_steps=3)
        assert s(0) == pytest.approx(1.0)
        assert s(1) == pytest.approx(0.5)
        assert s(2) == pytest.approx(0.0)

    def test_repr(self):
        assert 'LinearSchedule' in repr(LinearSchedule(0.0, 1.0, n_steps=10))


class TestExponentialSchedule:
    def test_epoch_zero(self):
        s = ExponentialSchedule(start_value=1.0, gamma=0.5)
        assert s(0) == pytest.approx(1.0)

    def test_decay(self):
        s = ExponentialSchedule(start_value=1.0, gamma=0.5)
        assert s(2) == pytest.approx(0.25)

    def test_min_value_floor(self):
        s = ExponentialSchedule(start_value=1.0, gamma=0.1, min_value=0.05)
        assert s(100) == pytest.approx(0.05)

    def test_growth(self):
        s = ExponentialSchedule(start_value=1.0, gamma=2.0)
        assert s(3) == pytest.approx(8.0)

    def test_overflow_returns_finite_value(self):
        """OverflowError from a huge gamma should be caught; result must be finite."""
        s = ExponentialSchedule(start_value=1.0, gamma=1e308, min_value=0.0)
        assert np.isfinite(s(1000))

    def test_repr(self):
        assert 'ExponentialSchedule' in repr(ExponentialSchedule(1.0, 0.5))


class TestStepSchedule:
    def test_no_drop_before_step_size(self):
        s = StepSchedule(start_value=1.0, step_size=10, gamma=0.5)
        assert s(0) == pytest.approx(1.0)
        assert s(9) == pytest.approx(1.0)

    def test_drops_at_step_size(self):
        s = StepSchedule(start_value=1.0, step_size=10, gamma=0.5)
        assert s(10) == pytest.approx(0.5)

    def test_two_drops(self):
        s = StepSchedule(start_value=1.0, step_size=10, gamma=0.5)
        assert s(20) == pytest.approx(0.25)

    def test_min_value_floor(self):
        s = StepSchedule(start_value=1.0, step_size=1, gamma=0.5, min_value=0.1)
        assert s(1000) == pytest.approx(0.1)

    def test_repr(self):
        assert 'StepSchedule' in repr(StepSchedule(1.0, step_size=10, gamma=0.5))


class TestCosineAnnealingSchedule:
    def test_starts_at_max(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=100)
        assert s(0) == pytest.approx(1.0)

    def test_ends_at_min(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=100)
        assert s(100) == pytest.approx(0.0, abs=1e-10)

    def test_midpoint(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=100)
        assert s(50) == pytest.approx(0.5)

    def test_nonzero_min_midpoint(self):
        """Midpoint should sit halfway between min_value and max_value."""
        s = CosineAnnealingSchedule(max_value=2.0, min_value=1.0, t_max=100)
        assert s(50) == pytest.approx(1.5)

    def test_cycle_restarts_to_max(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=10, cycle=True)
        assert s(10) == pytest.approx(s(0))

    def test_cycle_second_period_matches_first(self):
        """Values in the second cycle must be identical to the first."""
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=10, cycle=True)
        assert s(15) == pytest.approx(s(5))

    def test_no_cycle_clamps_at_min(self):
        s = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=10, cycle=False)
        assert s(1000) == pytest.approx(0.0, abs=1e-10)

    def test_repr(self):
        assert 'CosineAnnealingSchedule' in repr(CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=100))


class TestWarmupSchedule:
    def test_first_epoch_is_fraction(self):
        after = ConstantSchedule(1.0)
        s = WarmupSchedule(warmup_steps=10, after_schedule=after)
        assert s(0) == pytest.approx(0.1)

    def test_last_warmup_epoch_reaches_target(self):
        after = ConstantSchedule(1.0)
        s = WarmupSchedule(warmup_steps=10, after_schedule=after)
        assert s(9) == pytest.approx(1.0)

    def test_delegates_after_warmup(self):
        after = ConstantSchedule(0.5)
        s = WarmupSchedule(warmup_steps=5, after_schedule=after)
        assert s(5) == pytest.approx(0.5)
        assert s(10) == pytest.approx(0.5)

    def test_warmup_then_cosine_at_boundary(self):
        cosine = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=10)
        s = WarmupSchedule(warmup_steps=5, after_schedule=cosine)
        # epoch 5 → cosine(0) = 1.0
        assert s(5) == pytest.approx(1.0)

    def test_epoch_shift_after_warmup(self):
        """After warmup the inner schedule's epoch counter restarts at 0."""
        cosine = CosineAnnealingSchedule(max_value=1.0, min_value=0.0, t_max=10)
        s = WarmupSchedule(warmup_steps=5, after_schedule=cosine)
        # epoch 6 → cosine(1), epoch 10 → cosine(5)
        assert s(6) == pytest.approx(cosine(1))
        assert s(10) == pytest.approx(cosine(5))

    def test_repr(self):
        assert 'WarmupSchedule' in repr(WarmupSchedule(warmup_steps=10, after_schedule=ConstantSchedule(1.0)))


@pytest.mark.parametrize(
    'optimizer_cls,kwargs',
    [
        (SGD, {'lr': 0.1, 'max_iter': 2000, 'tolerance': 1e-4}),
        (SGD, {'lr': 0.1, 'momentum': 0.9, 'max_iter': 2000, 'tolerance': 1e-4}),
        (SGD, {'lr': 0.05, 'momentum': 0.9, 'nesterov': True, 'max_iter': 2000, 'tolerance': 1e-4}),
        (RMSProp, {'lr': 0.05, 'max_iter': 2000, 'tolerance': 1e-4}),
        (RMSProp, {'lr': 0.05, 'momentum': 0.5, 'max_iter': 2000, 'tolerance': 1e-4}),
        (Adam, {'lr': 0.05, 'max_iter': 2000, 'tolerance': 1e-4}),
        (Adam, {'lr': 0.05, 'amsgrad': True, 'max_iter': 2000, 'tolerance': 1e-4}),
    ],
)
def test_optimizer_converges_quadratic(optimizer_cls, kwargs, X):
    # _ShiftedQuadraticObjective: minimum at ones, NOT at zeros.
    # Weights initialise to zeros so the optimizer must do real gradient steps.
    obj = _ShiftedQuadraticObjective(n_features=X.shape[1])
    opt = optimizer_cls(**kwargs)
    result = opt(obj, X)
    assert isinstance(result, OptimizeResult)
    dist = np.linalg.norm(result.x - np.ones(X.shape[1]))
    assert dist < 0.1, f'{optimizer_cls.__name__} did not converge: dist = {dist:.4f}'


class TestSGDValidation:
    def test_nesterov_requires_momentum(self):
        with pytest.raises(ValueError, match='momentum'):
            SGD(momentum=0.0, nesterov=True)

    def test_dampening_incompatible_with_nesterov(self):
        with pytest.raises(ValueError, match='dampening'):
            SGD(momentum=0.9, nesterov=True, dampening=0.1)


class TestLRScheduleIntegration:
    def test_constant_schedule_same_as_plain_lr(self, X):
        """ConstantSchedule(lr) should produce identical weights to using no schedule."""
        obj1 = _ShiftedQuadraticObjective()
        obj2 = _ShiftedQuadraticObjective()
        opt_plain = SGD(lr=0.1, max_iter=50, tolerance=1e-9, patience=9999)
        opt_sched = SGD(lr=0.1, max_iter=50, tolerance=1e-9, patience=9999, lr_schedule=ConstantSchedule(0.1))
        r1 = opt_plain(obj1, X)
        r2 = opt_sched(obj2, X)
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-8)

    def test_tiny_lr_schedule_barely_moves_weights(self, X):
        """A schedule returning ≈ 0 should leave weights near the initial zeros."""
        obj = _ShiftedQuadraticObjective()
        opt = SGD(lr=0.5, max_iter=10, tolerance=1e-12, patience=9999, lr_schedule=ConstantSchedule(1e-8))
        result = opt(obj, X)
        assert np.linalg.norm(result.x) < 1e-4

    def test_exponential_decay_runs_without_error(self, X):
        obj = _ShiftedQuadraticObjective()
        schedule = ExponentialSchedule(start_value=0.5, gamma=0.98, min_value=1e-4)
        opt = SGD(lr=0.5, max_iter=200, tolerance=1e-4, lr_schedule=schedule)
        result = opt(obj, X)
        assert result.nit > 0

    def test_cosine_annealing_with_adam(self, X):
        obj = _ShiftedQuadraticObjective()
        schedule = CosineAnnealingSchedule(max_value=0.1, min_value=1e-5, t_max=500)
        opt = Adam(lr=0.1, max_iter=2000, tolerance=1e-4, lr_schedule=schedule)
        result = opt(obj, X)
        assert np.linalg.norm(result.x - np.ones(X.shape[1])) < 0.1

    def test_warmup_cosine_with_adam(self, X):
        obj = _ShiftedQuadraticObjective()
        cosine = CosineAnnealingSchedule(max_value=0.05, min_value=1e-5, t_max=400)
        schedule = WarmupSchedule(warmup_steps=50, after_schedule=cosine)
        opt = Adam(lr=0.05, max_iter=2000, tolerance=1e-4, lr_schedule=schedule)
        result = opt(obj, X)
        assert np.linalg.norm(result.x - np.ones(X.shape[1])) < 0.1

    def test_step_decay_with_rmsprop(self, X):
        obj = _ShiftedQuadraticObjective()
        schedule = StepSchedule(start_value=0.05, step_size=100, gamma=0.5, min_value=1e-5)
        opt = RMSProp(lr=0.05, max_iter=2000, tolerance=1e-4, lr_schedule=schedule)
        result = opt(obj, X)
        assert np.linalg.norm(result.x - np.ones(X.shape[1])) < 0.1

    def test_lr_schedule_receives_correct_epoch_indices(self, X):
        """The schedule must be called with 0-based epoch indices 0, 1, 2, … in order."""
        epochs_seen: list[int] = []

        class _SpySchedule(Schedule):
            def __call__(self, epoch: int) -> float:
                epochs_seen.append(epoch)
                return 0.1

        obj = _ShiftedQuadraticObjective()
        n_steps = 8
        # tolerance=0 and patience=9999 prevent early exit so all n_steps run.
        opt = SGD(lr=0.1, max_iter=n_steps, tolerance=0.0, patience=9999, lr_schedule=_SpySchedule())
        opt(obj, X)
        assert epochs_seen == list(range(n_steps))


class TestAlphaScheduleIntegration:
    def test_set_alpha_called_each_step(self, X):
        obj = _AlphaTrackingObjective()
        schedule = LinearSchedule(start_value=1.0, end_value=10.0, n_steps=11)
        opt = SGD(lr=0.1, max_iter=10, tolerance=1e-12, patience=9999, alpha_schedule=schedule)
        opt(obj, X)
        assert len(obj.alphas_received) == 10

    def test_set_alpha_first_value_is_epoch_0(self, X):
        obj = _AlphaTrackingObjective()
        schedule = LinearSchedule(start_value=1.0, end_value=10.0, n_steps=11)
        opt = SGD(lr=0.1, max_iter=10, tolerance=1e-12, patience=9999, alpha_schedule=schedule)
        opt(obj, X)
        assert obj.alphas_received[0] == pytest.approx(schedule(0))

    def test_alpha_schedule_exponential_values(self, X):
        obj = _AlphaTrackingObjective()
        schedule = ExponentialSchedule(start_value=0.5, gamma=2.0, min_value=0.0)
        opt = SGD(lr=0.1, max_iter=5, tolerance=1e-12, patience=9999, alpha_schedule=schedule)
        opt(obj, X)
        expected = [0.5 * (2.0**i) for i in range(5)]
        for got, exp in zip(obj.alphas_received, expected, strict=True):
            assert got == pytest.approx(exp)

    def test_alpha_schedule_with_adam(self, X):
        """Alpha schedule should work with Adam, not just SGD."""
        obj = _AlphaTrackingObjective()
        opt = Adam(lr=0.05, max_iter=6, tolerance=1e-12, patience=9999, alpha_schedule=ConstantSchedule(5.0))
        opt(obj, X)
        assert len(obj.alphas_received) == 6
        assert all(v == pytest.approx(5.0) for v in obj.alphas_received)

    def test_alpha_schedule_with_rmsprop(self, X):
        """Alpha schedule should work with RMSProp, not just SGD."""
        obj = _AlphaTrackingObjective()
        opt = RMSProp(lr=0.05, max_iter=4, tolerance=1e-12, patience=9999, alpha_schedule=ConstantSchedule(3.0))
        opt(obj, X)
        assert len(obj.alphas_received) == 4
        assert all(v == pytest.approx(3.0) for v in obj.alphas_received)

    def test_alpha_schedule_ignored_if_no_set_alpha(self, X):
        """Objectives without set_alpha should not raise."""

        class _NoAlphaObjective:
            def logit_loss_gradient(self, weights):
                diff = weights - np.ones_like(weights)
                return 0.5 * float(np.dot(diff, diff)), diff.copy()

            def with_indices(self, indices):
                return _NoAlphaObjective()

        obj = _NoAlphaObjective()
        opt = SGD(lr=0.1, max_iter=50, tolerance=1e-4, alpha_schedule=ConstantSchedule(5.0))
        result = opt(obj, X)
        assert result.nit > 0


class TestMiniBatch:
    def test_batch_size_none_uses_full_dataset(self, X):
        obj = _ShiftedQuadraticObjective()
        opt = SGD(lr=0.1, max_iter=200, tolerance=1e-4, batch_size=None)
        result = opt(obj, X)
        assert np.linalg.norm(result.x - np.ones(X.shape[1])) < 0.1

    def test_with_indices_called_each_step(self, X):
        call_count = [0]

        class _TrackingObjective(_ShiftedQuadraticObjective):
            def with_indices(self, indices: np.ndarray) -> _ShiftedQuadraticObjective:
                call_count[0] += 1
                return super().with_indices(indices)

        obj = _TrackingObjective()
        n_steps = 10
        opt = SGD(lr=0.1, max_iter=n_steps, tolerance=1e-12, patience=9999, batch_size=8, random_state=0)
        opt(obj, X)
        assert call_count[0] == n_steps

    def test_batch_indices_correct_size(self, X):
        batch_sizes_seen: list[int] = []

        class _TrackingObjective(_QuadraticObjective):
            def with_indices(self, indices: np.ndarray) -> _QuadraticObjective:
                batch_sizes_seen.append(len(indices))
                return super().with_indices(indices)

        obj = _TrackingObjective()
        opt = SGD(lr=0.1, max_iter=10, tolerance=1e-12, patience=9999, batch_size=8, random_state=0)
        opt(obj, X)
        assert all(s == 8 for s in batch_sizes_seen)

    def test_batch_indices_are_unique(self, X):
        """Each mini-batch must sample without replacement (no duplicate indices)."""
        all_indices: list[np.ndarray] = []

        class _TrackingObjective(_QuadraticObjective):
            def with_indices(self, indices: np.ndarray) -> _QuadraticObjective:
                all_indices.append(indices.copy())
                return super().with_indices(indices)

        obj = _TrackingObjective()
        opt = SGD(lr=0.1, max_iter=10, tolerance=1e-12, patience=9999, batch_size=8, random_state=0)
        opt(obj, X)
        for indices in all_indices:
            assert len(set(indices)) == len(indices), 'Duplicate indices in mini-batch'

    def test_batch_size_capped_at_n_samples(self, X):
        batch_sizes_seen: list[int] = []

        class _TrackingObjective(_QuadraticObjective):
            def with_indices(self, indices: np.ndarray) -> _QuadraticObjective:
                batch_sizes_seen.append(len(indices))
                return super().with_indices(indices)

        obj = _TrackingObjective()
        opt = SGD(lr=0.1, max_iter=5, tolerance=1e-12, patience=9999, batch_size=999, random_state=0)
        opt(obj, X)
        assert all(s == X.shape[0] for s in batch_sizes_seen)

    def test_minibatch_raises_without_with_indices(self, X):
        class _NoWithIndicesObjective:
            def logit_loss_gradient(self, weights):
                diff = weights - np.ones_like(weights)
                return 0.5 * float(np.dot(diff, diff)), diff.copy()

        obj = _NoWithIndicesObjective()
        opt = SGD(lr=0.1, max_iter=10, batch_size=5)
        with pytest.raises(TypeError, match='with_indices'):
            opt(obj, X)

    def test_same_random_state_gives_identical_weights(self, X):
        obj1 = _QuadraticObjective()
        obj2 = _QuadraticObjective()
        kwargs: dict = {
            'lr': 0.1,
            'max_iter': 30,
            'tolerance': 1e-12,
            'patience': 9999,
            'batch_size': 8,
            'random_state': 42,
        }
        r1 = SGD(**kwargs)(obj1, X)
        r2 = SGD(**kwargs)(obj2, X)
        np.testing.assert_array_equal(r1.x, r2.x)

    def test_different_random_states_give_different_weights(self, X):
        # _DataObjective slices real data per batch so different seeds diverge.
        obj1 = _DataObjective(X)
        obj2 = _DataObjective(X)
        r1 = SGD(lr=0.05, max_iter=30, tolerance=1e-12, patience=9999, batch_size=5, random_state=0)(obj1, X)
        r2 = SGD(lr=0.05, max_iter=30, tolerance=1e-12, patience=9999, batch_size=5, random_state=99)(obj2, X)
        assert not np.array_equal(r1.x, r2.x)

    def test_minibatch_adam_converges(self, X):
        obj = _ShiftedQuadraticObjective()
        opt = Adam(lr=0.05, max_iter=2000, tolerance=1e-4, batch_size=10, random_state=7)
        result = opt(obj, X)
        assert np.linalg.norm(result.x - np.ones(X.shape[1])) < 0.2

    def test_generator_random_state_accepted(self, X):
        """numpy Generator instance should be accepted as random_state."""
        obj = _QuadraticObjective()
        rng = np.random.default_rng(123)
        opt = SGD(lr=0.1, max_iter=10, tolerance=1e-12, patience=9999, batch_size=8, random_state=rng)
        result = opt(obj, X)
        assert result.nit > 0


class TestOptimizeResult:
    def test_result_has_expected_fields(self, X):
        obj = _ShiftedQuadraticObjective()
        result = Adam(lr=0.05, max_iter=500, tolerance=1e-5)(obj, X)
        for field in ('x', 'fun', 'jac', 'nit', 'nfev', 'success', 'message', 'status'):
            assert hasattr(result, field), f'Missing field: {field}'

    def test_result_fun_and_jac_are_correct(self, X):
        """result.fun and result.jac must match the objective evaluated at result.x."""
        obj = _ShiftedQuadraticObjective(n_features=X.shape[1])
        # Converge fully so the convergence-path return is used (not the max-iter
        # path where weights are updated after the last loss/grad computation).
        result = SGD(lr=0.1, max_iter=1000, tolerance=1e-6)(obj, X)
        assert result.success is True
        assert result.fun == pytest.approx(obj.logit_loss(result.x), rel=1e-5)
        np.testing.assert_allclose(result.jac, obj.logit_gradient(result.x), rtol=1e-5)

    def test_success_is_bool(self, X):
        obj = _ShiftedQuadraticObjective()
        result = SGD(lr=0.1, max_iter=200, tolerance=1e-4)(obj, X)
        assert isinstance(result.success, bool)

    def test_message_is_str(self, X):
        obj = _ShiftedQuadraticObjective()
        result = SGD(lr=0.1, max_iter=200, tolerance=1e-4)(obj, X)
        assert isinstance(result.message, str)

    def test_status_on_success(self, X):
        obj = _ShiftedQuadraticObjective()
        result = SGD(lr=0.9, max_iter=5000, tolerance=1e-4)(obj, X)
        assert result.success is True
        assert result.status == 0

    def test_status_on_failure(self, X):
        obj = _ShiftedQuadraticObjective()
        result = SGD(lr=0.01, max_iter=3, tolerance=0.0, patience=9999)(obj, X)
        assert result.success is False
        assert result.status == 1

    def test_max_iter_reports_failure(self, X):
        """With tolerance=0 and very few iterations the run must exhaust max_iter."""
        obj = _ShiftedQuadraticObjective()
        result = SGD(lr=0.01, max_iter=3, tolerance=0.0, patience=9999)(obj, X)
        assert result.nit == 3
        assert result.success is False

    def test_nfev_equals_nit(self, X):
        """Every iteration must call logit_loss_gradient exactly once."""
        obj = _ShiftedQuadraticObjective()
        # tolerance=0 and patience=9999 prevent early exit so all 50 steps run.
        result = SGD(lr=0.1, max_iter=50, tolerance=0.0, patience=9999)(obj, X)
        assert result.nfev == result.nit == 50

    def test_convergence_by_gradient_norm(self, X):
        """Gradient-norm convergence should set success=True with matching message."""
        obj = _ShiftedQuadraticObjective()
        # lr=0.9: ||gradient||_inf halves every step, reaching 1e-4 in ~40 steps.
        result = SGD(lr=0.9, max_iter=5000, tolerance=1e-4)(obj, X)
        assert result.success is True
        assert 'gradient' in result.message.lower()

    def test_convergence_by_loss_plateau(self, X):
        """Loss-plateau (patience) convergence path should set success=True.

        With lr=0.1 and patience=5 the loss improvement over 5 consecutive steps
        drops below 1e-6 at around step 71, well before the gradient norm reaches
        1e-6 (~131 steps) – so the plateau path fires first.
        """
        obj = _ShiftedQuadraticObjective()
        result = SGD(lr=0.1, max_iter=500, tolerance=1e-6, patience=5)(obj, X)
        assert result.success is True
        assert 'loss' in result.message.lower() or 'improvement' in result.message.lower()

    def test_weights_shape_matches_n_features(self):
        obj = _ShiftedQuadraticObjective(n_features=6)
        result = Adam(lr=0.05, max_iter=100)(obj, _make_X(n_features=6))
        assert result.x.shape == (6,)
