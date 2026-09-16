from abc import ABC
from collections.abc import Generator
from dataclasses import dataclass, replace

import numpy as np

from .._types import Float64Array, FloatNDArray


def objective_scale_from_costs(
    y_true: FloatNDArray,
    tp_benefit: FloatNDArray | float,
    tn_benefit: FloatNDArray | float,
    fp_cost: FloatNDArray | float,
    fn_cost: FloatNDArray | float,
) -> float:
    """
    Compute the natural magnitude of a cost matrix's per-sample training signal.

    This is the mean absolute derivative of the expected cost with respect to the predicted
    probability, ``mean_i |c1_i - c2_i|``, where ``c1`` is the cost of predicting positive and
    ``c2`` the cost of predicting negative. It is what the elastic-net penalty is scaled against,
    so that a given ``C`` selects the same model whether costs are expressed in euros or in cents.

    The same quantity is the right scale for all three logit strategies: it is exactly
    :class:`~empulse.metrics.Cost`'s per-sample gradient factor, it is
    :class:`~empulse.metrics.LogCost`'s gradient magnitude at ``s = 0.5`` up to a factor of two
    (and equals ``1.0`` for plain log loss, so ``C`` then means what it means in scikit-learn's
    :class:`~sklearn:sklearn.linear_model.LogisticRegression`), and for
    :class:`~empulse.metrics.MaxProfit` it equals ``|coeff_tpr| + |coeff_fpr|``.

    Parameters
    ----------
    y_true : ndarray
        Binary labels, recoded to 0/1.
    tp_benefit : float or ndarray
        Benefit of true positives (the negation of their cost).
    tn_benefit : float or ndarray
        Benefit of true negatives (the negation of their cost).
    fp_cost : float or ndarray
        Cost of false positives.
    fn_cost : float or ndarray
        Cost of false negatives.

    Returns
    -------
    scale : float
        A strictly positive, finite scale. Falls back to ``1.0`` when the cost matrix carries no
        training signal at all, so that the penalty never collapses to zero or ``NaN``;
        ``empulse.metrics.metric.strategies._training_signal.warn_if_no_training_signal`` is what
        reports that case, from the objectives that actually differentiate the cost matrix.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    # np.asarray keeps scalars as 0-d arrays, which broadcast against `y` exactly as a float would.
    tp_b = np.asarray(tp_benefit, dtype=np.float64).reshape(-1)
    tn_b = np.asarray(tn_benefit, dtype=np.float64).reshape(-1)
    fp_c = np.asarray(fp_cost, dtype=np.float64).reshape(-1)
    fn_c = np.asarray(fn_cost, dtype=np.float64).reshape(-1)

    per_sample = y * (tp_b + fn_c) - (1 - y) * (fp_c + tn_b)
    scale = float(np.mean(np.abs(per_sample)))
    if not np.isfinite(scale) or scale == 0.0:
        return 1.0
    return scale


@dataclass(frozen=True)
class ElasticNetPenalty:
    r"""
    Elastic-net penalty on the non-intercept coefficients of a logit objective.

    The penalty added to the (sample-averaged) data loss is

    .. math:: \lambda \left( \rho \sum_j |w_j| + \frac{1 - \rho}{2} \sum_j w_j^2 \right)

    with :math:`\rho` the ``l1_ratio`` and
    :math:`\lambda = \mathrm{objective\_scale} / (C \cdot n\_samples)`.

    Dividing by ``n_samples`` is what puts ``C`` on scikit-learn's footing: scikit-learn adds an
    unnormalized penalty to a *summed* data loss, which is the same thing as adding
    ``1 / (C * n_samples)`` to a *mean* one. Multiplying by ``objective_scale`` additionally makes
    the regularization path invariant to rescaling the cost matrix.

    .. seealso::

        :func:`objective_scale_from_costs` : Computes the ``objective_scale`` term.

    Parameters
    ----------
    C : float
        Inverse regularization strength; smaller means stronger regularization.
    l1_ratio : float
        Elastic-net mixing parameter in ``[0, 1]``. ``0`` is pure L2, ``1`` is pure L1.
    objective_scale : float
        Magnitude of the objective's per-sample training signal, from
        :func:`objective_scale_from_costs`.
    n_samples : int
        Number of training samples the data loss is averaged over.
    start_coef : int
        Index at which the penalized coefficients begin. ``1`` when an intercept is fitted, since
        the intercept is never penalized.
    """

    C: float
    l1_ratio: float
    objective_scale: float
    n_samples: int
    start_coef: int

    @property
    def lambda_(self) -> float:
        """Overall penalty weight, ``objective_scale / (C * n_samples)``."""
        if not np.isfinite(self.C) or self.C <= 0.0 or self.n_samples <= 0:
            return 0.0
        return self.objective_scale / (self.C * self.n_samples)

    @property
    def is_active(self) -> bool:
        """Whether the penalty contributes anything to the objective."""
        return self.lambda_ > 0.0

    @property
    def is_nonsmooth(self) -> bool:
        """Whether the penalty has a kink at zero, which a smooth solver cannot minimize."""
        return self.is_active and self.l1_ratio > 0.0

    @property
    def l1_weight(self) -> float:
        """Weight of the L1 term, ``lambda_ * l1_ratio``."""
        return self.lambda_ * self.l1_ratio

    @property
    def l2_weight(self) -> float:
        """Weight of the L2 term, ``lambda_ * (1 - l1_ratio)``."""
        return self.lambda_ * (1.0 - self.l1_ratio)

    def with_n_samples(self, n_samples: int) -> 'ElasticNetPenalty':
        """
        Return a copy of this penalty rescaled to a different sample count.

        Parameters
        ----------
        n_samples : int
            The new number of samples.

        Returns
        -------
        penalty : ElasticNetPenalty
            A new penalty; this one is left untouched.
        """
        return replace(self, n_samples=n_samples)

    def value(self, weights: FloatNDArray) -> float:
        """
        Compute the penalty's contribution to the objective.

        Parameters
        ----------
        weights : ndarray
            Full coefficient vector, intercept included.

        Returns
        -------
        value : float
            Penalty value.
        """
        if not self.is_active:
            return 0.0
        coef = np.asarray(weights, dtype=np.float64)[self.start_coef :]
        value = 0.0
        if self.l1_weight:
            value += self.l1_weight * float(np.sum(np.abs(coef)))
        if self.l2_weight:
            value += 0.5 * self.l2_weight * float(np.dot(coef, coef))
        return value

    def gradient(self, weights: FloatNDArray) -> Float64Array:
        """
        Compute the penalty's contribution to the gradient.

        Parameters
        ----------
        weights : ndarray
            Full coefficient vector, intercept included.

        Returns
        -------
        gradient : ndarray of numpy.float64
            Same shape as *weights*, zero in the unpenalized intercept slots. For ``l1_ratio > 0``
            this is a subgradient: it is zero at ``w_j = 0``, where the penalty is not
            differentiable.
        """
        w = np.asarray(weights, dtype=np.float64)
        gradient: Float64Array = np.zeros_like(w)
        if not self.is_active:
            return gradient
        coef = w[self.start_coef :]
        if self.l1_weight:
            gradient[self.start_coef :] += self.l1_weight * np.sign(coef)
        if self.l2_weight:
            gradient[self.start_coef :] += self.l2_weight * coef
        return gradient

    def add_to(self, loss: float, gradient: Float64Array, weights: FloatNDArray) -> tuple[float, Float64Array]:
        """
        Add the penalty to a data loss and its gradient in one pass.

        Parameters
        ----------
        loss : float
            Data loss.
        gradient : ndarray
            Data gradient; modified in place.
        weights : ndarray
            Full coefficient vector, intercept included.

        Returns
        -------
        loss : float
            Penalized loss.
        gradient : ndarray
            Penalized gradient.
        """
        if not self.is_active:
            return loss, gradient
        coef = np.asarray(weights, dtype=np.float64)[self.start_coef :]
        if self.l1_weight:
            loss += self.l1_weight * float(np.sum(np.abs(coef)))
            gradient[self.start_coef :] += self.l1_weight * np.sign(coef)
        if self.l2_weight:
            loss += 0.5 * self.l2_weight * float(np.dot(coef, coef))
            gradient[self.start_coef :] += self.l2_weight * coef
        return loss, gradient

    def l2_value(self, coef: FloatNDArray) -> float:
        """
        Compute the smooth (L2) half of the penalty for already-sliced coefficients.

        Used by the split-variable solver, which handles the L1 half separately as a linear term.

        Parameters
        ----------
        coef : ndarray
            Penalized coefficients only, i.e. ``weights[start_coef:]``.

        Returns
        -------
        value : float
            L2 contribution.
        """
        if not self.l2_weight:
            return 0.0
        coef_f = np.asarray(coef, dtype=np.float64)
        return 0.5 * self.l2_weight * float(np.dot(coef_f, coef_f))

    def l2_gradient(self, coef: FloatNDArray) -> Float64Array:
        """
        Compute the gradient of the smooth (L2) half for already-sliced coefficients.

        Parameters
        ----------
        coef : ndarray
            Penalized coefficients only, i.e. ``weights[start_coef:]``.

        Returns
        -------
        gradient : ndarray of numpy.float64
            L2 contribution, same shape as *coef*.
        """
        coef_f: Float64Array = np.asarray(coef, dtype=np.float64)
        if not self.l2_weight:
            return np.zeros_like(coef_f)
        return self.l2_weight * coef_f

    def prox(self, weights: FloatNDArray, step: float) -> Float64Array:
        """
        Apply the proximal operator of the penalty to *weights*.

        Soft-thresholds by ``step * l1_weight`` and then shrinks by
        ``1 / (1 + step * l2_weight)``. This is the exact proximal step of a proximal-gradient
        method, and it is what makes such a method produce exact zeros.

        Parameters
        ----------
        weights : ndarray
            Full coefficient vector, intercept included.
        step : float
            Step size of the gradient step this prox follows.

        Returns
        -------
        weights : ndarray of numpy.float64
            A new array; *weights* is left untouched. The intercept is passed through unchanged.
        """
        w: Float64Array = np.array(weights, dtype=np.float64, copy=True)
        if not self.is_active or step <= 0.0:
            return w
        coef = w[self.start_coef :]
        if self.l1_weight:
            coef = np.sign(coef) * np.maximum(np.abs(coef) - step * self.l1_weight, 0.0)
        if self.l2_weight:
            coef = coef / (1.0 + step * self.l2_weight)
        w[self.start_coef :] = coef
        return w

    @classmethod
    def from_costs(
        cls,
        *,
        y_true: FloatNDArray,
        tp_benefit: FloatNDArray | float,
        tn_benefit: FloatNDArray | float,
        fp_cost: FloatNDArray | float,
        fn_cost: FloatNDArray | float,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        n_samples: int,
    ) -> 'ElasticNetPenalty':
        """
        Build a penalty whose scale is derived from a cost matrix.

        Parameters
        ----------
        y_true : ndarray
            Binary labels, recoded to 0/1.
        tp_benefit : float or ndarray
            Benefit of true positives.
        tn_benefit : float or ndarray
            Benefit of true negatives.
        fp_cost : float or ndarray
            Cost of false positives.
        fn_cost : float or ndarray
            Cost of false negatives.
        C : float
            Inverse regularization strength.
        l1_ratio : float
            Elastic-net mixing parameter.
        fit_intercept : bool
            Whether an (unpenalized) intercept is fitted.
        n_samples : int
            Number of training samples.

        Returns
        -------
        penalty : ElasticNetPenalty
            The configured penalty.
        """
        return cls.from_scale(
            objective_scale=objective_scale_from_costs(y_true, tp_benefit, tn_benefit, fp_cost, fn_cost),
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_samples=n_samples,
        )

    @classmethod
    def from_scale(
        cls, *, objective_scale: float, C: float, l1_ratio: float, fit_intercept: bool, n_samples: int
    ) -> 'ElasticNetPenalty':
        """
        Build a penalty from an already-computed objective scale.

        Parameters
        ----------
        objective_scale : float
            Magnitude of the objective's per-sample training signal.
        C : float
            Inverse regularization strength.
        l1_ratio : float
            Elastic-net mixing parameter.
        fit_intercept : bool
            Whether an (unpenalized) intercept is fitted.
        n_samples : int
            Number of training samples.

        Returns
        -------
        penalty : ElasticNetPenalty
            The configured penalty.
        """
        scale = objective_scale if np.isfinite(objective_scale) and objective_scale > 0.0 else 1.0
        return cls(
            C=C,
            l1_ratio=l1_ratio,
            objective_scale=float(scale),
            n_samples=n_samples,
            start_coef=1 if fit_intercept else 0,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two penalties field by field."""
        if not isinstance(other, ElasticNetPenalty):
            return NotImplemented
        return (
            self.C == other.C
            and self.l1_ratio == other.l1_ratio
            and self.objective_scale == other.objective_scale
            and self.n_samples == other.n_samples
            and self.start_coef == other.start_coef
        )

    def __hash__(self) -> int:
        """Hash the penalty by its fields, matching :meth:`__eq__`."""
        return hash((self.C, self.l1_ratio, self.objective_scale, self.n_samples, self.start_coef))


#: A penalty that contributes nothing, for objectives fitted without regularization.
NO_PENALTY = ElasticNetPenalty(C=np.inf, l1_ratio=0.0, objective_scale=0.0, n_samples=1, start_coef=0)


class LogitObjective(ABC):  # ruff: ignore[abstract-base-class-without-abstract-method]
    """
    Class to compute the loss and gradient of a logistic regression objective.

    An objective is the sum of a *data term* and an :class:`~empulse.metrics.ElasticNetPenalty`.
    Concrete objectives implement the data term through :meth:`data_loss` and :meth:`data_gradient`
    and set :attr:`penalty`; the regularized ``logit_*`` methods are derived from those here.

    Keeping the two separable is what lets a solver treat them differently -- most importantly
    :class:`~empulse.optimizers.LBFGSBOptimizer`, which reformulates a non-smooth L1 penalty rather
    than handing its subgradient to a solver that assumes smoothness.

    Overriding ``logit_loss``/``logit_gradient`` directly and leaving :attr:`penalty` as ``None``
    remains supported: the objective is then opaque to solvers, which fall back to treating it as
    an arbitrary, possibly non-smooth function.
    """

    #: Penalty applied on top of the data term. ``None`` means the objective already includes
    #: whatever penalty it wants inside ``logit_loss``/``logit_gradient``, and solvers must treat
    #: it as opaque.
    penalty: 'ElasticNetPenalty | None' = None

    def data_loss(self, weights: FloatNDArray) -> float:
        """
        Compute the unregularized loss for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Loss of the data term alone.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not expose its data term separately. '
            'Override data_loss() to enable solvers that handle the penalty themselves.'
        )

    def data_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """
        Compute the gradient of the unregularized loss for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient of the data term alone.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not expose its data term separately. '
            'Override data_gradient() to enable solvers that handle the penalty themselves.'
        )

    def data_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """
        Compute the unregularized loss and its gradient for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        loss : float
            Loss of the data term alone.
        gradient : ndarray
            Gradient of the data term alone.
        """
        return self.data_loss(weights), self.data_gradient(weights)

    def logit_loss(self, weights: FloatNDArray) -> float:
        """
        Compute the loss for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Regularized loss.
        """
        loss = self.data_loss(weights)
        return loss if self.penalty is None else loss + self.penalty.value(weights)

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """
        Compute the gradient of the loss for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Regularized gradient.
        """
        gradient = self.data_gradient(weights)
        return gradient if self.penalty is None else gradient + self.penalty.gradient(weights)

    def _logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients for successive weight vectors.

        Because the constants are derived from fixed data and parameters,
        there is no expensive state to reconstruct between steps.  The
        generator accepts the same send-protocol as
        ``MaxProfitLogitGradientPiecewise.logit_gradient_steps`` for API
        compatibility: passing ``(weights, refresh)`` works but ``refresh``
        is silently ignored.

        Send in either a ``weights`` vector or a ``(weights, refresh)`` tuple; ``refresh`` is
        ignored here.

        Yields
        ------
        gradient : ndarray
            Gradient at the current weights.
        """
        weights: FloatNDArray

        sent = yield  # type: ignore[misc]

        while True:
            if sent is None:
                return
            if isinstance(sent, tuple):
                weights, _ = sent
            else:
                weights = sent

            sent = yield self.logit_gradient(weights)

    def logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients for successive weight vectors.

        Send either a ``weights`` vector or a ``(weights, refresh)`` tuple into the generator;
        for this objective ``refresh`` is accepted but ignored.

        Yields
        ------
        gradient : ndarray
            Gradient at the weights last sent in.

        Examples
        --------
        Driving the generator by hand (``objective`` is a built objective,
        ``theta`` a coefficient vector)::

            gen = objective.logit_gradient_steps()
            grad = gen.send(theta)  # first time gradient is computed from scratch
            grad = gen.send(theta)  # gradient computed from cached information
            grad = gen.send((theta, True))  # gradient computed from scratch
            gen.close()
        """
        generator = self._logit_gradient_steps()
        next(generator)
        return generator

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """
        Compute the loss and its gradient for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        loss : float
            Regularized loss.
        gradient : ndarray
            Regularized gradient.
        """
        loss, gradient = self.data_loss_gradient(weights)
        if self.penalty is None:
            return loss, gradient
        return self.penalty.add_to(loss, np.asarray(gradient, dtype=np.float64), weights)

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """
        Compute the loss and its gradient for minimization.

        Here for backward compatibility.  Delegates to ``logit_loss_gradient``.
        """
        return self.logit_loss_gradient(weights)

    def set_alpha(self, alpha: float) -> None:  # ruff: ignore[empty-method-without-abstract-decorator]
        """
        Override the smoothing parameter *alpha* (no-op for objectives without alpha annealing).

        Gradient optimizers with an ``alpha_schedule`` call this before each gradient
        computation to externally drive the annealing schedule.  Objectives that
        implement alpha annealing (e.g. :class:`MaxProfitLogitGradientPiecewise`)
        override this method; all others silently ignore the call.

        Parameters
        ----------
        alpha : float
            New alpha value to use for the next gradient computation.
        """
        # no-op: override in subclasses that support alpha annealing

    def with_indices(self, indices: np.ndarray) -> 'LogitObjective':
        """
        Return a new objective restricted to the sample subset given by *indices*.

        Used by gradient optimizers for mini-batch training.  The default
        implementation raises :exc:`NotImplementedError`; concrete objectives
        that store their data should override this method.

        Only the data arrays are sliced.  :attr:`penalty` is shared unchanged, because its scale is
        already an average and mini-batch data gradients are themselves ``1 / batch_size`` means,
        so the penalty stays consistent across batch sizes.

        Parameters
        ----------
        indices : ndarray of int
            Row indices into the full training set.

        Returns
        -------
        LogitObjective
            A new objective for the selected samples.

        Raises
        ------
        NotImplementedError
            If this objective does not support mini-batch slicing.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support mini-batch training. Override with_indices() to enable it.'
        )
