from abc import ABC, abstractmethod
from collections.abc import Generator
from functools import lru_cache
from typing import ClassVar, Self

import numpy as np
import sympy

from ...._types import FloatNDArray, IntNDArray
from ..capabilities import Capability
from ..common import Direction
from ._penalty import ElasticNetPenalty  # ruff: ignore[typing-only-first-party-import]

#: Maps each capability that corresponds directly to a `MetricStrategy` method to that method's
#: name, for `_capabilities_from_overrides` to detect an override by. The three capabilities with
#: no entry here (`COST_ONLY_DECISION`, `CLASS_COSTS`, `PRECOMPUTED_BOOST_OBJECTIVE`) describe a
#: property of *what* a method computes, not *whether* it is implemented, so no override-sniffing
#: can infer them -- a strategy that supports one of them must declare it via `_capabilities`.
_CAPABILITY_METHOD_NAMES: dict[Capability, str] = {
    Capability.OPTIMAL_THRESHOLD: 'optimal_threshold',
    Capability.OPTIMAL_RATE: 'optimal_rate',
    Capability.LOGIT_OBJECTIVE: 'logit_objective',
    Capability.BOOST_OBJECTIVE: 'gradient_boost_objective',
    Capability.PRECOMPUTED_BOOST_OBJECTIVE: 'prepare_boost_objective',
}


@lru_cache
def _capabilities_from_overrides(cls: type['MetricStrategy']) -> frozenset[Capability]:
    """Infer capabilities from which optional `MetricStrategy` methods *cls* overrides.

    A capability whose corresponding method is still the base class's own `raise
    NotImplementedError` body is absent; a capability the base class has no method for at all
    (`COST_ONLY_DECISION`, `CLASS_COSTS`, `PRECOMPUTED_BOOST_OBJECTIVE`'s sibling
    `BOOST_OBJECTIVE` aside, see `_CAPABILITY_METHOD_NAMES`) is never inferred this way.

    This is what lets a third-party `MetricStrategy` subclass that overrides e.g.
    `logit_objective` keep working unedited: the capability follows the override automatically,
    without the subclass having to also declare `_capabilities`.
    """
    return frozenset(
        capability
        for capability, method_name in _CAPABILITY_METHOD_NAMES.items()
        if getattr(cls, method_name) is not getattr(MetricStrategy, method_name)
    )


class LogitObjective(ABC):  # ruff: ignore[abstract-base-class-without-abstract-method]
    """Class to compute the loss and gradient of a logistic regression objective.

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


class MetricStrategy(ABC):
    """
    Abstract base class for metric strategies.

    This class defines the interface for metric strategies.
    Metric strategies are used to compute the metric value, gradient, and hessian.

    Parameters
    ----------
    name : str
        Human-readable name of the strategy, surfaced through :attr:`Metric.__name__`.

    direction : Direction
        Whether the user-facing metric value is to be maximized or minimized.

    Attributes
    ----------
    name : str
        The ``name`` passed to the constructor.

    direction : Direction
        The ``direction`` passed to the constructor.
    """

    #: Capabilities this strategy has regardless of the cost matrix it was built from. A strategy
    #: whose support depends on the built matrix (e.g. :class:`~empulse.metrics.MaxProfit`, whose
    #: ``logit_objective``/``gradient_boost_objective`` support depends on which integration
    #: backend :meth:`build` picked) overrides :attr:`capabilities` itself instead of setting this.
    _capabilities: ClassVar[frozenset[Capability]] = frozenset()

    def __init__(self, name: str, direction: Direction):
        self.name = name
        self.direction = direction

    @property
    def capabilities(self) -> frozenset[Capability]:
        """
        The set of :class:`~empulse.metrics.Capability` members this strategy supports.

        Combines :attr:`_capabilities` with whatever :meth:`optimal_threshold`,
        :meth:`optimal_rate`, :meth:`logit_objective`, :meth:`gradient_boost_objective` and
        :meth:`prepare_boost_objective` this strategy overrides -- see
        :func:`~empulse.metrics.metric.strategies.metric_strategy._capabilities_from_overrides`.
        A caller should use this instead of ``isinstance(strategy, SomeConcreteStrategy)`` or
        calling a method and catching ``NotImplementedError``.
        """
        return self._capabilities | _capabilities_from_overrides(type(self))

    @property
    def requires_dynamic_boost_objective(self) -> bool:
        """
        Whether gradients must be recomputed from the metric each boosting round.

        .. deprecated::
            Use ``Capability.PRECOMPUTED_BOOST_OBJECTIVE not in strategy.capabilities`` instead.

        ``True`` for strategies whose per-sample loss is not linear in the predicted
        probability (e.g. :class:`~empulse.metrics.LogCost`) or that need the current round's
        scores to locate a threshold (e.g. :class:`~empulse.metrics.MaxProfit`); such strategies
        cannot use the precomputed constant returned by :meth:`prepare_boost_objective`.
        """
        return Capability.PRECOMPUTED_BOOST_OBJECTIVE not in self.capabilities

    @property
    def _extra_kwargs(self) -> set[str]:
        """
        Extra keyword arguments accepted by :meth:`score` beyond the cost-matrix parameters.

        Subclasses should override this to declare any additional keyword arguments that
        their :meth:`score` implementation accepts (e.g. ``{'baseline'}`` for
        :class:`~empulse.metrics.Savings`).  These are excluded from the unknown-parameter
        warning raised by :class:`~empulse.metrics.Metric`.
        """
        return set()

    @abstractmethod
    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """
        Compile the four cost-matrix expressions into the strategy's scoring functions.

        Called once by :class:`~empulse.metrics.Metric` at construction.

        Parameters
        ----------
        tp_benefit : sympy.Expr
            Benefit of a true positive.
        tn_benefit : sympy.Expr
            Benefit of a true negative.
        fp_cost : sympy.Expr
            Cost of a false positive.
        fn_cost : sympy.Expr
            Cost of a false negative.

        Returns
        -------
        MetricStrategy
            The built strategy, to allow method chaining.
        """

    @abstractmethod
    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the metric score or loss.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score : float
            The computed metric score or loss.
        """

    def optimal_threshold(
        self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> float | FloatNDArray:
        """
        Compute the classification threshold(s) to optimize the metric value.

        i.e., the score threshold at which an observation should be classified as positive to optimize the metric.
        For instance-dependent costs and benefits, this will return an array of thresholds, one for each sample.
        For class-dependent costs and benefits, this will return a single threshold value.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_threshold : float | FloatNDArray
            The optimal classification threshold(s).
        """
        raise NotImplementedError(f'Optimal threshold is not defined for the {self.name} strategy')

    def optimal_rate(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the predicted positive rate to optimize the metric value.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_rate : float
            The optimal predicted positive rate.
        """
        raise NotImplementedError(f'Optimal rate is not defined for the {self.name} strategy')

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """
        Compute the logit loss and its gradient with respect to the logistic regression weights.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        C : float
            Regularization strength parameter. Smaller values specify stronger regularization.
        l1_ratio : float
            The Elastic-Net mixing parameter, with range 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1 penalty.
        fit_intercept : bool
            Specifies if an intercept should be included in the model.
        **parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        logistic_objective : LogitObjective
            A class that implements the logit loss and its gradient.
        """
        raise NotImplementedError(f'Gradient of the logit function is not defined for the {self.name} strategy')

    def gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient of the metric with respect to gradient boosting instances.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        gradient : NDArray of shape (n_samples,)
            The gradient of the metric loss with respect to the gradient boosting weights.
        hessian : NDArray of shape (n_samples,)
            The hessian of the metric loss with respect to the gradient boosting weights.
        """
        raise NotImplementedError(
            f'Gradient and Hessian of the gradient boosting function is not defined for the {self.name} strategy'
        )

    def prepare_boost_objective(self, y_true: FloatNDArray, **parameters: FloatNDArray | float) -> FloatNDArray:
        """
        Compute the gradient's constant term of the metric wrt gradient boost.

        Parameters
        ----------
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        **parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        gradient_const : NDArray of shape (n_samples, n_features)
            The constant term of the gradient.
        """
        raise NotImplementedError(
            f'Gradient and Hessian of the gradient boosting function is not defined for the {self.name} strategy'
        )

    @abstractmethod
    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """
        Return the LaTeX representation of the metric.

        Parameters
        ----------
        tp_benefit : sympy.Expr
            Benefit of a true positive.
        tn_benefit : sympy.Expr
            Benefit of a true negative.
        fp_cost : sympy.Expr
            Cost of a false positive.
        fn_cost : sympy.Expr
            Cost of a false negative.

        Returns
        -------
        str
            The metric's formula as a LaTeX string.
        """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(direction={self.direction})'
