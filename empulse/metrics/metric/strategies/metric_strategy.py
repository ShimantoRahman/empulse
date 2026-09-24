from abc import ABC, abstractmethod
from functools import lru_cache
from typing import ClassVar, Self

import sympy

from ...._common._objective import LogitObjective
from ...._types import FloatNDArray, IntNDArray
from .._compile import CountScoreFn
from .._direction import Direction
from ..capabilities import Capability

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

    def _prepare_count_score(self, **parameters: FloatNDArray | float) -> CountScoreFn | None:
        """
        Prepare :meth:`score` for samples grouped by score, or return ``None`` if it needs every sample.

        A model whose predictions take few distinct values (e.g. one per leaf of a decision tree)
        can then score them from each value's numbers of positive and negative samples, without
        predicting every sample. The parameter values are fixed for every call of the returned
        function, so a strategy can check and resolve them once here.

        Parameters
        ----------
        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric, as for :meth:`score`.

        Returns
        -------
        score : callable or None
            ``score(y_score, n_positive, n_negative)``, which returns what :meth:`score` would for
            the samples those groups stand for, or ``None`` when this strategy cannot score groups.
        """
        return None

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

    def logit_value_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """
        Build the logit objective for an optimizer that needs only its value, not its gradient.

        Takes the same arguments as :meth:`logit_objective`, which it returns by default: that
        objective exposes the value too. Strategies whose gradient is costly, or approximated,
        override this with an objective that computes the value alone.
        """
        return self.logit_objective(
            features=features, y_true=y_true, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **parameters
        )

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
