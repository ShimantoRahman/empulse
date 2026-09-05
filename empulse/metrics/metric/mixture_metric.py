from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from ..._types import FloatArrayLike, FloatNDArray
from ..common import classification_threshold
from .base_metric import BaseMetric
from .common import Direction
from .strategies import LogitObjective, MetricStrategy

Weight = float | str | Callable[[dict[str, Any]], float]


class MixtureComponent(NamedTuple):
    """
    One term of a :class:`MixtureMetric`.

    Attributes
    ----------
    weight : float | str | Callable[[dict], float]
        The mixture weight for this component.

        - If ``float``, a constant weight.
        - If ``str``, the name of a parameter supplied at call time (e.g. ``'success_rate'``),
          looked up in the keyword arguments passed to the :class:`MixtureMetric`.
        - If callable, called with the full dict of keyword arguments passed to the
          :class:`MixtureMetric` and should return a ``float``. Useful for a weight that is
          derived from other parameters
          (e.g. ``lambda p: 1 - p['success_rate'] - p['default_rate']``).

    metric : BaseMetric
        The metric to evaluate for this component. Usually a :class:`Metric`, but any
        :class:`BaseMetric` works, including a nested :class:`MixtureMetric`.

    parameters : dict[str, Any]
        Parameter overrides fixed for this component, merged on top of (and taking priority
        over) the keyword arguments passed to the :class:`MixtureMetric`
        (e.g. ``{'gamma': 0.0}`` to evaluate a deterministic metric at a specific point mass).
    """

    weight: Weight
    metric: BaseMetric
    parameters: dict[str, Any]


class MixtureMetric(BaseMetric):
    r"""
    A weighted linear combination ("mixture") of :class:`Metric` objects.

    Some domain metrics assume that an uncertain cost-matrix parameter follows a distribution
    that mixes point masses and/or continuous pieces. For example, EMP for Credit Scoring
    assumes the fraction of a defaulted loan that is recovered is 0 with probability ``p0``
    (full recovery), 1 with probability ``p1`` (full loss), and otherwise follows a
    Uniform(0, 1) distribution. :mod:`sympy.stats` has no way to represent such a
    "spike + spike + continuous" random variable as a single object.

    :class:`MixtureMetric` sidesteps this by exploiting linearity of expectation.
    :class:`~empulse.metrics.MaxProfit` (and the other strategies) compute a linear functional
    of the assumed density of the uncertain parameter: the score, its gradient and Hessian with
    respect to model scores, and the optimal predicted-positive rate are all integrals (or
    derivatives of integrals) against that density. Since integration and differentiation are
    both linear, a mixture density's value for any of these quantities is *exactly* the
    weight-averaged sum of each mixture component's own value, evaluated independently. No
    approximation is involved, as long as each component's own :class:`Metric` already handles
    its piece of the mixture: a point mass is a deterministic :class:`Metric` evaluated with the
    parameter fixed to that point, and a continuous piece is a stochastic :class:`Metric` with
    that random variable.

    ``optimal_threshold`` is the one exception: a threshold is a non-linear function of a rate
    (the score value at that rank in ``y_score``), so it does not commute with the mixture's
    linear weighting the way score and rate do. :class:`MixtureMetric` computes it correctly by
    first combining the components' rates and then converting that single combined rate to a
    threshold, rather than combining the components' own thresholds.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    Parameters
    ----------
    components : Sequence[MixtureComponent]
        The components making up the mixture. Every component's
        :attr:`~MixtureComponent.metric` should share the same interpretation of ``y_true`` and
        ``y_score``, and should be built from the same underlying cost-matrix pattern, only
        differing in how the uncertain symbol is fixed or distributed for that component.

    defaults : Mapping[str, float], optional
        Default values for parameters (including weight parameters named by a component's
        :attr:`~MixtureComponent.weight`), used when not supplied at call time.
        Mirrors :meth:`~empulse.metrics.CostMatrix.set_default` for a plain :class:`Metric`.

    Examples
    --------
    Reimplementing the cost structure behind :func:`~empulse.metrics.empcs_score` using
    :class:`MixtureMetric`.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import CostMatrix, MaxProfit, Metric, MixtureComponent, MixtureMetric

        gamma, roi = sp.symbols('gamma roi')
        credit_matrix = CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi)
        metric_det = Metric(credit_matrix, MaxProfit())

        gamma_rv = sp.stats.Uniform('gamma', 0, 1)
        credit_matrix_stoch = CostMatrix().add_tp_benefit(gamma_rv).add_fp_cost(roi)
        metric_stoch = Metric(credit_matrix_stoch, MaxProfit())

        empcs_metric = MixtureMetric([
            MixtureComponent('success_rate', metric_det, {'gamma': 0.0}),
            MixtureComponent('default_rate', metric_det, {'gamma': 1.0}),
            MixtureComponent(
                lambda p: 1 - p['success_rate'] - p['default_rate'], metric_stoch, {}
            ),
        ])

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
        empcs_metric(y_true, y_proba, success_rate=0.55, default_rate=0.1, roi=0.2644)
    """

    def __init__(self, components: Sequence[MixtureComponent], defaults: Mapping[str, float] | None = None) -> None:
        if not components:
            raise ValueError('MixtureMetric requires at least one component.')
        self.components = list(components)
        self.defaults: dict[str, float] = dict(defaults) if defaults is not None else {}
        self._name_override: str | None = None

    def _apply_defaults(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Fill in missing parameters (including weight parameters) from :attr:`defaults`."""
        return {**self.defaults, **parameters}

    @property
    def direction(self) -> Direction:
        """The optimization direction shared by all components."""
        directions = {component.metric.direction for component in self.components}
        if len(directions) > 1:
            raise ValueError(
                f'MixtureMetric components have inconsistent optimization directions: {directions}. '
                'All components must optimize in the same direction.'
            )
        return next(iter(directions))

    @property
    def __name__(self) -> str:
        if self._name_override is not None:
            return self._name_override
        names = '+'.join(component.metric.__name__ for component in self.components)
        return f'MixtureMetric({names})'

    @__name__.setter  # noqa: A003
    def __name__(self, value: str) -> None:
        self._name_override = value

    @property
    def strategy(self) -> MetricStrategy:
        """
        A representative strategy shared by all components.

        Raises a :exc:`ValueError` if components use different :class:`MetricStrategy` types
        (e.g. mixing a :class:`~empulse.metrics.MaxProfit` component with a
        :class:`~empulse.metrics.Cost` component). This lets model code that inspects
        ``loss.strategy`` (e.g. to check ``isinstance(loss.strategy, MaxProfit)``) work
        transparently with a :class:`MixtureMetric`, exactly as it would with a plain
        :class:`Metric`.
        """
        strategies = [component.metric.strategy for component in self.components]
        strategy_types = {type(strategy) for strategy in strategies}
        if len(strategy_types) > 1:
            raise ValueError(
                f'MixtureMetric components use inconsistent strategies: {strategy_types}. '
                'All components must use the same MetricStrategy type.'
            )
        return strategies[0]

    @property
    def _all_symbols(self) -> set[str]:
        """The set of all parameter names accepted by the mixture (weight names and component symbols)."""
        symbols = set(self._weight_parameter_names)
        for component in self.components:
            # Exclude symbols this component fixes internally: they are never read from the
            # keyword arguments passed in at call time, since the fixed override always wins.
            symbols |= component.metric._all_symbols - set(component.parameters.keys())
        return symbols

    @property
    def _all_parameters(self) -> set[str]:
        """The set of parameter names the mixture expects to be supplied by the caller."""
        symbols = set(self._weight_parameter_names)
        for component in self.components:
            symbols |= component.metric._all_parameters - set(component.parameters.keys())
        return symbols

    @property
    def _default_parameter_names(self) -> set[str]:
        """The set of parameter names that have a default value and need not be supplied.

        A mixture weight has no default of its own unless one is set via :attr:`defaults`, so
        this is the union of each component's own defaults (excluding whatever that component
        fixes internally, which is never read from the caller-supplied parameters anyway) plus
        any names covered by :attr:`defaults`.
        """
        names: set[str] = set(self.defaults.keys())
        for component in self.components:
            names |= component.metric._default_parameter_names - set(component.parameters.keys())
        return names

    @property
    def _is_deterministic(self) -> bool:
        """Whether every component is free of stochastic (random) variables."""
        return all(component.metric._is_deterministic for component in self.components)

    def _missing_parameters(self, supplied: Iterable[str]) -> set[str]:
        """
        Return the required parameter names not covered by *supplied*.

        A mixture-level default (:attr:`defaults`) counts as covered for every component, in
        addition to whatever the caller passed, since it is merged in before parameters are
        forwarded (see :meth:`_apply_defaults`). Parameters a component fixes internally
        (:attr:`MixtureComponent.parameters`) are never required from the caller.
        """
        effectively_supplied = set(supplied) | self.defaults.keys()
        missing = {name for name in self._weight_parameter_names if name not in effectively_supplied}
        for component in self.components:
            fixed = component.parameters.keys()
            missing |= component.metric._missing_parameters(effectively_supplied) - fixed
        return missing

    @property
    def _weight_parameter_names(self) -> set[str]:
        """Names of keyword arguments that are consumed only to resolve component weights."""
        return {component.weight for component in self.components if isinstance(component.weight, str)}

    @staticmethod
    def _resolve_weight(weight: Weight, parameters: dict[str, Any]) -> float:
        if isinstance(weight, str):
            if weight not in parameters:
                raise ValueError(f'MixtureMetric expected a value for weight parameter {weight!r}, did not receive it.')
            return float(parameters[weight])
        if callable(weight):
            return float(weight(parameters))
        return float(weight)

    def _forward_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Keyword arguments to forward to each component, excluding mixture-only weight names."""
        exclude = self._weight_parameter_names
        return {key: value for key, value in parameters.items() if key not in exclude}

    @staticmethod
    def _component_parameters(component: MixtureComponent, forwarded: dict[str, Any]) -> dict[str, Any]:
        return {**forwarded, **component.parameters}

    def __call__(self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: Any) -> float:
        """
        Compute the weighted sum of each component's metric score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.
        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores.
        parameters : float or array-like of shape (n_samples,)
            Parameter values, including any weight parameters named by a component's
            :attr:`~MixtureComponent.weight`.

        Returns
        -------
        score : float
            The mixture's combined score.
        """
        parameters = self._apply_defaults(parameters)
        forwarded = self._forward_parameters(parameters)
        total = 0.0
        for component in self.components:
            weight = self._resolve_weight(component.weight, parameters)
            total += weight * component.metric(y_true, y_score, **self._component_parameters(component, forwarded))
        return float(total)

    def optimal_rate(self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: Any) -> float:
        """
        Compute the weighted sum of each component's optimal predicted positive rate.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.
        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores.
        parameters : float or array-like of shape (n_samples,)
            Parameter values, including any weight parameters named by a component's
            :attr:`~MixtureComponent.weight`.

        Returns
        -------
        optimal_rate : float
            The mixture's combined optimal predicted positive rate.
        """
        parameters = self._apply_defaults(parameters)
        forwarded = self._forward_parameters(parameters)
        total = 0.0
        for component in self.components:
            weight = self._resolve_weight(component.weight, parameters)
            total += weight * component.metric.optimal_rate(
                y_true, y_score, **self._component_parameters(component, forwarded)
            )
        return float(total)

    def optimal_threshold(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: Any
    ) -> float | FloatNDArray:
        """
        Compute the classification threshold that achieves the mixture's combined optimal rate.

        This is *not* the weighted sum of each component's own optimal threshold: a threshold
        is a non-linear function of a rate (the score value at that rank), so it does not
        commute with the mixture's linear weighting the way score and rate do. Instead, the
        combined :meth:`optimal_rate` is computed first, and the threshold that achieves that
        rate on the pooled ``y_score`` is returned, exactly mirroring how
        :class:`~empulse.metrics.MaxProfit` computes its own optimal threshold from its optimal
        rate.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.
        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores.
        parameters : float or array-like of shape (n_samples,)
            Parameter values, including any weight parameters named by a component's
            :attr:`~MixtureComponent.weight`.

        Returns
        -------
        optimal_threshold : float | FloatNDArray
            The optimal classification threshold(s).
        """
        parameters = self._apply_defaults(parameters)
        rate = self.optimal_rate(y_true, y_score, **parameters)
        return classification_threshold(y_true, y_score, rate)  # type: ignore[return-value]

    def _prepare_boost_objective(self, y_true: FloatNDArray, **parameters: Any) -> FloatNDArray:
        parameters = self._apply_defaults(parameters)
        forwarded = self._forward_parameters(parameters)
        total: FloatNDArray | None = None
        for component in self.components:
            weight = self._resolve_weight(component.weight, parameters)
            contribution = weight * component.metric._prepare_boost_objective(
                y_true, **self._component_parameters(component, forwarded)
            )
            total = contribution if total is None else total + contribution
        assert total is not None  # components is non-empty, checked in __init__
        return total

    def _gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any
    ) -> tuple[FloatNDArray, FloatNDArray]:
        parameters = self._apply_defaults(parameters)
        forwarded = self._forward_parameters(parameters)
        total_gradient: FloatNDArray | None = None
        total_hessian: FloatNDArray | None = None
        for component in self.components:
            weight = self._resolve_weight(component.weight, parameters)
            gradient, hessian = component.metric._gradient_boost_objective(
                y_true, y_score, **self._component_parameters(component, forwarded)
            )
            weighted_gradient = weight * gradient
            weighted_hessian = weight * hessian
            total_gradient = weighted_gradient if total_gradient is None else total_gradient + weighted_gradient
            total_hessian = weighted_hessian if total_hessian is None else total_hessian + weighted_hessian
        assert total_gradient is not None
        assert total_hessian is not None
        return total_gradient, total_hessian

    def _logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        **parameters: Any,
    ) -> LogitObjective:
        parameters = self._apply_defaults(parameters)
        forwarded = self._forward_parameters(parameters)
        weighted_objectives = []
        for component in self.components:
            weight = self._resolve_weight(component.weight, parameters)
            objective = component.metric._logit_objective(
                features=features,
                y_true=y_true,
                C=C,
                l1_ratio=l1_ratio,
                soft_threshold=soft_threshold,
                fit_intercept=fit_intercept,
                **self._component_parameters(component, forwarded),
            )
            weighted_objectives.append((weight, objective))
        return _MixtureLogitObjective(weighted_objectives)

    def _evaluate_costs(
        self, *, replace_stochastic: bool = False, **parameters: Any
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """Compute the weighted sum of each component's (class- or instance-dependent) costs."""
        parameters = self._apply_defaults(parameters)
        forwarded = self._forward_parameters(parameters)
        total_fp: FloatNDArray | float = 0.0
        total_fn: FloatNDArray | float = 0.0
        total_tp: FloatNDArray | float = 0.0
        total_tn: FloatNDArray | float = 0.0
        for component in self.components:
            weight = self._resolve_weight(component.weight, parameters)
            fp_cost, fn_cost, tp_cost, tn_cost = component.metric._evaluate_costs(
                replace_stochastic=replace_stochastic, **self._component_parameters(component, forwarded)
            )
            total_fp = total_fp + weight * fp_cost
            total_fn = total_fn + weight * fn_cost
            total_tp = total_tp + weight * tp_cost
            total_tn = total_tn + weight * tn_cost
        return total_fp, total_fn, total_tp, total_tn

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(components={self.components!r})'


class _MixtureLogitObjective(LogitObjective):
    """Weighted sum of several :class:`LogitObjective` instances. See :class:`MixtureMetric`."""

    def __init__(self, weighted_objectives: list[tuple[float, LogitObjective]]) -> None:
        self._weighted_objectives = weighted_objectives

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Compute the weighted sum of each component objective's loss."""
        return float(sum(weight * objective.logit_loss(weights) for weight, objective in self._weighted_objectives))

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Compute the weighted sum of each component objective's gradient."""
        total: FloatNDArray | None = None
        for weight, objective in self._weighted_objectives:
            contribution = weight * objective.logit_gradient(weights)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Compute the weighted sum of each component objective's loss and gradient."""
        loss = 0.0
        gradient: FloatNDArray | None = None
        for weight, objective in self._weighted_objectives:
            component_loss, component_gradient = objective.logit_loss_gradient(weights)
            loss += weight * component_loss
            contribution = weight * component_gradient
            gradient = contribution if gradient is None else gradient + contribution
        assert gradient is not None
        return float(loss), gradient

    def set_alpha(self, alpha: float) -> None:
        """Forward the smoothing parameter override to every component objective."""
        for _, objective in self._weighted_objectives:
            objective.set_alpha(alpha)

    def with_indices(self, indices: np.ndarray) -> '_MixtureLogitObjective':
        """Return a mixture objective restricted to the sample subset given by *indices*."""
        return _MixtureLogitObjective([
            (weight, objective.with_indices(indices)) for weight, objective in self._weighted_objectives
        ])

    def _logit_gradient_steps(self):  # type: ignore[no-untyped-def]
        generators = [(weight, objective.logit_gradient_steps()) for weight, objective in self._weighted_objectives]
        sent = yield
        while True:
            if sent is None:
                for _, generator in generators:
                    generator.close()
                return
            total: FloatNDArray | None = None
            for weight, generator in generators:
                contribution = weight * generator.send(sent)
                total = contribution if total is None else total + contribution
            sent = yield total
