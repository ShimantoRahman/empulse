import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, Literal, NamedTuple, Self, TypeAlias

import numpy as np
import sympy

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray
from ..common import classification_threshold
from ._compile import CountScoreFn, PicklableLambda, _safe_lambdify, _safe_run_lambda
from ._direction import Direction
from ._parameter_domain import _check_parameter_domains
from ._symbolic import _sympify_term
from .base_metric import BaseMetric, _reject_sample_weight
from .capabilities import Capability
from .cost_matrix import (
    _CONDITION_TAKES_NO_BOUNDS,
    CostMatrix,
    ParameterPredicate,
    _add_bounds,
    _check_bounds,
    _constraint_expression,
    _ExpressionConstraint,
)
from .metric import Metric
from .strategies import ElasticNetPenalty, LogitObjective, MetricStrategy

Weight: TypeAlias = float | str | sympy.Expr | Callable[[dict[str, Any]], float]


class MixtureComponent(NamedTuple):
    """
    One term of a :class:`MixtureMetric`.

    Parameters
    ----------
    weight : float, str or sympy.Expr
        The mixture weight for this component.

        - If ``float``, a constant weight.
        - If ``str`` or a sympy expression, an expression in parameters supplied at call time, such as
          ``'success_rate'`` or ``'1 - success_rate - default_rate'``. Its symbols are parameters of
          the :class:`MixtureMetric`, so they can have defaults and constraints, be passed through
          a model's ``fit``, and be routed like any other parameter.

        .. deprecated::
            A callable weight, called with the keyword arguments passed to the
            :class:`MixtureMetric`, is deprecated. The mixture cannot see which parameters it reads,
            nor can it be pickled when it is a ``lambda``. Pass an expression instead.

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


MixtureComponent.weight.__doc__ = 'The mixture weight for this component.'
MixtureComponent.metric.__doc__ = 'The metric to evaluate for this component.'
MixtureComponent.parameters.__doc__ = 'Parameter overrides fixed for this component.'


def _check_component(index: int, component: Any) -> MixtureComponent:
    """Return *component* as a :class:`MixtureComponent`, or raise if it is not one."""
    if not isinstance(component, tuple) or len(component) != 3:
        raise TypeError(
            f'MixtureMetric component {index} must be a MixtureComponent(weight, metric, parameters), '
            f'got {component!r}.'
        )
    weight, metric, parameters = component
    if not isinstance(weight, Real | str | sympy.Expr) and not callable(weight):
        raise TypeError(
            f'The weight of MixtureMetric component {index} must be a number or an expression, '
            f'got {weight!r}. MixtureComponent takes its fields in the order (weight, metric, parameters).'
        )
    if not isinstance(metric, BaseMetric):
        raise TypeError(
            f'The metric of MixtureMetric component {index} must be a BaseMetric, such as a Metric, '
            f'got {metric!r}. MixtureComponent takes its fields in the order (weight, metric, parameters).'
        )
    if not isinstance(parameters, Mapping):
        raise TypeError(
            f'The parameters of MixtureMetric component {index} must be a mapping of parameter names to values, '
            f'got {parameters!r}.'
        )
    return MixtureComponent(weight, metric, dict(parameters))


class _CompiledWeight:
    """A component's weight, compiled once, with the names of the parameters it reads."""

    function: PicklableLambda | Callable[[dict[str, Any]], float]

    def __init__(self, weight: Weight) -> None:
        self.weight = weight
        if callable(weight) and not isinstance(weight, sympy.Basic):
            warnings.warn(
                'A callable MixtureComponent weight is deprecated: the mixture cannot see which parameters '
                "it reads, nor can a lambda be pickled. Pass an expression instead, such as '1 - a - b'.",
                FutureWarning,
                stacklevel=4,
            )
            self.expression: sympy.Expr | None = None
            #: ``None`` when the parameters cannot be known, for a callable weight.
            self.names: frozenset[str] | None = None
            self.function = weight
            return
        self.expression = _sympify_term(weight) if isinstance(weight, str) else sympy.sympify(weight)
        self.names = frozenset(str(symbol) for symbol in self.expression.free_symbols)
        self.function = _safe_lambdify(self.expression)

    @property
    def constant(self) -> float | None:
        """The weight, if it does not depend on any parameter."""
        if self.expression is None or self.names:
            return None
        return float(self.expression)

    def __call__(self, parameters: Mapping[str, Any]) -> float:
        if self.expression is None:
            return float(self.function(dict(parameters)))  # type: ignore[arg-type]
        missing = sorted(name for name in self.names or () if name not in parameters)
        if missing:
            raise ValueError(f'MixtureMetric expected a value for weight parameter(s) {missing}, did not receive it.')
        value = np.asarray(_safe_run_lambda(self.function, self.expression, **dict(parameters)), dtype=np.float64)
        if value.size != 1:
            raise ValueError(f'The MixtureMetric weight {self.weight!r} must be a single value, got {value.size}.')
        return float(value.reshape(-1)[0])


def _extra_parameters(metric: BaseMetric) -> frozenset[str]:
    """Keyword arguments *metric* accepts besides its cost-matrix symbols, such as Savings' ``baseline``."""
    try:
        return frozenset(metric.strategy._extra_kwargs)
    except (AttributeError, ValueError):  # a nested mixture of inconsistent strategies
        return frozenset()


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

    When every component decides from its costs alone (``Capability.COST_ONLY_DECISION``, as for
    :class:`~empulse.metrics.Cost`), the decision is instead that of the weighted combined costs,
    and the components must share one strategy type for the mixture to offer it. A mixture that
    offers neither raises instead of averaging its components' decisions.

    :class:`~empulse.metrics.Savings` divides each component's cost by the cost of that component's
    baseline, so a mixture of savings combines the costs, and trains models, with each weight divided
    by that baseline cost too. That depends on the labels, so such a mixture lacks
    ``Capability.COST_ONLY_DECISION`` and ``Capability.EXPECTED_COST``.

    Read more in the :ref:`User Guide <metric_objects>`.

    Parameters
    ----------
    components : Sequence[MixtureComponent]
        The components making up the mixture. Every component's
        :attr:`~MixtureComponent.metric` should share the same interpretation of ``y_true`` and
        ``y_score``, and should be built from the same underlying cost-matrix pattern, only
        differing in how the uncertain symbol is fixed or distributed for that component.
        The components must optimize in the same direction. Constant weights must not be negative
        and, when every weight is constant, must sum to 1.

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
            MixtureComponent('1 - success_rate - default_rate', metric_stoch, {}),
        ])

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
        empcs_metric(y_true, y_proba, success_rate=0.55, default_rate=0.1, roi=0.2644)
    """

    def __init__(self, components: Sequence[MixtureComponent], defaults: Mapping[str, float] | None = None) -> None:
        if not components:
            raise ValueError('MixtureMetric requires at least one component.')
        self.components = [_check_component(index, component) for index, component in enumerate(components)]
        directions = {component.metric.direction for component in self.components}
        if len(directions) > 1:
            raise ValueError(
                f'MixtureMetric components have inconsistent optimization directions: {directions}. '
                'All components must optimize in the same direction.'
            )
        if len({component.metric._is_unitless for component in self.components}) > 1:
            raise ValueError(
                'MixtureMetric components must all be amounts of money, or all pure numbers such as AUEPC, '
                'since their weighted sum adds them up.'
            )
        self._weights = [_CompiledWeight(component.weight) for component in self.components]
        constant_weights = [weight.constant for weight in self._weights]
        _check_weights([weight for weight in constant_weights if weight is not None])
        if None not in constant_weights and not np.isclose(sum(constant_weights), 1.0):  # type: ignore[arg-type]
            raise ValueError(f'MixtureMetric weights must sum to 1, got {constant_weights}.')
        self.defaults: dict[str, float] = dict(defaults) if defaults is not None else {}
        self._name_override: str | None = None
        # Bounds on the mixture's own parameters -- chiefly the weights, which are not symbols of
        # any component's cost matrix and so are not reached by CostMatrix.constrain().
        self._bounds: dict[str, Any] = {}
        self._predicates: list[ParameterPredicate] = []

    def _validate_parameters(self, **parameters: Any) -> None:
        """
        Check parameter values against the domain this metric declares, and raise if they fall outside.

        Called once, where the user's values first arrive -- the public scoring methods do it
        themselves, and models do it in ``CostSensitiveClassifier.fit``. Training then re-enters the
        metric with the same values many times over and passes ``validate=False``, because the check
        reads array data and would otherwise scale with the iteration count.

        Raises
        ------
        ValueError
            If a distribution rejects its shape parameters, a value falls outside bounds declared
            with :meth:`~empulse.metrics.CostMatrix.constrain`, or a declared predicate fails.
        """
        for _, metric, component_parameters in self._weighted_components(parameters, validate=True):
            metric._validate_parameters(**component_parameters)

    def _weighted_components(
        self, parameters: Mapping[str, Any], *, validate: bool
    ) -> list[tuple[float, BaseMetric, dict[str, Any]]]:
        """
        Resolve the parameters once, into each component's weight, metric and keyword arguments.

        Every method of the mixture goes through this, so that defaults, fixed values, aliases and
        weights are handled the same way everywhere. With ``validate``, the mixture's own constraints
        are checked, and so are the weights: none may be negative, and together they must sum to 1.
        """
        # A default yields to the caller's value under any spelling, as a cost matrix default does.
        supplied = {self._canonical_name(key) for key in parameters}
        defaults = {key: value for key, value in self.defaults.items() if self._canonical_name(key) not in supplied}
        resolved = {**defaults, **parameters}
        _reject_sample_weight(resolved, self._accepted_parameter_names())
        if validate and (self._bounds or self._predicates):
            # Bounds and predicates name a parameter by its symbol, whichever spelling the caller used, and
            # see a value the components take from their own defaults as the components do.
            by_symbol = {
                **self._parameter_defaults,
                **{self._canonical_name(key): value for key, value in resolved.items()},
            }
            caller_names = {self._canonical_name(key): key for key in resolved}
            _check_parameter_domains((), by_symbol, self._bounds, self._predicates, caller_names=caller_names)
        self._warn_about_unknown_parameters(resolved)
        weights = [weight(resolved) for weight in self._weights]
        if validate:
            _check_weights(weights)
            if not np.isclose(sum(weights), 1.0):
                raise ValueError(f'MixtureMetric weights must sum to 1, got {weights}.')
        return [
            (weight, component.metric, self._component_parameters(component, resolved))
            for weight, component in zip(weights, self.components, strict=True)
        ]

    def constrain(
        self,
        target: str | sympy.Basic | Callable[[Mapping[str, Any]], bool],
        lower: float | None = None,
        upper: float | None = None,
        *,
        message: str | None = None,
    ) -> Self:
        """
        Restrict the values a mixture-level parameter is allowed to take.

        The counterpart of :meth:`~empulse.metrics.CostMatrix.constrain` for parameters that belong
        to the mixture rather than to any one component's cost matrix -- in particular the weights.

        Parameters
        ----------
        target : str, sympy relation or callable
            The parameter to bound; a condition the parameters must satisfy, such as ``'w < 0.9'``,
            written as for :meth:`CostMatrix.constrain`; or a callable taking the mapping of parameter
            values and returning whether they are acceptable.

        lower : float, optional
            Smallest allowed value, inclusive.

        upper : float, optional
            Largest allowed value, inclusive.

        message : str, optional
            Explanation to report when a condition or callable `target` rejects the parameters.
            Required when `target` is a callable; a condition reports itself by default.

        Returns
        -------
        self : MixtureMetric
            The metric with the constraint added.

        Raises
        ------
        ValueError
            If `target` is a name and neither `lower` nor `upper` is given, if `lower` is greater than
            `upper`, if `target` names no parameter of the mixture, if `target` is a condition and
            `lower` or `upper` is given, or if `target` is a callable and `message` is not given.

        Notes
        -----
        A parameter may be bounded by its symbol name or by an alias, and the bound holds under every
        spelling a caller passes it as. Bounding the same parameter twice keeps the tighter of each bound.

        A callable receives the parameters keyed by **symbol name**, with aliases already resolved and
        defaults already applied: the mixture's :attr:`defaults`, and those its components share. A
        parameter whose components have different defaults is checked only when passed. As for
        :meth:`CostMatrix.constrain`, a lambda or nested function cannot be pickled, and a condition can.
        """
        expression = _constraint_expression(target)
        if expression is not None:
            if lower is not None or upper is not None:
                raise ValueError(_CONDITION_TAKES_NO_BOUNDS)
            constraint = _ExpressionConstraint(expression)
            if unknown := sorted(constraint.names - self._all_parameters):
                raise ValueError(
                    f"Cannot constrain '{expression}': {unknown} is not a parameter of the mixture. "
                    f'Known parameters: {sorted(self._all_parameters)}.'
                )
            self._predicates.append(
                ParameterPredicate(
                    predicate=constraint.renamed(self._canonical_name), message=message or str(expression)
                )
            )
            return self
        if callable(target):
            if message is None:
                raise ValueError('A message is required when constraining with a callable.')
            self._predicates.append(ParameterPredicate(predicate=target, message=message))
            return self
        bounds = _check_bounds(target, lower, upper)
        if target not in self._all_parameters:
            raise ValueError(
                f"Cannot constrain '{target}': it is not a parameter of the mixture. "
                f'Known parameters: {sorted(self._all_parameters)}.'
            )
        _add_bounds(self._bounds, self._canonical_name(target), bounds)
        return self

    @property
    def direction(self) -> Direction:
        """The optimization direction shared by all components."""
        return self.components[0].metric.direction

    @property
    def __name__(self) -> str:
        if self._name_override is not None:
            return self._name_override
        names = '+'.join(component.metric.__name__ for component in self.components)
        return f'MixtureMetric({names})'

    @__name__.setter  # ruff: ignore[builtin-attribute-shadowing]
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
    def capabilities(self) -> frozenset[Capability]:
        """
        The set of :class:`~empulse.metrics.Capability` members every component supports.

        The *intersection* of each component's own :attr:`~empulse.metrics.BaseMetric.capabilities`,
        computed directly from the components rather than through :attr:`strategy` -- unlike
        :attr:`strategy`, this does not require the components to share one
        :class:`~empulse.metrics.MetricStrategy` type. In practice every bundled
        :class:`MixtureMetric` (e.g. :func:`~empulse.metrics.empcs_score`) does use one strategy
        type throughout, so this agrees with routing through :attr:`strategy` for those.
        """
        capabilities = frozenset.intersection(*(component.metric.capabilities for component in self.components))
        if self._components_decide_from_costs and self._shared_strategy() is None:
            # The decision follows from the combined costs only through one strategy's rule.
            capabilities -= {Capability.COST_ONLY_DECISION, Capability.OPTIMAL_THRESHOLD, Capability.OPTIMAL_RATE}
        if self._label_dependent_cost_weight and len(self.components) > 1:
            # The labels weigh the components' costs, as the savings divide each by its baseline's cost.
            # So the decision needs the labels, and the costs summed with the mixture weights alone
            # are not what the score weighs.
            capabilities -= {Capability.COST_ONLY_DECISION, Capability.EXPECTED_COST}
        return capabilities

    @property
    def _components_decide_from_costs(self) -> bool:
        return all(component.metric._decides_from_costs for component in self.components)

    def _shared_strategy(self) -> MetricStrategy | None:
        """Return the strategy every component uses, or ``None`` if they use more than one type."""
        try:
            return self.strategy
        except (AttributeError, ValueError):
            return None

    def _combined_decision_strategy(self) -> MetricStrategy | None:
        """
        Return the strategy whose rule decides from the combined costs, or ``None`` if there is none.

        There is one when every component decides from its costs alone, by the same strategy's rule.
        """
        return self._shared_strategy() if self._components_decide_from_costs else None

    @property
    def _decides_from_costs(self) -> bool:
        return self._combined_decision_strategy() is not None

    @property
    def _label_dependent_cost_weight(self) -> bool:
        return any(component.metric._label_dependent_cost_weight for component in self.components)

    def _cost_weight(self, y_true: FloatNDArray, **parameters: Any) -> float:
        """Return the weighted sum of the components' cost weights, which is how much the mixture's value scales."""
        if not self._label_dependent_cost_weight:
            return 1.0
        return float(
            sum(
                weight * metric._cost_weight(y_true, **component_parameters)
                for weight, metric, component_parameters in self._weighted_components(parameters, validate=False)
            )
        )

    def _decision_weights(
        self, y_true: FloatNDArray, parameters: Mapping[str, Any], *, validate: bool
    ) -> list[tuple[float, BaseMetric, dict[str, Any]]]:
        """
        Like :meth:`_weighted_components`, but with the weights that combine the components' costs and objectives.

        These are the mixture weights, each times the component's :meth:`_cost_weight`, normalized to
        sum to 1. The score adds up :class:`~empulse.metrics.Savings`, for one, each of which divides its
        component's cost by its baseline's cost, so the decision and training must weigh the costs alike.
        Without such a component, they are the mixture weights, and so they are for a single component,
        whose weight normalizes to 1 whatever its labels.
        """
        weighted = self._weighted_components(parameters, validate=validate)
        if not self._label_dependent_cost_weight or len(weighted) == 1:
            return weighted
        scaled = [
            weight * metric._cost_weight(y_true, **component_parameters)
            for weight, metric, component_parameters in weighted
        ]
        total = sum(scaled)
        if not np.all(np.isfinite(scaled)) or total <= 0:
            raise ValueError(
                f'Cannot weigh the components of {self.__name__!r} by their scores: the baseline of a Savings '
                'component costs nothing on these labels, or there are no labels, so its savings are undefined.'
            )
        return [
            (weight / total, metric, component_parameters)
            for weight, (_, metric, component_parameters) in zip(scaled, weighted, strict=True)
        ]

    def _decision_costs(
        self, y_true: FloatNDArray, **parameters: Any
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """Combine the components' decision costs with the :meth:`_decision_weights`."""
        total_fp: FloatNDArray | float = 0.0
        total_fn: FloatNDArray | float = 0.0
        total_tp: FloatNDArray | float = 0.0
        total_tn: FloatNDArray | float = 0.0
        for weight, metric, component_parameters in self._decision_weights(y_true, parameters, validate=False):
            fp_cost, fn_cost, tp_cost, tn_cost = metric._decision_costs(y_true, **component_parameters)
            total_fp = total_fp + weight * fp_cost
            total_fn = total_fn + weight * fn_cost
            total_tp = total_tp + weight * tp_cost
            total_tn = total_tn + weight * tn_cost
        return total_fp, total_fn, total_tp, total_tn

    def _require_decision(self, capability: Capability) -> None:
        """Raise if the mixture lacks *capability*, rather than average what its components decide."""
        if capability not in self.capabilities:
            raise ValueError(
                f'{self.__name__!r} does not support {capability.value}: not every component does, or its '
                'components decide from their costs alone but by the rules of different strategies, which '
                'do not combine.'
            )

    @property
    def _all_symbols(self) -> set[str]:
        """The set of all parameter names accepted by the mixture (weight names and component symbols)."""
        symbols = set(self._weight_parameter_names)
        for component in self.components:
            # Exclude symbols this component fixes internally: they are never read from the
            # keyword arguments passed in at call time, since the fixed override always wins.
            symbols |= component.metric._all_symbols - self._fixed_names(component)
        return symbols

    @property
    def _all_parameters(self) -> set[str]:
        """The set of parameter names the mixture expects to be supplied by the caller."""
        symbols = set(self._weight_parameter_names)
        for component in self.components:
            symbols |= component.metric._all_parameters - self._fixed_names(component)
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
            names |= component.metric._default_parameter_names - self._fixed_names(component)
        return names

    @property
    def _parameter_defaults(self) -> dict[str, Any]:
        """
        The defaults of the mixture, and those its components share, keyed by symbol name.

        A parameter whose components have different defaults has no one value to check a constraint of
        the mixture against, so it is left out. A parameter a component fixes is not read from its default.
        """
        shared: dict[str, Any] = {}
        conflicting: set[str] = set()
        for component in self.components:
            fixed = self._fixed_names(component)
            for name, value in component.metric._parameter_defaults.items():
                if name in fixed:
                    continue
                if name in shared and not np.array_equal(shared[name], value):
                    conflicting.add(name)
                shared.setdefault(name, value)
        own = {self._canonical_name(name): value for name, value in self.defaults.items()}
        return {**{name: value for name, value in shared.items() if name not in conflicting}, **own}

    @property
    def _is_deterministic(self) -> bool:
        """Whether every component is free of stochastic (random) variables."""
        return all(component.metric._is_deterministic for component in self.components)

    @property
    def _is_unitless(self) -> bool:
        """Whether every component's value is a pure number rather than an amount of money."""
        return all(component.metric._is_unitless for component in self.components)

    def _missing_parameters(self, supplied: Iterable[str]) -> set[str]:
        """
        Return the required parameter names not covered by *supplied*.

        A mixture-level default (:attr:`defaults`) counts as covered for every component, in
        addition to whatever the caller passed, since it is merged in before parameters are
        forwarded (see :meth:`_weighted_components`). Parameters a component fixes internally
        (:attr:`MixtureComponent.parameters`) are never required from the caller.
        """
        effectively_supplied = set(supplied) | self.defaults.keys()
        missing = {name for name in self._weight_parameter_names if name not in effectively_supplied}
        for component in self.components:
            missing |= component.metric._missing_parameters(effectively_supplied) - self._fixed_names(component)
        return missing

    @property
    def _weight_parameter_names(self) -> set[str]:
        """Names of the parameters the weights read, as far as they are known."""
        return set().union(*(weight.names or () for weight in self._weights))

    def _accepted_parameter_names(self) -> set[str]:
        """Return the keyword arguments the mixture accepts: its weights' and its components' parameters."""
        accepted = set(self._weight_parameter_names)
        for component in self.components:
            accepted |= self._accepted_names(component)
        return accepted

    def _warn_about_unknown_parameters(self, parameters: Mapping[str, Any]) -> None:
        if any(weight.names is None for weight in self._weights):
            return  # a callable weight may read any parameter
        unknown = set(parameters) - self._accepted_parameter_names()
        if unknown:
            warnings.warn(
                f'Unknown parameters passed to metric: {sorted(unknown)}. These will be ignored.',
                UserWarning,
                stacklevel=4,
            )

    @staticmethod
    def _fixed_names(component: MixtureComponent) -> set[str]:
        """Every spelling of the symbols a component fixes: their names and all their aliases."""
        return set().union(*(component.metric._parameter_spellings(name) for name in component.parameters))

    @staticmethod
    def _accepted_names(component: MixtureComponent) -> set[str]:
        """Return the keyword arguments a component's metric accepts from the caller."""
        return component.metric._all_parameters | _extra_parameters(component.metric)

    def _canonical_name(self, name: str) -> str:
        """Return the name of the parameter the keyword *name* stands for: a weight's, or a component's symbol."""
        if name in self._weight_parameter_names:
            return name
        for component in self.components:
            if name in component.metric._all_parameters:
                return component.metric._canonical_name(name)
        return name

    @classmethod
    def _component_parameters(cls, component: MixtureComponent, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """
        Return the keyword arguments for one component: those it accepts, with its fixed values on top.

        A fixed value takes priority over whichever spelling, symbol name or alias, the caller used.
        """
        accepted = cls._accepted_names(component)
        fixed = cls._fixed_names(component)
        own = {key: value for key, value in parameters.items() if key in accepted and key not in fixed}
        return {**own, **component.parameters}

    def __call__(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, *, validate: bool = True, **parameters: Any
    ) -> float:
        """
        Compute the weighted sum of each component's metric score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.
        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores.
        **parameters : float or array-like of shape (n_samples,)
            Parameter values, including any weight parameters named by a component's
            :attr:`~MixtureComponent.weight`.

        Returns
        -------
        score : float
            The mixture's combined score.
        """
        return float(
            sum(
                weight * metric(y_true, y_score, validate=validate, **component_parameters)
                for weight, metric, component_parameters in self._weighted_components(parameters, validate=validate)
            )
        )

    def _prepare_count_loss(
        self, *, n_samples: int | None = None, validate: bool = True, **parameters: Any
    ) -> CountScoreFn | None:
        """
        Prepare :meth:`_loss` for samples grouped by score (see :meth:`BaseMetric._prepare_count_loss`).

        The weighted sum of the components' own, so ``None`` unless every component has one.
        """
        weighted_losses = []
        for weight, metric, component_parameters in self._weighted_components(parameters, validate=validate):
            component_loss = metric._prepare_count_loss(n_samples=n_samples, validate=validate, **component_parameters)
            if component_loss is None:
                return None
            weighted_losses.append((weight, component_loss))

        def loss(y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray) -> float:
            return float(sum(weight * fn(y_score, n_positive, n_negative) for weight, fn in weighted_losses))

        return loss

    def optimal_rate(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, *, validate: bool = True, **parameters: Any
    ) -> float:
        """
        Compute the weighted sum of each component's optimal predicted positive rate.

        For components that decide from their costs alone, the rate of the decision made from the
        weighted combined costs instead.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.
        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores.
        **parameters : float or array-like of shape (n_samples,)
            Parameter values, including any weight parameters named by a component's
            :attr:`~MixtureComponent.weight`.

        Returns
        -------
        optimal_rate : float
            The mixture's combined optimal predicted positive rate.

        Raises
        ------
        ValueError
            If the mixture does not support ``Capability.OPTIMAL_RATE``.
        """
        self._require_decision(Capability.OPTIMAL_RATE)
        combined = self._combined_cost_decision(y_true, parameters, validate=validate)
        if combined is not None:
            metric, costs = combined
            return metric.optimal_rate(y_true, y_score, validate=validate, **costs)
        return float(
            sum(
                weight * metric.optimal_rate(y_true, y_score, validate=validate, **component_parameters)
                for weight, metric, component_parameters in self._weighted_components(parameters, validate=validate)
            )
        )

    def optimal_threshold(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, *, validate: bool = True, **parameters: Any
    ) -> float | FloatNDArray:
        """
        Compute the classification threshold that achieves the mixture's combined optimal rate.

        This is *not* the weighted sum of each component's own optimal threshold: a threshold
        is a non-linear function of a rate (the score value at that rank), so it does not
        commute with the mixture's linear weighting the way score and rate do. Instead, the
        combined :meth:`optimal_rate` is computed first, and the threshold that achieves that
        rate on the pooled ``y_score`` is returned, exactly mirroring how
        :class:`~empulse.metrics.MaxProfit` computes its own optimal threshold from its optimal
        rate. For components that decide from their costs alone, the threshold is that of the
        weighted combined costs.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.
        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores.
        **parameters : float or array-like of shape (n_samples,)
            Parameter values, including any weight parameters named by a component's
            :attr:`~MixtureComponent.weight`.

        Returns
        -------
        optimal_threshold : float | FloatNDArray
            The optimal classification threshold(s).

        Raises
        ------
        ValueError
            If the mixture does not support ``Capability.OPTIMAL_THRESHOLD``.
        """
        self._require_decision(Capability.OPTIMAL_THRESHOLD)
        combined = self._combined_cost_decision(y_true, parameters, validate=validate)
        if combined is not None:
            metric, costs = combined
            return metric.optimal_threshold(y_true, y_score, validate=validate, **costs)
        rate = self.optimal_rate(y_true, y_score, validate=validate, **parameters)
        return classification_threshold(y_true, y_score, rate)  # type: ignore[return-value]

    def _combined_cost_decision(
        self, y_true: FloatArrayLike, parameters: dict[str, Any], *, validate: bool
    ) -> tuple[Metric, dict[str, Any]] | None:
        """
        Return the metric and costs that decide for a mixture of cost-only components, or ``None``.

        When every component decides from its costs alone, as :class:`~empulse.metrics.Cost` does,
        the mixture's decision is that of its weighted combined costs: weighting the components'
        rates instead would average decisions made under each component's costs separately. The
        costs are weighted as the score weighs them (see :meth:`_decision_weights`).
        """
        strategy = self._combined_decision_strategy()
        if strategy is None:
            return None
        if validate:
            self._validate_parameters(**parameters)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        fp_cost, fn_cost, tp_cost, tn_cost = self._decision_costs(y_true, **parameters)
        cached = getattr(self, '_combined_cost_metric', None)
        if cached is None or type(cached.strategy) is not type(strategy):
            cost_matrix = CostMatrix().add_tp_cost('tp').add_tn_cost('tn').add_fp_cost('fp').add_fn_cost('fn')
            cached = Metric(cost_matrix, strategy)
            self._combined_cost_metric = cached
        return cached, {'tp': tp_cost, 'tn': tn_cost, 'fp': fp_cost, 'fn': fn_cost}

    def _prepare_boost_objective(self, y_true: FloatNDArray, **parameters: Any) -> FloatNDArray:
        total: FloatNDArray = sum(  # type: ignore[assignment]
            weight * metric._prepare_boost_objective(y_true, **component_parameters)
            for weight, metric, component_parameters in self._decision_weights(y_true, parameters, validate=False)
        )
        return total

    def _gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any
    ) -> tuple[FloatNDArray, FloatNDArray]:
        total_gradient: Any = 0.0
        total_hessian: Any = 0.0
        for weight, metric, component_parameters in self._decision_weights(y_true, parameters, validate=False):
            gradient, hessian = metric._gradient_boost_objective(y_true, y_score, **component_parameters)
            total_gradient = total_gradient + weight * gradient
            total_hessian = total_hessian + weight * hessian
        return total_gradient, total_hessian

    def _logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: Any,
    ) -> LogitObjective:
        return self._mixture_logit_objective(
            'objective', features, y_true, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **parameters
        )

    def _logit_value_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: Any,
    ) -> LogitObjective:
        """Build the weighted sum of the components' value-only logit objectives."""
        return self._mixture_logit_objective(
            'value', features, y_true, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **parameters
        )

    def _mixture_logit_objective(
        self,
        kind: Literal['objective', 'value'],
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: Any,
    ) -> LogitObjective:
        """Combine the components' logit objectives, full or value-only as *kind* says."""
        weighted_objectives = []
        for weight, metric, component_parameters in self._decision_weights(y_true, parameters, validate=False):
            build = metric._logit_objective if kind == 'objective' else metric._logit_value_objective
            objective = build(
                features=features,
                y_true=y_true,
                C=C,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                **component_parameters,
            )
            weighted_objectives.append((weight, objective))

        # Each component was built with the same C and l1_ratio, so it carries its own penalty.
        # Summing them weighted would apply the penalty once per component, scaled by that
        # component's mixture weight. Strip them and give the mixture a single penalty whose scale
        # is the weighted sum of the components' scales, matching the weighted data term.
        objective_scale = 0.0
        for weight, objective in weighted_objectives:
            if objective.penalty is not None:
                objective_scale += weight * objective.penalty.objective_scale
                objective.penalty = None
        mixture = _MixtureLogitObjective(weighted_objectives)
        mixture.penalty = ElasticNetPenalty.from_scale(
            objective_scale=objective_scale,
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_samples=features.shape[0],
        )
        return mixture

    def _evaluate_costs(
        self, *, replace_stochastic: bool = False, **parameters: Any
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """Compute the weighted sum of each component's (class- or instance-dependent) costs."""
        total_fp: FloatNDArray | float = 0.0
        total_fn: FloatNDArray | float = 0.0
        total_tp: FloatNDArray | float = 0.0
        total_tn: FloatNDArray | float = 0.0
        for weight, metric, component_parameters in self._weighted_components(parameters, validate=False):
            fp_cost, fn_cost, tp_cost, tn_cost = metric._evaluate_costs(
                replace_stochastic=replace_stochastic, **component_parameters
            )
            total_fp = total_fp + weight * fp_cost
            total_fn = total_fn + weight * fn_cost
            total_tp = total_tp + weight * tp_cost
            total_tn = total_tn + weight * tn_cost
        return total_fp, total_fn, total_tp, total_tn

    def _parameter_spellings(self, name: str) -> set[str]:
        """Return every keyword the parameter *name* can be passed as, in any component."""
        spellings = {name}
        for component in self.components:
            spellings |= component.metric._parameter_spellings(name)
        return spellings

    def _outlier_sensitive_parameters(self) -> dict[str, Literal['positive', 'negative', 'both']]:
        """
        Merge each component's outlier-sensitive parameters.

        See :meth:`BaseMetric._outlier_sensitive_parameters`. A parameter that components classify
        differently describes the rows of both classes, so it is ``'both'``. A parameter a component
        fixes is not read from the caller, so it is not outlier-sensitive there.
        """
        result: dict[str, Literal['positive', 'negative', 'both']] = {}
        for component in self.components:
            fixed = self._fixed_names(component)
            for name, rows in component.metric._outlier_sensitive_parameters().items():
                if name not in fixed:
                    result[name] = rows if result.get(name, rows) == rows else 'both'
        return result

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(components={self.components!r})'


def _check_weights(weights: Sequence[float]) -> None:
    if any(weight < 0 for weight in weights):
        raise ValueError(f'MixtureMetric weights must not be negative, got {list(weights)}.')


class _MixtureLogitObjective(LogitObjective):
    """Weighted sum of several :class:`LogitObjective` instances. See :class:`MixtureMetric`."""

    def __init__(self, weighted_objectives: list[tuple[float, LogitObjective]]) -> None:
        self._weighted_objectives = weighted_objectives

    @property
    def n_threads(self) -> int:  # type: ignore[override]
        """Number of threads each component objective may use for one evaluation."""
        return max((objective.n_threads for _, objective in self._weighted_objectives), default=1)

    @n_threads.setter
    def n_threads(self, n_threads: int) -> None:
        for _, objective in self._weighted_objectives:
            objective.n_threads = n_threads

    def data_loss(self, weights: FloatNDArray) -> float:
        """Compute the weighted sum of each component objective's unpenalized loss."""
        return float(sum(weight * objective.logit_loss(weights) for weight, objective in self._weighted_objectives))

    def data_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Compute the weighted sum of each component objective's unpenalized gradient."""
        total: FloatNDArray | None = None
        for weight, objective in self._weighted_objectives:
            contribution = weight * objective.logit_gradient(weights)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total

    def data_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Compute the weighted sum of each component objective's unpenalized loss and gradient."""
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
        sliced = _MixtureLogitObjective([
            (weight, objective.with_indices(indices)) for weight, objective in self._weighted_objectives
        ])
        sliced.penalty = self.penalty
        return sliced

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
            assert total is not None
            if self.penalty is not None:
                current = sent[0] if isinstance(sent, tuple) else sent
                total = total + self.penalty.gradient(current)
            sent = yield total
