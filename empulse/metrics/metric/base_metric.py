from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal

from ..._types import FloatArrayLike, FloatNDArray
from ._compile import CountScoreFn
from ._direction import Direction
from .capabilities import Capability
from .model_interface import ModelInterface
from .strategies import LogitObjective, MetricStrategy


def _reject_sample_weight(parameters: Mapping[str, Any], accepted: Collection[str]) -> None:
    """
    Raise when *parameters* has a ``sample_weight`` that is not one of the *accepted* parameter names.

    The costs of the cost matrix are what weigh each sample, so a separate weight would count the
    weighting twice, or override the costs where they disagree. A cost matrix may still name one of its
    own parameters ``sample_weight``.
    """
    if 'sample_weight' in parameters and 'sample_weight' not in accepted:
        raise TypeError(
            'Metrics do not take sample_weight: the costs of the cost matrix already weigh each sample. '
            'Fold any extra weight into the costs instead, for example as an instance-dependent parameter.'
        )


class BaseMetric(ABC):
    """
    Abstract interface shared by every metric usable as a cost-sensitive ``loss``.

    :class:`~empulse.metrics.Metric` and :class:`~empulse.metrics.MixtureMetric` both implement
    this interface. Cost-sensitive models (e.g. :class:`~empulse.models.CSLogitClassifier`,
    :class:`~empulse.models.CSBoostClassifier`, :class:`~empulse.models.CSThresholdClassifier`)
    accept any :class:`BaseMetric` as their ``loss`` parameter, rather than being hard-coded to
    the concrete :class:`~empulse.metrics.Metric` class. This is what allows
    :class:`~empulse.metrics.MixtureMetric` -- or any future composite/custom metric type -- to
    be used as a drop-in replacement for :class:`~empulse.metrics.Metric` wherever a metric-based
    loss is expected, without every model having to special-case each concrete metric type.

    Subclassing :class:`BaseMetric` directly is only necessary when implementing a new kind of
    metric from scratch. To combine existing :class:`~empulse.metrics.Metric` objects, use
    :class:`~empulse.metrics.MixtureMetric` instead.

    Models reach a metric through :attr:`model`, a :class:`~empulse.metrics.ModelInterface` that
    gives them the metric's parameters, decision costs, losses and training objectives.

    Notes
    -----
    :class:`~empulse.metrics.ModelInterface` forwards to the protected methods below, and a new kind
    of metric implements them. A test keeps the list complete.

    - Parameters: :attr:`_all_parameters`, :attr:`_extra_parameters`, :meth:`_missing_parameters`,
      :meth:`_validate_parameters`, :meth:`_parameter_spellings` and
      :meth:`_outlier_sensitive_parameters`.
    - Values: :meth:`_loss`, :meth:`_prepare_count_loss`, :meth:`_evaluate_costs`,
      :meth:`_decision_costs`, :meth:`_cost_weight` and :attr:`_decides_from_costs`.
    - Training: :meth:`_logit_objective`, :meth:`_logit_value_objective`,
      :meth:`_gradient_boost_objective` and :meth:`_prepare_boost_objective`.
    - Description: :meth:`_require`, :attr:`_is_deterministic`, :attr:`_has_single_cost_matrix` and
      :attr:`_is_unitless`.

    :meth:`_loss` and the four training objectives are always minimized, whatever :attr:`direction`
    says; the objectives need only have the same minimizer as :meth:`_loss`. A
    :class:`~empulse.metrics.MixtureMetric` weighs each component's objective by its
    :meth:`_cost_weight`, so that the sum of the objectives keeps the minimizer of the sum of the
    scores. :meth:`_closed_form_expected_max_profit` is optional, and only ProfTree's compiled code
    uses it.
    """

    @property
    @abstractmethod
    def strategy(self) -> MetricStrategy:
        """
        The strategy used to compute the metric.

        For a composite metric, a representative strategy shared by all of its components.
        """

    @property
    def model(self) -> ModelInterface:
        """
        What a model needs of this metric: its parameters, decision costs, losses and training objectives.

        Only someone writing their own estimator needs it; see :class:`~empulse.metrics.ModelInterface`.
        """
        return ModelInterface(self)

    def _shared_strategy(self) -> MetricStrategy | None:
        """Return :attr:`strategy`, or ``None`` for a composite metric whose components use more than one type."""
        return self.strategy

    @property
    def _extra_parameters(self) -> frozenset[str]:
        """Keyword arguments the metric accepts besides its cost-matrix symbols, such as Savings' ``baseline``."""
        return frozenset(self.strategy._extra_kwargs)

    @property
    def capabilities(self) -> frozenset[Capability]:
        """
        The set of :class:`~empulse.metrics.Capability` members this metric supports.

        Forwards to :attr:`strategy`'s own :attr:`~empulse.metrics.MetricStrategy.capabilities`.
        :class:`~empulse.metrics.MixtureMetric` overrides this to the *intersection* of its
        components' capabilities, since a composite metric can only do what every component can.

        Use this instead of ``isinstance(metric.strategy, SomeConcreteStrategy)`` to check
        whether a metric supports what a model needs, e.g.
        ``Capability.CLASS_COSTS in loss.capabilities``.
        """
        return self.strategy.capabilities

    def _require(self, capability: Capability, *, requester: str, hint: str | None = None) -> None:
        """
        Raise a ``ValueError`` naming *requester* and *capability* if this metric lacks it.

        A small, uniformly-worded alternative to each caller writing its own
        ``if capability not in loss.capabilities: raise ValueError(...)``. A *hint*, such as what to
        use instead, ends the message.
        """
        if capability not in self.capabilities:
            message = (
                f'{requester} only supports losses whose strategy supports {capability.value!r}; '
                f'{self.__name__!r} does not.'
            )
            raise ValueError(message if hint is None else f'{message} {hint}')

    @property
    @abstractmethod
    def __name__(self) -> str:
        """Human-readable name of the metric, e.g. for use as a scorer name."""

    @property
    @abstractmethod
    def direction(self) -> Direction:
        """Whether the metric is to be maximized or minimized."""

    @property
    @abstractmethod
    def _all_symbols(self) -> set[str]:
        """The set of all parameter names accepted by the metric, including stochastic variables."""

    @property
    def parameter_names(self) -> set[str]:
        """
        The names of the parameters a caller can pass to this metric, under each of their spellings.

        These are the cost matrix's symbols, the parameters of its stochastic variables'
        distributions and the aliases of either. A stochastic variable itself is not among them: its
        values are drawn from its distribution, not passed.
        """
        return self._all_parameters

    @property
    @abstractmethod
    def _all_parameters(self) -> set[str]:
        """The set of parameter names the metric expects to be supplied by the caller."""

    @property
    @abstractmethod
    def _default_parameter_names(self) -> set[str]:
        """The set of parameter names that have a default value and need not be supplied."""

    @property
    def _parameter_defaults(self) -> dict[str, Any]:
        """The default value of each parameter that has one, keyed by symbol name."""
        return {}

    @property
    @abstractmethod
    def _is_deterministic(self) -> bool:
        """Whether the metric is free of stochastic (random) variables."""

    @property
    def _has_single_cost_matrix(self) -> bool:
        """
        Whether the metric is computed from one cost matrix, rather than combining several.

        A maximum profit over the class costs is then the maximum profit of the metric itself, which
        a composite metric, such as a weighted sum of maximum profits, is not.
        """
        return False

    @property
    def _is_unitless(self) -> bool:
        """Whether the metric's value is a pure number, like a ratio of profits, rather than an amount of money."""
        return False

    @property
    def _label_dependent_cost_weight(self) -> bool:
        """Whether :meth:`_cost_weight` depends on the labels, as it does for :class:`~empulse.metrics.Savings`."""
        return False

    def _cost_weight(self, y_true: FloatNDArray, **parameters: Any) -> float:
        """
        Return how much the metric's value changes per unit of the expected cost its decision and training minimize.

        See :meth:`MetricStrategy._cost_weight`. One unless :attr:`_label_dependent_cost_weight`.
        """
        return 1.0

    @property
    def _decides_from_costs(self) -> bool:
        """
        Whether the optimal decision is the one that minimizes the expected cost of :meth:`_decision_costs`.

        So it is for a metric with ``Capability.COST_ONLY_DECISION``, and for a
        :class:`~empulse.metrics.MixtureMetric` of such metrics, which lacks that capability when the
        labels weigh its components' costs.
        """
        return Capability.COST_ONLY_DECISION in self.capabilities

    def _decision_costs(
        self, y_true: FloatNDArray, **parameters: Any
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """
        Return the ``(fp, fn, tp, tn)`` costs that the metric's decision and training minimize the expected cost of.

        Its :meth:`_evaluate_costs`, with stochastic costs replaced by their mean, except for a
        :class:`~empulse.metrics.MixtureMetric` whose components' costs the labels *y_true* weigh.
        """
        return self._evaluate_costs(replace_stochastic=True, **parameters)

    @abstractmethod
    def _missing_parameters(self, supplied: Iterable[str]) -> set[str]:
        """Return the required parameter names not covered by *supplied*.

        A parameter counts as covered when *supplied* contains it under any of its accepted
        spellings (its raw symbol name or any alias registered for it), or when it has a default
        value and therefore need not be supplied at all. Unlike comparing against
        :attr:`_all_parameters` directly, this correctly handles metrics with aliases (where a
        parameter is satisfied by *either* spelling, never both) and tolerates unrelated extra
        keys in *supplied* (e.g. ``sample_weight``).
        """

    @abstractmethod
    def __call__(
        self,
        y_true: FloatArrayLike,
        y_score: FloatArrayLike,
        *,
        validate: bool = True,
        **parameters: FloatArrayLike | float,
    ) -> float:
        """Compute the metric score or loss."""

    def score(
        self,
        y_true: FloatArrayLike,
        y_score: FloatArrayLike,
        *,
        validate: bool = True,
        **parameters: FloatArrayLike | float,
    ) -> float:
        """Compute the metric score or loss (see :meth:`__call__`)."""
        return self(y_true, y_score, validate=validate, **parameters)

    @abstractmethod
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

    def _loss(
        self,
        y_true: FloatArrayLike,
        y_score: FloatArrayLike,
        *,
        validate: bool = True,
        **parameters: FloatArrayLike | float,
    ) -> float:
        """Compute the metric as a value to be minimized, whatever its :attr:`direction`.

        Models optimize a loss, while a metric may naturally be a score (:attr:`direction` is
        :attr:`~empulse.metrics.metric.common.Direction.MAXIMIZE`). This is the single place where
        the two conventions are reconciled, so that no model has to negate a metric itself.

        Pass ``validate=False`` when re-entering on a per-iteration training path; see
        ``Metric._prepare_parameters``.
        """
        value = self(y_true, y_score, validate=validate, **parameters)
        return -value if self.direction is Direction.MAXIMIZE else value

    def _prepare_count_loss(
        self, *, n_samples: int | None = None, validate: bool = True, **parameters: FloatArrayLike | float
    ) -> CountScoreFn | None:
        """
        Prepare :meth:`_loss` for samples grouped by score, or return ``None`` if it needs every sample.

        For a model whose predictions take few distinct values, such as the leaves of a decision
        tree: ``loss(y_score, n_positive, n_negative)`` returns the loss of the samples those groups
        stand for. The parameters are resolved, and validated when ``validate`` is set, once here
        rather than on every call, and no call re-checks ``y_true`` or ``y_score``.

        ``n_samples`` is the number of samples the groups stand for, which array-valued parameters
        must match.
        """
        return None

    def _closed_form_expected_max_profit(
        self, *, n_positives: int, n_negatives: int, validate: bool = True, **parameters: FloatArrayLike | float
    ) -> Any:
        """
        Return what compiled code needs to compute this metric's expected maximum profit, or ``None``.

        Only a metric whose value is (up to its sign) an expected maximum profit computed in closed
        form returns something, a ``ClosedFormExpectedMaxProfit`` for samples with these class counts.
        """
        return None

    @abstractmethod
    def optimal_rate(
        self,
        y_true: FloatArrayLike,
        y_score: FloatArrayLike,
        *,
        validate: bool = True,
        **parameters: FloatArrayLike | float,
    ) -> float:
        """Compute the optimal predicted positive rate."""

    @abstractmethod
    def optimal_threshold(
        self,
        y_true: FloatArrayLike,
        y_score: FloatArrayLike,
        *,
        validate: bool = True,
        **parameters: FloatArrayLike | float,
    ) -> FloatNDArray | float:
        """Compute the optimal classification threshold(s)."""

    @abstractmethod
    def _prepare_boost_objective(self, y_true: FloatNDArray, **parameters: FloatNDArray | float) -> FloatNDArray:
        """Compute the gradient's constant term of the metric with respect to gradient boosting."""

    @abstractmethod
    def _gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the metric loss with respect to gradient boosting weights."""

    @abstractmethod
    def _logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """Compute the logit loss and its gradient with respect to the logistic regression weights."""

    def _logit_value_objective(
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

        Takes the same arguments as :meth:`_logit_objective`, which it returns by default, since
        that objective exposes the value too. Metrics whose gradient is costly to compute, or only
        approximated, return an objective that computes the value alone.
        """
        return self._logit_objective(
            features=features, y_true=y_true, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **parameters
        )

    @abstractmethod
    def _evaluate_costs(
        self, *, replace_stochastic: bool = False, **parameters: FloatNDArray | float
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """Evaluate the (class-dependent or instance-dependent) cost expressions."""

    def _canonical_name(self, name: str) -> str:
        """Return the name of the parameter the keyword *name* stands for, resolving any alias."""
        return name

    def _parameter_spellings(self, name: str) -> set[str]:
        """Return every keyword the parameter *name* can be passed as: its symbol name and its aliases."""
        return {name}

    @abstractmethod
    def _outlier_sensitive_parameters(self) -> dict[str, Literal['positive', 'negative', 'both']]:
        """Map each outlier-sensitive parameter to the class whose rows it describes.

        A parameter is outlier-sensitive when it was declared with
        :meth:`~empulse.metrics.CostMatrix.mark_outlier_sensitive`. The key is the parameter's
        symbol name, whichever spelling marked it; :meth:`_parameter_spellings` lists every keyword it
        can be passed as --
        :class:`~empulse.models.RobustCSClassifier` uses this to decide, for a given
        outlier-sensitive parameter, which rows of ``X``/``y`` to fit an outlier-detecting
        regressor on: ``'positive'`` restricts to the rows where ``y`` is positive (the parameter
        appears only in :attr:`Metric.tp_cost <empulse.metrics.Metric.tp_cost>`/
        :attr:`~empulse.metrics.Metric.fn_cost`), ``'negative'`` to where it is not
        (:attr:`~empulse.metrics.Metric.tn_cost`/:attr:`~empulse.metrics.Metric.fp_cost` only),
        and ``'both'`` uses every row (the parameter appears on both sides, or the metric cannot
        distinguish -- e.g. a mixture of components that disagree).

        Empty for a metric that declares no outlier-sensitive parameters.
        """
