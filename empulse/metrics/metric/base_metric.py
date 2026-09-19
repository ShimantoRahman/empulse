from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

from ..._types import FloatArrayLike, FloatNDArray
from ._direction import Direction
from .capabilities import Capability
from .strategies import LogitObjective, MetricStrategy


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

    Notes
    -----
    Implementations follow a single orientation rule, so that models never have to branch on
    :attr:`direction`:

    - :meth:`__call__` returns the metric in its natural orientation, which :attr:`direction`
      describes. This is the only orientation a user sees.
    - :meth:`_loss`, :meth:`_logit_objective`, :meth:`_gradient_boost_objective` and
      :meth:`_prepare_boost_objective` are always **minimized**, whatever :attr:`direction` says.
      These are what models optimize.
    - :meth:`optimal_rate`, :meth:`optimal_threshold` and :meth:`_evaluate_costs` are
      orientation-free: an optimal threshold is the same point whether the metric is phrased as a
      cost to minimize or a profit to maximize.

    The gradient-based objectives are only required to be *monotonically equivalent* to
    :meth:`_loss`, not equal to it: they must have the same minimizer, but may differ by an
    increasing transformation. :class:`~empulse.metrics.Savings` is the standing example -- it is
    maximized, but reuses :class:`~empulse.metrics.Cost`'s objectives, and minimizing expected cost
    is equivalent to maximizing savings because savings is a decreasing affine map of cost.
    """

    @property
    @abstractmethod
    def strategy(self) -> MetricStrategy:
        """
        The strategy used to compute the metric.

        For a composite metric, a representative strategy shared by all of its components.
        """

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

    def _require(self, capability: Capability, *, requester: str) -> None:
        """
        Raise a ``ValueError`` naming *requester* and *capability* if this metric lacks it.

        A small, uniformly-worded alternative to each caller writing its own
        ``if capability not in loss.capabilities: raise ValueError(...)``.
        """
        if capability not in self.capabilities:
            raise ValueError(
                f'{requester} only supports losses whose strategy supports {capability.value!r}; '
                f"the '{self.strategy.name}' strategy of {self!r} does not."
            )

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
        The set of all parameter names this metric accepts, including stochastic variables.

        A public alias for :attr:`_all_symbols`, for a caller that wants to know what to pass to
        this metric without reaching into a private name.
        """
        return self._all_symbols

    @property
    @abstractmethod
    def _all_parameters(self) -> set[str]:
        """The set of parameter names the metric expects to be supplied by the caller."""

    @property
    @abstractmethod
    def _default_parameter_names(self) -> set[str]:
        """The set of parameter names that have a default value and need not be supplied."""

    @property
    @abstractmethod
    def _is_deterministic(self) -> bool:
        """Whether the metric is free of stochastic (random) variables."""

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

    @abstractmethod
    def _outlier_sensitive_parameters(self) -> dict[str, Literal['positive', 'negative', 'both']]:
        """Map each outlier-sensitive parameter to the class whose rows it describes.

        A parameter is outlier-sensitive when it was declared with
        :meth:`~empulse.metrics.CostMatrix.mark_outlier_sensitive`. The key is the parameter's
        caller-facing spelling (its alias, if it has one, else its raw symbol name) --
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
