from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..._types import FloatArrayLike, FloatNDArray
from .common import Direction
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
    """

    @property
    @abstractmethod
    def strategy(self) -> MetricStrategy:
        """
        The strategy used to compute the metric.

        For a composite metric, a representative strategy shared by all of its components.
        """

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
    def __call__(self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float) -> float:
        """Compute the metric score or loss."""

    @abstractmethod
    def optimal_rate(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float
    ) -> float:
        """Compute the optimal predicted positive rate."""

    @abstractmethod
    def optimal_threshold(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float
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
        soft_threshold: bool,
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
