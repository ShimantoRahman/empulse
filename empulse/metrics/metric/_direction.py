"""The one enum every metric, strategy, and the logit/boosting optimizers all need."""

from enum import Enum, auto


class Direction(Enum):
    """
    Whether a metric is better when higher or when lower.

    Every :class:`~empulse.metrics.MetricStrategy` passes one to its constructor, and every metric
    reports it as :attr:`~empulse.metrics.BaseMetric.direction`. Models and optimizers read it to know
    whether to maximize or minimize the metric, and it gives the ``greater_is_better`` of a scorer.

    Attributes
    ----------
    MAXIMIZE : Direction
        Higher is better, as for a profit or savings.
    MINIMIZE : Direction
        Lower is better, as for a cost.
    """

    MAXIMIZE = auto()
    MINIMIZE = auto()
