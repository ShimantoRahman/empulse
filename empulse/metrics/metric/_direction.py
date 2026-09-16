"""The one enum every metric, strategy, and the logit/boosting optimizers all need."""

from enum import Enum, auto


class Direction(Enum):
    """Optimization direction of metric."""

    MAXIMIZE = auto()
    MINIMIZE = auto()
