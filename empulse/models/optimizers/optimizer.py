from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def optimize(self, objective, bounds):
        """Optimize the objective function."""
