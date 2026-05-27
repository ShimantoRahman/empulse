from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from .._types import FloatNDArray
from ..metrics import LogitObjective


class Optimizer(ABC):
    """
    Abstract base class for all logit model optimizers.

    Parameters
    ----------
    objective : :class:`~empulse.metrics.LogitObjective`
        Prepared objective exposing ``logit_loss``, ``logit_gradient``, and
        ``logit_loss_gradient`` methods.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (used only to determine the number of parameters).

    Returns
    -------
    result : :class:`scipy.optimize.OptimizeResult`
        Optimization result with at least the following fields:

        - ``x`` – final weight vector
        - ``fun`` – final loss value
        - ``nit`` – number of iterations performed
        - ``success`` – ``True`` if a convergence criterion was met
        - ``message`` – human-readable status string
    """

    @abstractmethod
    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """Run the optimization and return an :class:`~scipy.optimize.OptimizeResult`."""

    def _initial_weights(self, X: FloatNDArray) -> FloatNDArray:
        """
        Return a zero weight vector sized to match *X*.

        Subclasses may override this to use a different initialization strategy.
        """
        return np.zeros(X.shape[1], order='F', dtype=X.dtype)
