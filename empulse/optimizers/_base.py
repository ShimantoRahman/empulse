from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from .._types import Float64Array, FloatNDArray
from ..metrics import LogitObjective


class Optimizer(ABC):
    """
    Abstract base class for all logit model optimizers.

    A concrete optimizer is a callable: it receives a prepared objective and the feature
    matrix, runs its search, and returns a :class:`scipy.optimize.OptimizeResult`.
    """

    @abstractmethod
    def __call__(
        self,
        objective: LogitObjective,
        X: FloatNDArray,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Run the optimization.

        Parameters
        ----------
        objective : :class:`~empulse.metrics.LogitObjective`
            Prepared objective exposing ``logit_loss``, ``logit_gradient``, and
            ``logit_loss_gradient`` methods.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (used only to determine the number of parameters).
        **kwargs : Any
            Extra keyword arguments forwarded to the underlying solver.

        Returns
        -------
        result : :class:`scipy.optimize.OptimizeResult`
            Optimization result with at least the fields ``x`` (final weight vector),
            ``fun`` (final loss value), ``nit`` (iterations performed), ``success`` and
            ``message``.
        """

    @property
    def requires_gradient(self) -> bool:
        """
        Whether this optimizer uses the objective's gradient, or only its value.

        A model builds the objective it hands to the optimizer from this: an optimizer that needs
        only values gets an objective that scores the model directly, without the work (or the
        smooth approximations) that computing a gradient takes. Defaults to ``True``, which is
        always safe, since the full objective also exposes the value.
        """
        return True

    def _initial_weights(self, X: FloatNDArray) -> Float64Array:
        """
        Return a zero weight vector sized to match *X*.

        Subclasses may override this to use a different initialization strategy.
        """
        return np.zeros(X.shape[1], order='F', dtype=np.float64)
