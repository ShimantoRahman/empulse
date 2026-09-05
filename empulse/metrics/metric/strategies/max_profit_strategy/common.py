import copy
from collections.abc import Iterable
from typing import Any, Self

import numpy as np
import sympy
from scipy.special import expit

from ....._types import Float64Array, FloatNDArray, IntNDArray
from ...._cy_convex_hull import convex_hull
from ..metric_strategy import LogitObjective


def _convex_hull(y_true: IntNDArray, y_score: FloatNDArray) -> tuple[IntNDArray, FloatNDArray]:
    return convex_hull(y_true.astype(np.int32), y_score.astype(np.float64))  # type: ignore[no-any-return]


def extract_distribution_parameters(
    parameters: dict[str, Any], distribution_args: Iterable[sympy.Symbol]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the distribution parameters from the other parameters."""
    distribution_parameters = {
        str(key): parameters.pop(str(key)) for key in distribution_args if str(key) in parameters
    }
    return distribution_parameters, parameters


class _BaseMaxProfitLogitObjective(LogitObjective):
    """Shared state and helpers for MaxProfit's logistic-regression objectives.

    Both the deterministic and piecewise-stochastic MaxProfit logit objectives need identical
    soft-thresholding, elastic-net regularization, alpha-based smoothing, and index-based
    mini-batch slicing; this base class holds that shared machinery so the two subclasses only
    implement their differing profit/gradient computation.
    """

    def __init__(
        self,
        *,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        alpha: float,
    ) -> None:
        self.features = features
        self.y_true = y_true.ravel().astype(np.int32)
        self.C = C
        self.l1_ratio = l1_ratio
        self.soft_threshold = soft_threshold
        self.fit_intercept = fit_intercept
        self.alpha = float(alpha)
        self._refresh_sample_state()

    def _refresh_sample_state(self) -> None:
        """(Re)derive sample masks, counts, and per-class feature matrices from features/y_true."""
        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = int(self.pos_mask.sum())
        self.n_neg = int(self.neg_mask.sum())
        self.X_pos: Float64Array = np.asarray(self.features[self.pos_mask], dtype=np.float64)
        self.X_neg: Float64Array = np.asarray(self.features[self.neg_mask], dtype=np.float64)
        self.pi0 = float(self.n_pos / len(self.y_true))
        self.pi1 = 1.0 - self.pi0

    @property
    def _start_coef(self) -> int:
        return 1 if self.fit_intercept else 0

    def set_alpha(self, alpha: float) -> None:
        """Override the smoothing parameter used for subsequent gradient computations.

        Parameters
        ----------
        alpha : float
            Smoothing parameter value.
        """
        self.alpha = float(alpha)

    def _apply_soft_threshold(self, weights: FloatNDArray) -> FloatNDArray:
        """Return a copy of *weights* with soft-thresholding applied (if enabled)."""
        start_coef = self._start_coef
        w = np.asarray(weights, dtype=np.float64).copy()
        if self.soft_threshold:
            abs_w = np.abs(w[start_coef:])
            diff = abs_w - self.C
            w[start_coef:] = np.where(
                diff > 0,
                np.sign(w[start_coef:]) * diff,
                np.where(diff < 0, 0.0, w[start_coef:]),
            )
        return w

    def _compute_y_score(self, w: FloatNDArray) -> FloatNDArray:
        """Compute logistic scores for every sample."""
        return expit(self.features @ w)  # type: ignore[return-value]

    def _regularization_value(self, coef: FloatNDArray) -> float:
        """Regularization contribution to the scalar objective."""
        if self.l1_ratio == 0.0:
            return 0.5 * float(np.dot(coef, coef)) / self.C
        if self.l1_ratio == 1.0:
            return float(np.sum(np.abs(coef))) / self.C
        return (
            (1.0 - self.l1_ratio) * 0.5 * float(np.dot(coef, coef)) + self.l1_ratio * float(np.sum(np.abs(coef)))
        ) / self.C

    def _regularization_gradient(self, coef: FloatNDArray) -> Float64Array:
        """Regularization contribution to the gradient."""
        coef_f = np.asarray(coef, dtype=np.float64)
        if self.l1_ratio == 0.0:
            return coef_f / self.C
        if self.l1_ratio == 1.0:
            return np.sign(coef_f) / self.C
        return ((1.0 - self.l1_ratio) * coef_f + self.l1_ratio * np.sign(coef_f)) / self.C

    def with_indices(self, indices: FloatNDArray) -> Self:
        """Return a shallow copy of this objective restricted to *indices*.

        Parameters
        ----------
        indices : array-like of int
            Row indices into the full training set.

        Returns
        -------
        Self
            A new objective for the selected samples.
        """
        obj = copy.copy(self)
        obj.features = self.features[indices]
        obj.y_true = self.y_true[indices]  # already int32
        obj._refresh_sample_state()
        return obj
