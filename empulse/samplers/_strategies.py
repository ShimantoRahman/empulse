from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from .._types import FloatNDArray, IntNDArray

Strategy = Literal['statistical parity', 'demographic parity']
StrategyFn = Callable[[FloatNDArray, IntNDArray], FloatNDArray]


def _independent_weights(y_true: FloatNDArray, protected_attr: IntNDArray) -> FloatNDArray:
    """
    Compute weights so that y would be statistically independent of the protected attribute.

    Parameters
    ----------
    protected_attr : np.ndarray of shape (n_samples,)
        Protected attribute for each sample.
    y_true : np.ndarray of shape (n_samples,)
        Target variable for each sample.

    Returns
    -------
    group_weights : np.ndarray of shape (n_samples,)
        Weights for each y_true and protected_attr pair.
    """
    prob_protected_attr = protected_attr.mean()
    prob_y = y_true.mean()
    prior_protected_attr = np.array([1 - prob_protected_attr, prob_protected_attr]).reshape((1, 2))
    prior_y = np.array([1 - prob_y, prob_y]).reshape((2, 1))
    if protected_attr.ndim == 2:
        protected_attr = protected_attr.reshape(-1)
    joint_prob = np.histogram2d(y_true, protected_attr, bins=(2, 2))[0] / len(y_true)
    epsilon: np.floating[Any] = np.finfo(float).eps  # to avoid division by zero
    weights: FloatNDArray = (1 / (joint_prob + epsilon)) * prior_protected_attr * prior_y
    return weights
