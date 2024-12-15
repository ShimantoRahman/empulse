from typing import Literal, Callable

import numpy as np

Strategy = Literal['statistical parity', 'demographic parity']
StrategyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _independent_weights(y_true: np.ndarray, protected_attr: np.ndarray) -> np.ndarray:
    """
    compute weights so that y would be statistically independent of the protected attribute.

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
    epsilon = np.finfo(float).eps  # to avoid division by zero
    return (1 / (joint_prob + epsilon)) * prior_protected_attr * prior_y
