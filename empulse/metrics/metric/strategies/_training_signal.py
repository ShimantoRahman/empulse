"""Warning when a cost matrix leaves a gradient-based objective with nothing to optimize."""

import warnings

import numpy as np

from ...._types import FloatNDArray


def warn_if_no_training_signal(gradient_constant: FloatNDArray, objective: str) -> None:
    """Warn when a cost matrix leaves a gradient-based objective with nothing to optimize.

    The expected cost is linear in the predicted probability, so its derivative is the per-sample
    constant ``y (tp_cost - fn_cost) + (1 - y) (fp_cost - tn_cost)``. When the cost matrix has
    constant rows -- ``tp_cost == fn_cost`` and ``fp_cost == tn_cost`` -- that constant is zero for
    every sample: classifying a sample either way costs exactly the same, so there is no signal to
    descend.

    This is checked here, in the objectives that differentiate the cost matrix, rather than in
    :meth:`~empulse.models.CostSensitiveClassifier.fit`, because it is not true of every
    cost-sensitive model. :class:`~empulse.models.CSTreeClassifier` and
    :class:`~empulse.models.CSForestClassifier` weight their split quality by a class-purity term
    (``criterion='gini'`` or ``'entropy'``) that does not involve the cost matrix at all, and go on
    learning perfectly well from a cost matrix that is flat in this sense.

    Parameters
    ----------
    gradient_constant : ndarray
        The per-sample derivative of the objective with respect to the predicted probability.
    objective : str
        Name of the objective, used in the message.
    """
    values = np.asarray(gradient_constant, dtype=np.float64)
    if values.size and np.all(values == 0.0):
        warnings.warn(
            f'The cost matrix leaves the {objective} objective with no gradient: tp_cost == fn_cost '
            'and fp_cost == tn_cost, so classifying a sample either way costs the same and the '
            'derivative with respect to the predicted probability is zero for every sample. This '
            'model will not learn from these costs. Check the cost matrix and the values passed to '
            'fit(). Tree-based models with criterion="gini" or "entropy" are not affected, because '
            'their split quality does not rely on the cost matrix alone.',
            UserWarning,
            stacklevel=3,
        )
