"""Helpers for meta-estimators that expose a method only when the estimator they wrap does."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def estimator_has(attr: str, *, delegates: tuple[str, ...] = ('estimator_', 'estimator')) -> Callable[[Any], Any]:
    """
    Return a check for :func:`~sklearn.utils.metaestimators.available_if` that delegates to a sub-estimator.

    Scikit-learn has this function as ``sklearn.utils.validation._estimator_has`` from 1.6 onwards,
    and as a separate private copy in each of its meta-estimators before that.

    Parameters
    ----------
    attr : str
        The attribute the wrapped estimator might or might not have.
    delegates : tuple of str, default=('estimator_', 'estimator')
        The attributes holding the wrapped estimator, checked in order: by default the fitted
        estimator when there is one, otherwise the unfitted one. A sequence of estimators is
        represented by its first element.

    Returns
    -------
    check : callable
        Takes the meta-estimator and returns the attribute, raising :class:`AttributeError` when
        the wrapped estimator lacks it.
    """

    def check(self: Any) -> Any:
        for delegate in delegates:
            if hasattr(self, delegate):
                delegator = getattr(self, delegate)
                if isinstance(delegator, Sequence):
                    return getattr(delegator[0], attr)
                return getattr(delegator, attr)
        raise AttributeError(f'None of the delegates {delegates} are present in the class.')

    return check
