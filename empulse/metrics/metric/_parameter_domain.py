"""
Checking a supplied parameter value against the domain a cost matrix declares.

Three sources of constraints, checked in :func:`_check_parameter_domains`: a stochastic
variable's own shape parameters (via :mod:`_stochastic`'s distribution checks), bounds registered
with :meth:`~empulse.metrics.CostMatrix.constrain`, and predicates registered the same way. This
runs only where a value first enters a metric -- see the hot-path note on
``BaseMetric._validate_parameters``.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import sympy

from ._stochastic import _check_distribution_parameters, _extremes


def _check_parameter_domains(
    expressions: Iterable[sympy.Expr],
    parameters: Mapping[str, Any],
    bounds: Mapping[str, Any],
    predicates: Iterable[Any],
    caller_names: Mapping[str, str] | None = None,
) -> None:
    """
    Raise if a supplied parameter value falls outside the domain the cost matrix defines.

    Three sources of constraints are checked, in the order they are declared:

    1. The shape parameters of every ``sympy.stats`` random variable in `expressions`, validated by
       the distribution's own ``check``. These need no declaration -- a Beta distribution with a
       negative shape simply does not exist, so ``alpha=-1`` is rejected with sympy's own message.
    2. Bounds registered with :meth:`~empulse.metrics.CostMatrix.constrain`.
    3. Predicates registered with :meth:`~empulse.metrics.CostMatrix.constrain`.

    This runs only where user-supplied values first enter the metric, never on a training loop's
    per-iteration path. See ``Metric._prepare_parameters``.

    Parameters
    ----------
    expressions : Iterable[sympy.Expr]
        The cost-matrix expressions, scanned for random variables.

    parameters : Mapping[str, Any]
        Parameter values keyed by resolved symbol name, with defaults already applied.

    bounds : Mapping[str, ParameterBounds]
        Declared bounds, keyed by resolved symbol name.

    predicates : Iterable[ParameterPredicate]
        Declared cross-parameter conditions.

    caller_names : Mapping[str, str], optional
        Maps a resolved symbol name back to the keyword the caller actually used, so that an error
        names the alias they typed rather than the underlying symbol.

    Raises
    ------
    ValueError
        If a distribution rejects its shape parameters, a value falls outside its declared bounds,
        or a predicate is not satisfied.
    """
    names = caller_names or {}

    def display(name: str) -> str:
        return names.get(name, name)

    _check_distribution_parameters(expressions, parameters, display)

    for name, bound in bounds.items():
        span = _extremes(parameters.get(name))
        if span is None:
            continue
        low, high = span
        if bound.lower is not None and low < bound.lower:
            raise ValueError(_bounds_message(display(name), bound, low))
        if bound.upper is not None and high > bound.upper:
            raise ValueError(_bounds_message(display(name), bound, high))

    for constraint in predicates:
        if not constraint.predicate(parameters):
            raise ValueError(f'The parameters do not satisfy a constraint on the cost matrix: {constraint.message}')


def _bounds_message(name: str, bound: Any, offending: float) -> str:
    """Phrase a bounds violation the way ``empulse.metrics._validation`` phrases its checks."""
    if bound.lower is not None and bound.upper is not None:
        expected = f'lay between {bound.lower} and {bound.upper}'
    elif bound.lower is not None:
        expected = f'be at least {bound.lower}'
    else:
        expected = f'be at most {bound.upper}'
    return f'{name} should {expected}, got a value of {offending} instead.'


def _check_parameters(symbols: Iterable[str | sympy.Expr], kwargs: dict[str, Any]) -> None:
    """
    Check if all parameters are provided.

    In particular:
        - deterministic parameters
        - distribution parameters of stochastic variables
    """
    for symbol in symbols:
        if str(symbol) not in kwargs:
            raise ValueError(f'Metric expected a value for {symbol}, did not receive it.')
