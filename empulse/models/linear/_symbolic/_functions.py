# Adapted from gplearn 0.4.3 (https://github.com/trevorstephens/gplearn), BSD-3-Clause,
# Copyright (c) 2015-2026 Trevor Stephens. See the license text in this package's ``__init__``.
"""The function nodes a symbolic expression is built from."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import expit

from ...._types import FloatNDArray

# Beyond this the exponential only overflows to infinity, which would invalidate the whole expression.
_EXP_CLIP = 100.0
# Protected operators return a fixed value instead of dividing by (or taking the logarithm of) a near-zero input.
_PROTECTION_THRESHOLD = 0.001


class Function:
    """
    A node of an expression that maps ``arity`` input vectors to one output vector.

    Instances are singletons looked up by name, so they pickle as just their name.
    """

    __slots__ = ('arity', 'func', 'name')

    def __init__(self, name: str, arity: int, func: Callable[..., Any]) -> None:
        self.name = name
        self.arity = arity
        self.func = func

    def __call__(self, *args: Any) -> Any:
        return self.func(*args)

    def __repr__(self) -> str:
        return f'Function({self.name!r})'

    def __reduce__(self) -> tuple[Callable[[str], 'Function'], tuple[str]]:
        return get_function, (self.name,)


def _protected_division(x1: Any, x2: Any) -> FloatNDArray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > _PROTECTION_THRESHOLD, np.divide(x1, x2), 1.0)


def _protected_sqrt(x1: Any) -> FloatNDArray:
    return np.sqrt(np.abs(x1))  # type: ignore[no-any-return]


def _protected_log(x1: Any) -> FloatNDArray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > _PROTECTION_THRESHOLD, np.log(np.abs(x1)), 0.0)


def _protected_inverse(x1: Any) -> FloatNDArray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > _PROTECTION_THRESHOLD, 1.0 / x1, 0.0)


def _protected_exp(x1: Any) -> FloatNDArray:
    return np.exp(np.minimum(x1, _EXP_CLIP))  # type: ignore[no-any-return]


_FUNCTIONS: dict[str, Function] = {
    function.name: function
    for function in (
        Function('add', 2, np.add),
        Function('sub', 2, np.subtract),
        Function('mul', 2, np.multiply),
        Function('div', 2, _protected_division),
        Function('sqrt', 1, _protected_sqrt),
        Function('log', 1, _protected_log),
        Function('abs', 1, np.abs),
        Function('neg', 1, np.negative),
        Function('inv', 1, _protected_inverse),
        Function('max', 2, np.maximum),
        Function('min', 2, np.minimum),
        Function('sin', 1, np.sin),
        Function('cos', 1, np.cos),
        Function('tan', 1, np.tan),
        Function('exp', 1, _protected_exp),
        Function('sig', 1, expit),
    )
}

FUNCTION_NAMES: tuple[str, ...] = tuple(_FUNCTIONS)


def get_function(name: str) -> Function:
    """Look up a function node by name."""
    try:
        return _FUNCTIONS[name]
    except KeyError:
        raise ValueError(f'Unknown function {name!r}. The available functions are: {", ".join(_FUNCTIONS)}.') from None
