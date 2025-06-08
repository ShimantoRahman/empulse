from collections.abc import Callable
from enum import Enum, auto
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar

import sympy

from ..._types import FloatNDArray

T = TypeVar('T', bound=Callable[..., Any])
P = ParamSpec('P')
R = TypeVar('R')


class Direction(Enum):
    """Optimization direction of metric."""

    MAXIMIZE = auto()
    MINIMIZE = auto()


class MetricFn(Protocol):  # noqa: D101
    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float: ...  # noqa: D102


class LogitObjective(Protocol):  # noqa: D101
    def __call__(self, x: FloatNDArray, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> FloatNDArray: ...  # noqa: D102


class BoostObjective(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray]: ...


class ThresholdFn(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> FloatNDArray | float: ...


class RateFn(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> float: ...


def _check_parameters(*parameters: str | sympy.Expr) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Check if all parameters are provided.

    In particular:
        - deterministic parameters
        - distribution parameters of stochastic variables
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for value in parameters:
                if str(value) not in kwargs:
                    raise ValueError(f'Metric expected a value for {value}, did not receive it.')
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _filter_parameters(
    expression: sympy.Expr, parameters: dict[str, float | FloatNDArray]
) -> dict[str, float | FloatNDArray]:
    """
    Filter the parameters dictionary to only include those that are free symbols in the expression.

    Parameters
    ----------
    expression : sympy.Expr
        The expression to filter the parameters against.

    parameters : dict[str, float | FloatNDArray]
        The parameters dictionary to filter.
        Keys are parameter names and values are their corresponding values.

    Returns
    -------
    filtered_parameters : dict[str, float | FloatNDArray]
        A dictionary containing only the parameters that are free symbols in the expression.
    """
    free_symbols = {str(symbol) for symbol in expression.free_symbols}
    filtered_parameters = {key: value for key, value in parameters.items() if key in free_symbols}
    return filtered_parameters


def _evaluate_expression(expression: sympy.Expr, **parameters: FloatNDArray | float) -> FloatNDArray | float:
    """
    Evaluate a sympy expression with the given parameters.

    Parameters
    ----------
    expression : sympy.Expr
        The sympy expression to convert.
    parameters : float or NDArray of shape (n_samples,)
        The parameter values for the costs and benefits defined in the metric.
        If any parameter is a stochastic variable, you should pass values for their distribution parameters.
        You can set the parameter values for either the symbol names or their aliases.

    Returns
    -------
    function : callable
        A numpy function that computes the value of the expression with the given parameters.
    """
    filtered_parameters = _filter_parameters(expression, parameters)
    result: float | FloatNDArray = sympy.lambdify(list(expression.free_symbols), expression)(**filtered_parameters)
    return result
