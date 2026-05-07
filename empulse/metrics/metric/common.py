from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import Any, ParamSpec, Protocol, TypeVar

import numpy as np
import sympy

from ..._types import FloatNDArray, IntNDArray

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


class Direction(Enum):
    """Optimization direction of metric."""

    MAXIMIZE = auto()
    MINIMIZE = auto()


class MetricFn(Protocol):  # noqa: D101
    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float: ...  # noqa: D102


class LogitConsts(Protocol):  # noqa: D101
    def prepare(  # noqa: D102
        self, x: FloatNDArray, y_true: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]: ...


class BoostGradientConst(Protocol):  # noqa: D101
    def __call__(self, y_true: FloatNDArray, **kwargs: Any) -> FloatNDArray: ...  # noqa: D102


class ThresholdFn(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> FloatNDArray | float: ...


class RateFn(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> float: ...


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


class PicklableLambda:
    """A callable wrapper that securely pickles lambdified Sympy functions."""

    func: Callable[..., Any]

    def __init__(self, expression: sympy.Expr, variables: Iterable[sympy.Symbol] | None = None):
        self.expression = expression
        self.variables = variables
        self._compile()

    def _compile(self) -> None:
        if not self.expression.free_symbols:
            val = float(self.expression.evalf())
            self.func = lambda *args, **kwargs: val
        else:
            variables = list(self.expression.free_symbols) if self.variables is None else self.variables
            self.func = sympy.lambdify(variables, self.expression)  # type: ignore[assignment]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        return self.func(*args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop('func', None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._compile()


# 3. Your Factory Function (What your classes actually use)
def _safe_lambdify(expression: sympy.Expr, variables: Iterable[sympy.Symbol] | None = None) -> PicklableLambda:
    """Safely lambdify a sympy expression and return a picklable callable."""
    return PicklableLambda(expression, variables)


def _safe_run_lambda(
    function: Callable[..., T],
    expression: sympy.Expr,
    **parameters: FloatNDArray | float,
) -> T:
    """
    Safely evaluate a lambdified expression with the given parameters.

    Parameters
    ----------
    function : callable
        A lambda function that computes the value of the expression.
    expression : sympy.Expr
        The sympy expression to convert.
    parameters : float or NDArray of shape (n_samples,)
        The parameter values for the costs and benefits defined in the metric.
        If any parameter is a stochastic variable, you should pass values for their distribution parameters.
        You can set the parameter values for either the symbol names or their aliases.

    Returns
    -------
    value : Any
        The result of evaluating the function with the given parameters.
    """
    filtered_parameters = _filter_parameters(expression, parameters)
    return function(**filtered_parameters)


def _safe_run_lambda_array(
    function: Callable[..., FloatNDArray | float],
    expression: sympy.Expr,
    shape: int | tuple[int, ...],
    **parameters: FloatNDArray | float,
) -> FloatNDArray:
    """
    Safely evaluate a lambdified expression with the given parameters and enforces it to be an array of the given shape.

    Parameters
    ----------
    function : callable
        A lambda function that computes the value of the expression.
    expression : sympy.Expr
        The sympy expression to convert.
    shape : int or tuple of int
        Shape of the output array.
    parameters : float or NDArray of shape (n_samples,)
        The parameter values for the costs and benefits defined in the metric.
        If any parameter is a stochastic variable, you should pass values for their distribution parameters.
        You can set the parameter values for either the symbol names or their aliases.

    Returns
    -------
    value : FloatNDArray
        The result of evaluating the function with the given parameters.
    """
    filtered_parameters = _filter_parameters(expression, parameters)
    output = function(**filtered_parameters)
    if isinstance(output, float):
        output = np.full(shape, output)
    return output
