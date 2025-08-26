from collections.abc import Callable, Iterable
from enum import Enum, auto
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar

import numpy as np
import sympy

from ..._types import FloatNDArray

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


class Direction(Enum):
    """Optimization direction of metric."""

    MAXIMIZE = auto()
    MINIMIZE = auto()


class MetricFn(Protocol):  # noqa: D101
    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float: ...  # noqa: D102


class LogitConsts(Protocol):  # noqa: D101
    def prepare(  # noqa: D102
        self, x: FloatNDArray, y_true: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]: ...


class BoostGradientConst(Protocol):  # noqa: D101
    def __call__(self, y_true: FloatNDArray, **kwargs: Any) -> FloatNDArray: ...  # noqa: D102


class ThresholdFn(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> FloatNDArray | float: ...


class RateFn(Protocol):  # noqa: D101
    def __call__(  # noqa: D102
        self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> float: ...


class SympyFnPickleMixin:
    """
    Class to allow pickling of classes which use lambdified functions.

    Lambdified are deleted before pickling and restored after unpickling.
    Define a class attribute in subclasses: _sympy_functions = {'func_name': 'equation_attr_name'}.
    """

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for func_name in getattr(self, '_sympy_functions', {}):
            if func_name in state:
                del state[func_name]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for func_name, eqn_attr in getattr(self, '_sympy_functions', {}).items():
            equation = getattr(self, eqn_attr)
            if sympy.simplify(equation).is_Number and not equation.free_symbols:
                setattr(self, func_name, lambda *args, **kwargs: float(equation))  # noqa: B023
            else:
                setattr(self, func_name, sympy.lambdify(list(equation.free_symbols), equation))


def _check_parameters_decorator(*parameters: str | sympy.Expr) -> Callable[[Callable[P, R]], Callable[P, R]]:
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


def _safe_lambdify(
    expression: sympy.Expr, variables: Iterable[sympy.Symbol] | None = None
) -> Callable[..., FloatNDArray | float]:
    """
    Safely lambdify a sympy expression.

    If the expression is a constant, return a function that returns the constant value.

    Parameters
    ----------
    expression : sympy.Expr
        The sympy expression to convert.
    variables : Iterable[sympy.Symbol], optional
        The variables to use in the lambdified function.
        If None, all free symbols in the expression will be used.

    Returns
    -------
    function : callable
        A numpy function that computes the value of the expression.
    """
    if sympy.simplify(expression).is_Number and not expression.free_symbols:
        return lambda *args, **kwargs: float(expression)
    if variables is None:
        variables = list(expression.free_symbols)
    function: Callable[..., float | FloatNDArray] = sympy.lambdify(variables, expression)
    return function


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
