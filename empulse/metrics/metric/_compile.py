"""
Compiling a sympy cost expression to a numpy function, and calling it safely.

:class:`PicklableLambda` wraps :func:`sympy.lambdify` so the result survives pickling (a plain
lambdified function does not: ``pickle`` cannot serialize the generated closure) -- what lets a
:class:`~empulse.metrics.Metric` built once and compiled here stay picklable, e.g. for
``n_jobs`` parallelism. :func:`_safe_run_lambda` and :func:`_safe_run_lambda_array` call a
compiled function with only the keyword arguments it actually uses, via :func:`_filter_parameters`,
since a cost expression's compiled function only accepts the free symbols it was built from.
"""

import copy
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any, Protocol, TypeVar

import numpy as np
import sympy

from ..._types import FloatNDArray, IntNDArray

T = TypeVar('T')


class MetricFn(Protocol):
    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float: ...


class LogitConsts(Protocol):
    def prepare(
        self, x: FloatNDArray, y_true: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]: ...


class BoostGradientConst(Protocol):
    def __call__(self, y_true: FloatNDArray, **kwargs: Any) -> FloatNDArray: ...


class ThresholdFn(Protocol):
    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> FloatNDArray | float: ...


class RateFn(Protocol):
    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float: ...


class CountScoreFn(Protocol):
    """A score of samples grouped by score, with one entry per group (e.g. per leaf of a tree)."""

    def __call__(self, y_score: FloatNDArray, n_positive: IntNDArray, n_negative: IntNDArray) -> float: ...


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
    try:
        free_symbols = _free_symbol_names(expression)
    except TypeError:  # an unhashable expression, e.g. a mutable sympy Matrix
        free_symbols = frozenset(str(symbol) for symbol in expression.free_symbols)
    filtered_parameters = {key: value for key, value in parameters.items() if key in free_symbols}
    return filtered_parameters


@lru_cache(maxsize=4096)
def _free_symbol_names(expression: sympy.Expr) -> frozenset[str]:
    """
    Return the names of the free symbols in *expression*.

    Cached because compiled cost expressions are evaluated over and over with the same expression
    (e.g. once per candidate model in a training loop), and walking the expression tree and
    printing each symbol cost more than evaluating the compiled function itself.
    """
    return frozenset(str(symbol) for symbol in expression.free_symbols)


class PicklableLambda:
    """
    A callable wrapper that securely pickles lambdified Sympy functions.

    With ``dummify=True`` the arguments get private names in the generated code, so a symbol named
    after a function the expression calls, such as a parameter ``beta`` beside the Beta function
    ``beta(alpha, beta)``, does not shadow it.
    """

    func: Callable[..., Any]

    def __init__(
        self, expression: sympy.Expr, variables: Iterable[sympy.Symbol] | None = None, *, dummify: bool = False
    ):
        self.expression = expression
        self.variables = variables
        self.dummify = dummify
        self._compile()

    def _compile(self) -> None:
        if not self.expression.free_symbols:
            val = float(self.expression.evalf())
            self.func = lambda *args, **kwargs: val
        else:
            # Sort free_symbols by name for deterministic variable ordering,
            # ensuring the lambdified function receives kwargs correctly.
            variables = sorted(self.expression.free_symbols, key=str) if self.variables is None else self.variables
            expression = self.expression
            # Instances pickled by older versions lack `dummify`.
            if getattr(self, 'dummify', False):
                # Substituted here rather than with lambdify's own `dummify`, which still binds the
                # symbols' names in the function's namespace, where they shadow functions of the same name.
                dummies = [sympy.Dummy() for _ in variables]
                expression = expression.xreplace(dict(zip(variables, dummies, strict=True)))
                variables = dummies
                # Fresh dummies are never equal across compiles, so the cached `_lambdify` cannot help here.
                self.func = sympy.lambdify(variables, expression)  # type: ignore[assignment]
            else:
                self.func = _lambdify(tuple(variables), expression)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __deepcopy__(self, memo: dict[int, Any]) -> 'PicklableLambda':
        # The compiled function is stateless and the expression immutable, so a copy shares both
        # instead of compiling again, as the default deepcopy through pickling would. Cloning an
        # estimator deep-copies its loss, so this saves one compile per clone in a grid search.
        new = copy.copy(self)
        memo[id(self)] = new
        return new

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop('func', None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._compile()


def _lambdify(variables: tuple[sympy.Symbol, ...], expression: sympy.Expr) -> Callable[..., Any]:
    """
    Compile *expression* with :func:`sympy.lambdify`, reusing the function compiled for an equal one.

    Compiling prints the expression to source code and executes it, which costs milliseconds, and
    every unpickled metric (e.g. one per parallel worker) would otherwise pay that again for each of
    its expressions. The compiled functions are stateless, so sharing one between metrics is safe.
    """
    try:
        return _cached_lambdify(variables, expression)
    except TypeError:  # an unhashable expression, e.g. a mutable sympy Matrix
        return sympy.lambdify(variables, expression)  # type: ignore[no-any-return]


@lru_cache(maxsize=1024)
def _cached_lambdify(variables: tuple[sympy.Symbol, ...], expression: sympy.Expr) -> Callable[..., Any]:
    return sympy.lambdify(variables, expression)  # type: ignore[no-any-return]


def _safe_lambdify(
    expression: sympy.Expr, variables: Iterable[sympy.Symbol] | None = None, *, dummify: bool = False
) -> PicklableLambda:
    """Safely lambdify a sympy expression and return a picklable callable."""
    return PicklableLambda(expression, variables, dummify=dummify)


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
    **parameters : float or NDArray of shape (n_samples,)
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
    **parameters : float or NDArray of shape (n_samples,)
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
