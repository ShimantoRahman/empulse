from collections.abc import Callable, Iterable, Mapping
from enum import Enum, auto
from typing import Any, ParamSpec, Protocol, TypeVar

import numpy as np
import sympy
import sympy.stats.crv_types

from ..._types import FloatNDArray, IntNDArray

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

# Mapping from distribution type to its closed-form mean expression.
# Used as a fast, reliable fallback for distributions whose expectation
# sympy.stats.E cannot compute in closed form.
_FIXED_MEANS: dict[
    type[sympy.stats.crv_types.SingleContinuousDistribution],
    Callable[[tuple[sympy.Expr, ...]], sympy.Expr],
] = {
    sympy.stats.crv_types.ArcsinDistribution: lambda params: (params[0] + params[1]) / 2,
    sympy.stats.crv_types.BetaPrimeDistribution: lambda params: params[0] / (params[1] - 1),
    sympy.stats.crv_types.StudentTDistribution: lambda params: 0,
    sympy.stats.crv_types.FDistributionDistribution: lambda params: params[1] / (params[1] - 2),
    sympy.stats.crv_types.GammaInverseDistribution: lambda params: params[1] / (params[0] - 1),
    sympy.stats.crv_types.LogNormalDistribution: lambda params: sympy.exp(params[0] + params[1] ** 2 / 2),
    sympy.stats.crv_types.LomaxDistribution: lambda params: params[1] / (params[0] - 1),
    sympy.stats.crv_types.ParetoDistribution: lambda params: (params[1] * params[0]) / (params[1] - 1),
    sympy.stats.crv_types.PowerFunctionDistribution: (
        lambda params: params[1] + params[0] * (params[2] - params[1]) / (params[0] + 1)
    ),
}


def _distribution_mean(symbol: sympy.Expr) -> sympy.Expr:
    """Compute the expectation of a single random symbol's distribution."""
    dist = symbol.pspace.distribution
    dist_type = type(dist)

    if dist_type in _FIXED_MEANS:
        return _FIXED_MEANS[dist_type](dist.args)

    try:
        mean_expr = sympy.stats.E(symbol)
        # Verify the expectation can actually be evaluated numerically.
        sympy.lambdify([], mean_expr, modules=['scipy', 'numpy'])
    except (NotImplementedError, TypeError) as error:
        raise NotImplementedError(
            f"Cannot compute or evaluate expectation for random variable '{symbol}'. "
            f"The distribution '{dist_type.__name__}' may not support "
            f'mean computation or lambdification in SymPy.'
        ) from error
    return mean_expr


def replace_random_var_with_mean(*expressions: sympy.Expr) -> tuple[sympy.Expr, ...]:
    """
    Replace stochastic (random) variables in the expressions with their expectation (mean).

    This allows expressions containing stochastic variables to be treated as deterministic,
    e.g. so that a single scalar cost/benefit value can be derived from them.

    Parameters
    ----------
    *expressions : sympy.Expr
        One or more expressions potentially containing random symbols.

    Returns
    -------
    tuple of sympy.Expr
        The input expressions, in the same order, with all random symbols substituted by their mean.
    """
    all_symbols: set[sympy.Expr] = set()
    for expression in expressions:
        all_symbols |= expression.free_symbols

    random_symbols = [symbol for symbol in all_symbols if sympy.stats.rv.is_random(symbol)]
    if not random_symbols:
        return expressions

    subs_map = {symbol: _distribution_mean(symbol) for symbol in random_symbols}

    # xreplace performs exact structural matching, which is faster than subs()
    # here since we are only replacing atomic random symbols (no pattern matching needed).
    return tuple(expression.xreplace(subs_map) for expression in expressions)


class Direction(Enum):
    """Optimization direction of metric."""

    MAXIMIZE = auto()
    MINIMIZE = auto()


# Symbol names used internally by Metric and its strategies to inject data (labels, scores,
# true/false positive rates, class priors) into the lambdified cost-matrix expressions, or used
# by to_latex() rendering, or as a keyword-only parameter of Metric._prepare_parameters(). A
# user-defined symbol or alias sharing one of these names would either be silently fused with the
# internal one, raise a confusing internal TypeError, or (for 'n_samples') actually be captured by
# _prepare_parameters()'s own n_samples parameter instead of reaching its **kwargs - see
# Metric.__init__ and _check_reserved_symbol_names().
RESERVED_SYMBOL_NAMES = frozenset({'y', 's', 'F_0', 'F_1', 'pi_0', 'pi_1', 'N', 'i', 'n_samples'})


def _check_reserved_symbol_names(*expressions: sympy.Expr, alias_names: Iterable[str] = ()) -> None:
    """
    Raise if a cost matrix uses a symbol name (or alias) reserved for internal use.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) to check for reserved free symbol names.
    alias_names : Iterable[str], default=()
        The alias names registered on the cost matrix, checked alongside the expressions' own
        free symbol names.

    Raises
    ------
    ValueError
        If any free symbol name or alias collides with a name in :data:`RESERVED_SYMBOL_NAMES`.
    """
    symbol_names = {str(symbol) for expression in expressions for symbol in expression.free_symbols}
    reserved_in_use = (symbol_names | set(alias_names)) & RESERVED_SYMBOL_NAMES
    if reserved_in_use:
        raise ValueError(
            f'The cost matrix uses symbol name(s) or alias(es) {sorted(reserved_in_use)} that are reserved '
            f'for internal use by Metric and its strategies. Reserved names: {sorted(RESERVED_SYMBOL_NAMES)}. '
            'Please rename the corresponding symbol(s) or alias(es).'
        )


def _collect_known_symbol_names(*expressions: sympy.Expr) -> set[str]:
    """
    Collect the names of every symbol a cost matrix's expressions can be evaluated with.

    This includes each expression's own free symbols (deterministic symbols and, for a
    stochastic expression, the random variable's own symbol, e.g. ``gamma`` for
    ``sympy.stats.Beta('gamma', alpha, beta)``) as well as any random variable's distribution
    parameters (e.g. ``alpha``/``beta`` above), since those are what a caller actually supplies
    a value for at call time.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) to collect symbol names from.

    Returns
    -------
    set of str
        The names of every symbol found.
    """
    free_symbols: set[sympy.Expr] = set()
    for expression in expressions:
        free_symbols |= expression.free_symbols

    stochastic_params: set[sympy.Expr] = set()
    for expression in expressions:
        for atom in expression.atoms(sympy.stats.rv.RandomSymbol):
            pspace = atom.pspace
            if hasattr(pspace, 'distribution') and hasattr(pspace.distribution, 'args'):
                for arg in pspace.distribution.args:
                    stochastic_params.update(arg.free_symbols)

    return {str(symbol) for symbol in free_symbols | stochastic_params}


def _check_known_alias_and_default_targets(
    *expressions: sympy.Expr, aliases: Mapping[str, str | sympy.Symbol], default_names: Iterable[str]
) -> None:
    """
    Raise if an alias's target, or a default's parameter name, matches no symbol in the cost matrix.

    ``CostMatrix`` is built incrementally, so ``alias()`` and ``set_default()`` cannot validate
    their arguments against the cost matrix's symbols at the time they are called. ``Metric``
    can, once the cost matrix is complete, and does so here - turning a typo'd alias target, or a
    ``set_default()`` call for an alias that had not been registered yet (see the ordering note
    in :meth:`~empulse.metrics.CostMatrix.set_default`), into one clear error at construction
    time instead of a confusing "did not receive it" error only once the metric is called.

    Parameters
    ----------
    *expressions : sympy.Expr
        The cost-matrix expressions (e.g. ``tp_benefit``, ``tn_benefit``, ``fp_cost``,
        ``fn_cost``) whose symbols are considered valid alias targets / default names.
    aliases : Mapping[str, str]
        The cost matrix's registered aliases, mapping alias name to target symbol name.
    default_names : Iterable[str]
        The cost matrix's registered default parameter names (already resolved to symbol names,
        for aliases registered before the corresponding ``set_default()`` call).

    Raises
    ------
    ValueError
        If any alias target or default name does not match a known symbol name.
    """
    known_names = _collect_known_symbol_names(*expressions)

    unknown_alias_targets = {
        f'{alias!r} -> {str(target)!r}' for alias, target in aliases.items() if str(target) not in known_names
    }
    unknown_defaults = {name for name in default_names if name not in known_names}

    if unknown_alias_targets or unknown_defaults:
        messages = []
        if unknown_alias_targets:
            messages.append(f'alias(es) {sorted(unknown_alias_targets)} whose target is not a cost matrix symbol')
        if unknown_defaults:
            messages.append(f'default(s) for {sorted(unknown_defaults)}, which is not a cost matrix symbol or alias')
        raise ValueError(
            f'The cost matrix has {" and ".join(messages)}. Known symbol names: {sorted(known_names)}. '
            'This is usually a typo, or set_default() was called with an alias name before that alias '
            'was registered with alias() - aliases must be registered before set_default() can resolve '
            'them (see the CostMatrix.set_default() docstring).'
        )


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
            # Sort free_symbols by name for deterministic variable ordering,
            # ensuring the lambdified function receives kwargs correctly.
            variables = sorted(self.expression.free_symbols, key=str) if self.variables is None else self.variables
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
