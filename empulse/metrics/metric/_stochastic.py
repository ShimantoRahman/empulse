"""
Reducing a stochastic (``sympy.stats``) cost-matrix term to its expectation, and validating one.

:func:`expectation_over_random_vars` is what lets a strategy that only handles deterministic
costs (:class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.Savings`) or that needs to reduce
per-instance costs to a scalar (:class:`~empulse.metrics.MaxProfit`) accept a stochastic cost
matrix at all. :func:`_check_distribution_parameters` and friends validate a random variable's
own shape parameters (e.g. a Beta distribution's ``alpha``/``beta`` must be positive) using the
distribution's own ``check``, so a cost matrix's stochastic terms get that validation for free.

:func:`compiled_expectation` is the one place a term's expectation is computed for evaluation:
what models read as a metric's class costs, and what scales MaxProfit's training penalty.
"""

import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from numbers import Real
from typing import Any

import numpy as np
import sympy
import sympy.stats.crv_types

from ._compile import PicklableLambda, _safe_lambdify
from ._distributions import spec_for
from ._symbolic import _subs_by_name


def _distribution_mean(symbol: sympy.Expr) -> sympy.Expr:
    """Compute the expectation of a single random symbol's distribution."""
    dist = symbol.pspace.distribution
    dist_type = type(dist)

    # Some distributions have a mean `sympy.stats.E` cannot compute.
    spec = spec_for(symbol)
    if spec is not None and spec.mean is not None:
        return spec.mean(dist.args)

    try:
        mean_expr = sympy.stats.E(symbol)
        sympy.lambdify([], mean_expr, modules=['scipy', 'numpy'])
    except (NotImplementedError, TypeError) as error:
        raise NotImplementedError(
            f"Cannot compute or evaluate expectation for random variable '{symbol}'. "
            f"The distribution '{dist_type.__name__}' may not support "
            f'mean computation or lambdification in SymPy.'
        ) from error
    return mean_expr


def expectation_over_random_vars(*expressions: sympy.Expr) -> tuple[sympy.Expr, ...]:
    """
    Replace each expression by its expectation over the stochastic (random) variables in it.

    This allows expressions containing stochastic variables to be treated as deterministic,
    e.g. so that a single scalar cost/benefit value can be derived from them. The random variables are
    assumed independent, as :class:`~empulse.metrics.CostMatrix` documents, so the expectation of a
    polynomial in them is a combination of their moments, and of a multilinear one, such as
    ``gamma * clv``, simply the expression at their means.

    A random variable entering an expression other than polynomially, e.g. through ``exp`` or
    ``Max``, is replaced by its mean with a warning, since that is only an approximation of the
    expectation; so is one whose higher moments sympy cannot compute.

    Parameters
    ----------
    *expressions : sympy.Expr
        One or more expressions potentially containing random symbols.

    Returns
    -------
    tuple of sympy.Expr
        The expectation of each input expression, in the same order.
    """
    all_symbols: set[sympy.Expr] = set()
    for expression in expressions:
        all_symbols |= expression.free_symbols

    random_symbols = sorted((symbol for symbol in all_symbols if sympy.stats.rv.is_random(symbol)), key=str)
    if not random_symbols:
        return expressions

    means = {symbol: _distribution_mean(symbol) for symbol in random_symbols}
    return tuple(_expectation(expression, means) for expression in expressions)


@lru_cache(maxsize=512)
def compiled_expectation(expression: sympy.Expr) -> tuple[sympy.Expr, PicklableLambda]:
    """
    Return the expectation of *expression* over its random variables, and that expectation compiled.

    Worked out once per expression, however many metrics or calls need it: the expectation can take
    sympy a while, and compiling it longer. The expectation of a term does not depend on the other
    terms, since the random variables are independent.
    """
    (expected,) = expectation_over_random_vars(expression)
    return expected, _safe_lambdify(expected)


def _expectation(expression: sympy.Expr, means: dict[sympy.Expr, sympy.Expr]) -> sympy.Expr:
    """Return the expectation of *expression* over independent random variables with these *means*."""
    present = [symbol for symbol in means if expression.has(symbol)]
    if not present:
        return expression
    # Polynomial in the random variables if they can be generators of a polynomial in it.
    generators = {symbol: sympy.Dummy(str(symbol)) for symbol in present}
    try:
        polynomial = sympy.Poly(sympy.expand(expression.xreplace(generators)), *generators.values())
    except sympy.polys.polyerrors.PolynomialError:
        polynomial = None
    if polynomial is not None and all(power <= 1 for monomial in polynomial.monoms() for power in monomial):
        # xreplace matches structurally and is faster than subs(), which is enough for atomic random symbols.
        return expression.xreplace(means)
    if polynomial is not None:
        try:
            return sympy.Add(
                *(
                    coefficient
                    * sympy.Mul(
                        *(
                            _moment(symbol, power, means[symbol])
                            for symbol, power in zip(present, monomial, strict=True)
                        )
                    )
                    for monomial, coefficient in zip(polynomial.monoms(), polynomial.coeffs(), strict=True)
                )
            )
        except NotImplementedError:
            pass
    warnings.warn(
        f'The random variable(s) {", ".join(map(str, present))} in {expression} are replaced by their mean, '
        'which only approximates the expectation of this term: it is not a polynomial in them, or sympy '
        'cannot compute their higher moments.',
        UserWarning,
        stacklevel=2,
    )
    return expression.xreplace(means)


def _moment(symbol: sympy.Expr, power: int, mean: sympy.Expr) -> sympy.Expr:
    """Return ``E[symbol ** power]``, raising ``NotImplementedError`` if sympy cannot compute it."""
    if power == 0:
        return sympy.Integer(1)
    if power == 1:
        return mean
    try:
        moment = sympy.stats.E(symbol**power)
    except (NotImplementedError, TypeError, ValueError) as error:
        raise NotImplementedError(str(error)) from error
    if moment.has(sympy.Integral):
        raise NotImplementedError(f'sympy could not integrate E[{symbol}**{power}].')
    return moment


def _extremes(value: Any) -> tuple[float, float] | None:
    """Return the smallest and largest numeric value in `value`, or None if it is not numeric."""
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.number):
            return None
        return float(value.min()), float(value.max())
    if isinstance(value, Real):
        return float(value), float(value)
    return None


def _check_distribution_parameters(
    expressions: Iterable[sympy.Expr],
    parameters: Mapping[str, Any],
    display: Callable[[str], str],
) -> None:
    """Validate each random variable's shape parameters using the distribution's own ``check``."""
    scalars: dict[str, Any] = {name: value for name, value in parameters.items() if isinstance(value, Real)}
    # A length-1 array applies one value to every sample, so it is checked as the scalar it stands for.
    scalars |= {
        name: float(value[0])
        for name, value in parameters.items()
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.size == 1 and _extremes(value) is not None
    }
    for expression in expressions:
        for random_symbol in expression.atoms(sympy.stats.rv.RandomSymbol):
            distribution = random_symbol.pspace.distribution
            arguments = [_subs_by_name(sympy.sympify(argument), scalars) for argument in distribution.args]
            if all(getattr(argument, 'is_number', False) for argument in arguments):
                _run_distribution_check(distribution, arguments, random_symbol)
            else:
                _check_array_distribution_parameters(distribution, parameters, scalars, random_symbol, display)


def _run_distribution_check(distribution: Any, arguments: list[Any], random_symbol: Any, suffix: str = '') -> None:
    try:
        type(distribution).check(*arguments)
    except ValueError as error:
        name = type(distribution).__name__.removesuffix('Distribution')
        raise ValueError(
            f'Invalid parameters for the {name} distribution of {random_symbol}: {error}{suffix}'
        ) from error


def _check_array_distribution_parameters(
    distribution: Any,
    parameters: Mapping[str, Any],
    scalars: Mapping[str, Any],
    random_symbol: Any,
    display: Callable[[str], str],
) -> None:
    """
    Check instance-dependent shape parameters on the rows where each one is smallest and largest.

    The distribution's own check cannot run on a whole vector, and a check per row would cost a sympy
    call per sample. These distributions impose interval constraints, so a parameter that leaves its
    interval does so at one of its extremes; checking whole rows keeps a constraint that couples two
    parameters (e.g. a uniform's bounds) from being judged on values that never occur together.
    """
    argument_names = {str(symbol) for argument in distribution.args for symbol in sympy.sympify(argument).free_symbols}
    arrays = {
        name: value
        for name, value in parameters.items()
        if name in argument_names and isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 1
    }
    for name, values in arrays.items():
        if _extremes(values) is None:
            continue
        for row in {int(np.argmin(values)), int(np.argmax(values))}:
            row_values = {
                **scalars,
                **{other: float(array[row]) for other, array in arrays.items() if array.shape == values.shape},
            }
            arguments = [_subs_by_name(sympy.sympify(argument), row_values) for argument in distribution.args]
            if all(getattr(argument, 'is_number', False) for argument in arguments):
                _run_distribution_check(
                    distribution, arguments, random_symbol, suffix=f' (from {display(name)}={values[row]})'
                )
