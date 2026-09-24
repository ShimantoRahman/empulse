"""
Reading and rendering the sympy expressions a cost matrix is built from.

Turning a string term into a sympy expression while rejecting sympy's own reserved names
(:func:`_sympify_term`), and rendering an expression as LaTeX with multi-letter symbols set
upright (:func:`_latex`) -- both ends of the same job, translating between what a user writes and
what sympy actually holds. :func:`_subs_by_name` substitutes values given by name, the way callers
pass them, into an expression.
"""

import re
from collections.abc import Mapping
from typing import Any

import sympy
from sympy.printing.conventions import split_super_sub
from sympy.printing.latex import greek_letters_set, other_symbols

# An identifier that is not immediately followed by "(", i.e. used as a value rather than called.
# `exp(clv)` is a legitimate function call; a bare `exp` is almost certainly meant as a parameter.
_BARE_NAME_PATTERN = re.compile(r'\b[A-Za-z_]\w*\b(?!\s*\()')


def _sympify_term(term: sympy.Expr | str) -> sympy.Expr:
    """
    Turn a cost-matrix term into a sympy expression, rejecting names sympy reads as its own.

    ``sympy.sympify`` resolves a number of ordinary-looking identifiers to sympy's own objects
    rather than to free symbols: ``E`` is Euler's number, ``I`` the imaginary unit, ``pi`` is pi,
    and ``beta``/``gamma``/``zeta``/``re``/``im`` are special functions. Several of those are
    plausible names for a cost parameter (``E`` for expense, ``I`` for incentive), and left alone
    they fail in three different unhelpful ways: ``E`` silently becomes the constant 2.718... and
    disappears from the metric's parameter list, ``I`` makes the cost complex, and ``beta`` raises
    a ``TypeError`` about ``FunctionClass`` from deep inside sympy.

    Only strings are checked. An explicitly constructed sympy object is taken at face value, so
    ``add_fp_cost(2 * sympy.pi)`` still means what it says.

    Parameters
    ----------
    term : sympy.Expr or str
        The term to convert. Non-string terms are returned unchanged.

    Returns
    -------
    expression : sympy.Expr
        The converted term.

    Raises
    ------
    ValueError
        If `term` is a string that sympy cannot read, or that uses a name sympy reserves.
    """
    if not isinstance(term, str):
        return term

    reserved = {
        name: value
        for name in dict.fromkeys(_BARE_NAME_PATTERN.findall(term))
        if not isinstance(value := _try_sympify(name), sympy.Symbol) and value is not None
    }
    if reserved:
        first = next(iter(reserved))
        details = ', '.join(f"'{name}' (sympy reads it as {value!r})" for name, value in reserved.items())
        raise ValueError(
            f'Cannot read {term!r} as a cost matrix term: {details}. '
            f'Pass sympy.Symbol({first!r}) if you meant a parameter of that name, '
            f"or sympy.{first} directly if you really meant sympy's own object."
        )

    try:
        expression: sympy.Expr = sympy.sympify(term)
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(f'Cannot read {term!r} as a cost matrix term: {exc}') from exc
    return expression


def _try_sympify(name: str) -> Any | None:
    """Return what sympy makes of a bare identifier, or ``None`` if it cannot read it at all."""
    try:
        return sympy.sympify(name)
    except (sympy.SympifyError, SyntaxError, TypeError):
        return None


def _upright_symbol_name(symbol: sympy.Symbol) -> str | None:
    r"""
    Spell a multi-letter symbol upright, or return ``None`` to let sympy render it as usual.

    sympy sets every symbol in math italic and separates factors with a thin space, so a product
    of multi-letter symbols like ``clv * r`` renders as ``clv r`` and reads as one variable named
    ``clvr``. Setting multi-letter names upright is the usual convention for identifiers and
    resolves the ambiguity without littering the expression with ``\cdot``.

    Greek names are left alone: ``gamma`` should stay ``\gamma``, not ``\mathrm{gamma}``.
    Sub- and superscripts are preserved, so the ``clv_i`` that the strategies build when they
    index a cost per sample still renders as ``\mathrm{clv}_{i}``.
    """
    name, superscripts, subscripts = split_super_sub(str(symbol))
    if len(name) <= 1 or name in greek_letters_set or name in other_symbols:
        return None
    rendered = rf'\mathrm{{{name}}}'
    # One group each, joined by spaces, the way sympy's own printer does it: emitting a separate
    # _{...} per subscript produces `fn_{cost}_{i}`, which is a LaTeX double subscript and fails
    # to typeset at all.
    if superscripts:
        rendered += '^{' + ' '.join(_strip_braces(part) for part in superscripts) + '}'
    if subscripts:
        rendered += '_{' + ' '.join(_strip_braces(part) for part in subscripts) + '}'
    return rendered


def _strip_braces(part: str) -> str:
    """Unwrap a sub/superscript that already carries its own braces, e.g. the `{0}` of `Cost_{0}`."""
    return part[1:-1] if part.startswith('{') and part.endswith('}') else part


def _latex(expression: sympy.Expr) -> str:
    """
    Render a cost-matrix expression as LaTeX.

    Parameters
    ----------
    expression : sympy.Expr
        The expression to render.

    Returns
    -------
    latex : str
        The LaTeX representation, without surrounding math-mode delimiters.
    """
    symbol_names = {}
    for symbol in expression.free_symbols:
        if (upright := _upright_symbol_name(symbol)) is not None:
            symbol_names[symbol] = upright
    return str(sympy.latex(expression, mode='plain', order=None, symbol_names=symbol_names))


def _subs_by_name(expression: Any, values: Mapping[str, Any]) -> Any:
    """
    Substitute values, keyed by symbol name, for the symbols of *expression* with those names.

    ``expression.subs({'a': 1})`` is not the same: sympy turns the name into ``Symbol('a')``, which
    is not equal to a symbol created with assumptions, such as ``Symbol('a', positive=True)``, so
    such symbols were silently left in place. This matches every symbol by its name instead,
    including the parameters of a random variable's distribution.
    """
    replacements = {
        symbol: values[symbol.name] for symbol in sympy.sympify(expression).atoms(sympy.Symbol) if symbol.name in values
    }
    return expression.subs(replacements) if replacements else expression
