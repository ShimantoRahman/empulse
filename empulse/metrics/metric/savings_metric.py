from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray

from .common import MetricFn, _check_parameters
from .cost_metric import _build_cost_equation, _format_cost_function


def _build_savings_score(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    cost_function = _build_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost)
    if any(sympy.stats.rv.is_random(symbol) for symbol in cost_function.free_symbols):
        raise NotImplementedError('Random variables are not supported for the savings metric.')
    all_zero_function, all_one_function = _build_naive_cost_functions(cost_function)

    cost_func = sympy.lambdify(list(cost_function.free_symbols), cost_function)
    all_zero_func = sympy.lambdify(list(all_zero_function.free_symbols), all_zero_function)
    all_one_func = sympy.lambdify(list(all_one_function.free_symbols), all_one_function)
    zero_params_str = {str(symbol) for symbol in all_zero_function.free_symbols}
    one_params_str = {str(symbol) for symbol in all_one_function.free_symbols}

    @_check_parameters(*(tp_benefit + tn_benefit + fp_cost + fn_cost).free_symbols)
    def savings(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
        # by prediction all ones or all zeros, parts of the cost function can be simplified
        zero_parameters = {k: v for k, v in kwargs.items() if k in zero_params_str}
        one_parameters = {k: v for k, v in kwargs.items() if k in one_params_str}
        # it is possible that with the substitution of the symbols, the function becomes a constant
        all_zero_score: float = np.mean(all_zero_func(y=y_true, **zero_parameters)) if all_zero_function != 0 else 0  # type: ignore[assignment]
        all_one_score: float = np.mean(all_one_func(y=y_true, **one_parameters)) if all_one_function != 0 else 0  # type: ignore[assignment]
        cost_base = min(all_zero_score, all_one_score)
        return float(1 - np.mean(cost_func(y=y_true, s=y_score, **kwargs)) / cost_base)

    return savings


def _build_naive_cost_functions(cost_function: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    all_zero_function = cost_function.subs('s', 0)
    all_one_function = cost_function.subs('s', 1)
    return all_zero_function, all_one_function


def _savings_score_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    i, N, c0, c1 = sympy.symbols('i N Cost_{0} Cost_{1}')  # noqa: N806
    savings_function = (1 / (N * sympy.Min(c0, c1))) * sympy.Sum(
        _format_cost_function(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 0, N)
    )

    for symbol in savings_function.free_symbols:
        if symbol not in {N, c0, c1}:
            savings_function = savings_function.subs(symbol, str(symbol) + '_i')

    output = latex(savings_function, mode='plain', order=None)

    return f'$\\displaystyle {output}$'
