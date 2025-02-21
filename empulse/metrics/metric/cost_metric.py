from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray

from .common import MetricFn


def _build_cost_loss(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> MetricFn:
    cost_function = _build_cost_function(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost)
    if any(sympy.stats.rv.is_random(symbol) for symbol in cost_function.free_symbols):
        raise NotImplementedError('Random variables are not supported for the cost metric.')
    cost_funct = sympy.lambdify(list(cost_function.free_symbols), cost_function)

    def cost_loss(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
        return float(np.mean(cost_funct(y=y_true, s=y_score, **kwargs)))

    return cost_loss


def _build_cost_function(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    y, s = sympy.symbols('y s')
    cost_function = y * (s * tp_cost + (1 - s) * fn_cost) + (1 - y) * ((1 - s) * tn_cost + s * fp_cost)
    return cost_function


def _cost_loss_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    i, N = sympy.symbols('i N')  # noqa: N806
    cost_function = (1 / N) * sympy.Sum(
        _format_cost_function(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 0, N)
    )

    for symbol in cost_function.free_symbols:
        if symbol != N:
            cost_function = cost_function.subs(symbol, str(symbol) + '_i')

    output = latex(cost_function, mode='plain', order=None)

    return f'$\\displaystyle {output}$'


def _format_cost_function(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    y, s = sympy.symbols('y s')
    cost_function = y * (s * tp_cost + (1 - s) * fn_cost) + (1 - y) * ((1 - s) * tn_cost + s * fp_cost)
    return cost_function
