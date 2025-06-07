from typing import Any

import numpy as np
import sympy

from ..._types import FloatNDArray
from .common import (
    BoostObjective,
    LogitObjective,
    MetricFn,
    ThresholdFn,
    _check_parameters,
    _filter_parameters,
)


def _build_cost_loss(
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
        raise NotImplementedError('Random variables are not supported for the cost metric.')
    cost_funct = sympy.lambdify(list(cost_function.free_symbols), cost_function)

    @_check_parameters(*(tp_benefit + tn_benefit + fp_cost + fn_cost).free_symbols)
    def cost_loss(y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        return float(np.mean(cost_funct(y=y_true, s=y_score, **kwargs)))

    return cost_loss


def _build_cost_equation(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    y, s = sympy.symbols('y s')
    cost_function = y * (s * tp_cost + (1 - s) * fn_cost) + (1 - y) * ((1 - s) * tn_cost + s * fp_cost)
    return cost_function


def _build_cost_logit_objective(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> LogitObjective:
    y, s, x = sympy.symbols('y s x')
    gradient = x * s * (1 - s) * (y * (-tp_benefit - fn_cost) + (1 - y) * (fp_cost + tn_benefit))
    gradient_fn = sympy.lambdify(list(gradient.free_symbols), gradient)

    def cost_gradient_logit(
        x: FloatNDArray, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> FloatNDArray:
        gradient: FloatNDArray = np.mean(gradient_fn(y=y_true, s=y_score, x=x, **kwargs), axis=0)
        return gradient

    return cost_gradient_logit


def _build_cost_gradient_boost_objective(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> BoostObjective:
    y, s, nabla = sympy.symbols('y s nabla')
    gradient = s * (1 - s) * (y * (-tp_benefit - fn_cost) + (1 - y) * (fp_cost + tn_benefit))
    gradient_fn = sympy.lambdify(list(gradient.free_symbols), gradient)
    hessian = (1 - 2 * s) * nabla
    hessian_fn = sympy.lambdify(list(hessian.free_symbols), hessian)

    def cost_gradient_hessian_gboost(
        y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray]:
        gradient = gradient_fn(y=y_true, s=y_score, **kwargs)
        hessian = np.abs(hessian_fn(s=y_score, nabla=gradient))
        return gradient, hessian

    return cost_gradient_hessian_gboost


def _build_cost_optimal_threshold(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.RandomState,
) -> ThresholdFn:
    denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
    numerator_expression = fp_cost + tn_benefit
    calculate_denominator = sympy.lambdify(list(denominator_expression.free_symbols), denominator_expression)
    calculate_numerator = sympy.lambdify(list(numerator_expression.free_symbols), numerator_expression)

    @_check_parameters(*denominator_expression.free_symbols)
    def threshold_function(y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any) -> FloatNDArray | float:
        denominator_parameters = _filter_parameters(denominator_expression, parameters)
        numerator_parameters = _filter_parameters(numerator_expression, parameters)
        denominator = calculate_denominator(**denominator_parameters)
        numerator = calculate_numerator(**numerator_parameters)
        # Avoid division by zero
        if isinstance(denominator, float | int):
            if denominator == 0:
                denominator += float(np.finfo(float).eps)
        else:
            denominator = np.clip(denominator, float(np.finfo(float).eps), denominator)
        optimal_thresholds: FloatNDArray | float = numerator / denominator
        return optimal_thresholds

    return threshold_function


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
