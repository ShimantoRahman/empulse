from typing import Any

import numpy as np
import sympy

from ..._types import FloatNDArray
from .common import (
    BoostObjective,
    LogitObjective,
    MetricFn,
    RateFn,
    ThresholdFn,
    _check_parameters,
    _filter_parameters,
)


def _build_cost_loss(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
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


def _build_cost_prepare_logit_objective(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> LogitObjective:
    y, x = sympy.symbols('y x')
    gradient_const = x * (y * (-tp_benefit - fn_cost) + (1 - y) * (fp_cost + tn_benefit))
    gradient_fn = sympy.lambdify(list(gradient_const.free_symbols), gradient_const)
    loss_const1 = y * -tp_benefit - fp_cost + fp_cost * y
    if isinstance(loss_const1, sympy.core.numbers.Zero):
        loss_const1_fn = lambda y, **kwargs: np.zeros(y.shape, dtype=np.float64)
    else:
        loss_const1_fn = sympy.lambdify(list(loss_const1.free_symbols), loss_const1)
    loss_const2 = y * fn_cost - tn_benefit + tn_benefit * y
    if isinstance(loss_const2, sympy.core.numbers.Zero):
        loss_const2_fn = lambda y, **kwargs: np.zeros(y.shape, dtype=np.float64)
    else:
        loss_const2_fn = sympy.lambdify(list(loss_const2.free_symbols), loss_const2)

    def cost_gradient_logit_consts(
        x: FloatNDArray, y_true: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
        gradient_const: FloatNDArray = gradient_fn(y=y_true, x=x, **kwargs)
        loss_const1_value: FloatNDArray = loss_const1_fn(y=y_true, **kwargs)
        loss_const2_value: FloatNDArray = loss_const2_fn(y=y_true, **kwargs)
        return gradient_const, loss_const1_value, loss_const2_value

    return cost_gradient_logit_consts


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


def _build_cost_prepare_gradient_boost_objective(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> BoostObjective:
    y = sympy.symbols('y')
    gradient_const = y * (-tp_benefit - fn_cost) + (1 - y) * (fp_cost + tn_benefit)
    gradient_const_fn = sympy.lambdify(list(gradient_const.free_symbols), gradient_const)

    def cost_gradient_const(y_true: FloatNDArray, **kwargs: Any) -> FloatNDArray:
        return gradient_const_fn(y=y_true, **kwargs)

    return cost_gradient_const


def _build_cost_optimal_threshold(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
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


def _build_cost_optimal_rate(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
) -> RateFn:
    denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
    numerator_expression = fp_cost + tn_benefit
    calculate_denominator = sympy.lambdify(list(denominator_expression.free_symbols), denominator_expression)
    calculate_numerator = sympy.lambdify(list(numerator_expression.free_symbols), numerator_expression)

    @_check_parameters(*denominator_expression.free_symbols)
    def rate_function(y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any) -> float:
        parameters = {
            key: float(np.mean(parameter)) if isinstance(parameter, np.ndarray) else parameter
            for key, parameter in parameters.items()
        }
        denominator_parameters = _filter_parameters(denominator_expression, parameters)
        numerator_parameters = _filter_parameters(numerator_expression, parameters)
        denominator = calculate_denominator(**denominator_parameters)
        numerator = calculate_numerator(**numerator_parameters)
        # Avoid division by zero
        if denominator == 0.0:
            denominator += float(np.finfo(float).eps)
        optimal_threshold: FloatNDArray | float = numerator / denominator
        optimal_rate = float(np.mean(y_score > optimal_threshold))
        return optimal_rate

    return rate_function


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
