import sys
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import numpy as np
import sympy
from sympy.utilities import lambdify

from ....._types import FloatNDArray
from ...common import SympyFnPickleMixin, _check_parameters
from .common import _convex_hull

if sys.version_info >= (3, 11):
    pass


def _calculate_profits_deterministic(
    y_true: FloatNDArray, y_score: FloatNDArray, calculate_profit: Callable[..., float], **kwargs: Any
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray, float, float]:
    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0
    tprs, fprs = _convex_hull(y_true, y_score)

    profits = np.zeros_like(tprs)
    for i, (tpr, fpr) in enumerate(zip(tprs, fprs, strict=False)):
        profits[i] = calculate_profit(pi_0=pi0, pi_1=pi1, F_0=tpr, F_1=fpr, **kwargs)

    return profits, tprs, fprs, pi0, pi1


class MaxProfitScoreDeterministic(SympyFnPickleMixin):
    """Compute the maximum profit for all deterministic variables."""

    _sympy_functions: ClassVar[dict[str, str]] = {'calculate_profit': 'profit_function'}

    def __init__(self, profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        profits, *_ = _calculate_profits_deterministic(y_true, y_score, self.calculate_profit, **kwargs)
        return float(profits.max())


def _calculate_optimal_rate_deterministic(
    y_true: FloatNDArray, y_score: FloatNDArray, calculate_profit: Callable[..., float], **kwargs: Any
) -> float:
    profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(y_true, y_score, calculate_profit, **kwargs)
    best_index = np.argmax(profits)
    return float(tprs[best_index] * pi0 + fprs[best_index] * pi1)


class MaxProfitRateDeterministic(SympyFnPickleMixin):
    """Compute the maximum profit for all deterministic variables."""

    _sympy_functions: ClassVar[dict[str, str]] = {'calculate_profit': 'profit_function'}

    def __init__(self, profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        return _calculate_optimal_rate_deterministic(y_true, y_score, self.calculate_profit, **kwargs)
