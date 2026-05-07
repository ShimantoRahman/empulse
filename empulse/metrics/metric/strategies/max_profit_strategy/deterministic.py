from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import sympy
from scipy.special import expit

from ....._types import FloatNDArray, IntNDArray
from ....common import classification_threshold
from ...common import _check_parameters, _safe_lambdify, _safe_run_lambda
from .common import _convex_hull


def _calculate_profits_deterministic(
    y_true: IntNDArray,
    y_score: FloatNDArray,
    calculate_profit: Callable[..., float],
    profit_function: sympy.Expr,
    **kwargs: Any,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray, float, float]:
    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0
    tprs, fprs = _convex_hull(y_true, y_score)

    profits = np.zeros_like(tprs)
    for i, (tpr, fpr) in enumerate(zip(tprs, fprs, strict=False)):
        eval_params = {'pi_0': pi0, 'pi_1': pi1, 'F_0': tpr, 'F_1': fpr, **kwargs}
        profits[i] = _safe_run_lambda(calculate_profit, profit_function, **eval_params)

    return profits, tprs, fprs, pi0, pi1


class MaxProfitScoreDeterministic:
    """Compute the maximum profit for all deterministic variables."""

    def __init__(self, profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = _safe_lambdify(profit_function)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        profits, *_ = _calculate_profits_deterministic(
            y_true, y_score, self.calculate_profit, self.profit_function, **kwargs
        )
        return float(profits.max())


def _calculate_optimal_rate_deterministic(
    y_true: IntNDArray,
    y_score: FloatNDArray,
    calculate_profit: Callable[..., float],
    profit_function: sympy.Expr,
    **kwargs: Any,
) -> float:
    profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(
        y_true, y_score, calculate_profit, profit_function, **kwargs
    )
    best_index = np.argmax(profits)
    return float(tprs[best_index] * pi0 + fprs[best_index] * pi1)


class MaxProfitRateDeterministic:
    """Compute the maximum profit for all deterministic variables."""

    def __init__(self, profit_function: sympy.Expr, deterministic_symbols: Iterable[sympy.Symbol]) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = _safe_lambdify(profit_function)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        return _calculate_optimal_rate_deterministic(
            y_true, y_score, self.calculate_profit, self.profit_function, **kwargs
        )


class MaxProfitBoostGradientDeterministic:
    """Prepared deterministic objective for MaxProfit gradient boosting."""

    def __init__(
        self,
        *,
        profit_function: sympy.Expr,
        deterministic_symbols: Iterable[sympy.Symbol],
        y_true: FloatNDArray,
        tp_benefit: float,
        tn_benefit: float,
        fp_cost: float,
        fn_cost: float,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = _safe_lambdify(profit_function)

        self.y_true = np.asarray(y_true).reshape(-1).astype(np.int32)
        self.tp_benefit = tp_benefit
        self.tn_benefit = tn_benefit
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.parameters = parameters

        self.n_pos = max(int(np.sum(self.y_true == 1)), 1)
        self.n_neg = max(int(np.sum(self.y_true == 0)), 1)

    def __call__(self, y_score: FloatNDArray, alpha: float) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the deterministic objective."""
        y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)

        profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(
            self.y_true,
            y_score_arr,
            self.calculate_profit,
            self.profit_function,
            **self.parameters,
        )
        best_idx = int(np.argmax(profits))
        rate = float(tprs[best_idx] * pi0 + fprs[best_idx] * pi1)
        threshold = float(classification_threshold(self.y_true, y_score_arr, rate))

        sigma = expit(alpha * (y_score_arr - threshold))
        sigma_prime = alpha * sigma * (1.0 - sigma)
        sigma_second = alpha**2 * sigma * (1.0 - sigma) * (1.0 - 2.0 * sigma)

        c_pos = -((self.tp_benefit + self.fn_cost) * pi0) / self.n_pos
        c_neg = ((self.tn_benefit + self.fp_cost) * pi1) / self.n_neg
        instance_weight = np.where(self.y_true == 1, c_pos, c_neg)

        gradient = instance_weight * sigma_prime
        hessian = np.abs(instance_weight * sigma_second)
        return gradient, hessian


# TODO: RESET NECESSARY FOR EPOCH
class MaxProfitLogitGradientDeterministic:
    """Picklable objective for deterministic MaxProfit optimized with logistic models."""

    def __init__(
        self,
        *,
        profit_function: sympy.Expr,
        deterministic_symbols: Iterable[sympy.Symbol],
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        alpha_0: float,
        alpha_growth: float,
        alpha_max: float,
        tp_benefit: float,
        tn_benefit: float,
        fp_cost: float,
        fn_cost: float,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = _safe_lambdify(profit_function)

        self.features = features
        self.y_true = y_true.ravel().astype(np.int32)
        self.C = C
        self.l1_ratio = l1_ratio
        self.soft_threshold = soft_threshold
        self.fit_intercept = fit_intercept
        self.alpha_0 = alpha_0
        self.alpha_growth = alpha_growth
        self.alpha_max = alpha_max
        self._epoch = 0
        self.tp_benefit = tp_benefit
        self.tn_benefit = tn_benefit
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.parameters = parameters

        self.pos_mask = self.y_true == 1
        self.neg_mask = ~self.pos_mask
        self.n_pos = int(self.pos_mask.sum())
        self.n_neg = int(self.neg_mask.sum())
        self.X_pos = self.features[self.pos_mask]
        self.X_neg = self.features[self.neg_mask]

    def _current_alpha(self) -> float:
        """Compute annealed temperature for the current objective evaluation."""
        try:
            alpha = self.alpha_0 * (self.alpha_growth**self._epoch)
        except OverflowError:
            alpha = self.alpha_max
        return float(min(self.alpha_max, alpha))

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated max-profit objective and gradient for minimization."""
        start_coef = 1 if self.fit_intercept else 0

        w = np.asarray(weights, dtype=np.float64).copy()
        if self.soft_threshold:
            abs_w = np.abs(w[start_coef:])
            diff = abs_w - self.C
            w[start_coef:] = np.where(
                diff > 0,
                np.sign(w[start_coef:]) * diff,
                np.where(diff < 0, 0.0, w[start_coef:]),
            )

        alpha = self._current_alpha()
        self._epoch += 1

        y_score = expit(self.features @ w)
        profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(
            self.y_true,
            y_score.astype(np.float64),
            self.calculate_profit,
            self.profit_function,
            **self.parameters,
        )
        best_idx = int(np.argmax(profits))

        rate = float(tprs[best_idx] * pi0 + fprs[best_idx] * pi1)
        threshold = float(classification_threshold(self.y_true, y_score, rate))

        s_pos = y_score[self.pos_mask]
        s_neg = y_score[self.neg_mask]

        sig_pos = expit(alpha * (s_pos - threshold))
        sig_neg = expit(alpha * (s_neg - threshold))
        dsig_pos = sig_pos * (1.0 - sig_pos)
        dsig_neg = sig_neg * (1.0 - sig_neg)

        sd_pos = s_pos * (1.0 - s_pos)
        sd_neg = s_neg * (1.0 - s_neg)

        grad_tpr = (alpha / self.n_pos) * ((dsig_pos * sd_pos)[:, None] * self.X_pos).sum(axis=0)
        grad_fpr = (alpha / self.n_neg) * ((dsig_neg * sd_neg)[:, None] * self.X_neg).sum(axis=0)

        coeff_tpr = (self.tp_benefit + self.fn_cost) * pi0
        coeff_fpr = (self.tn_benefit + self.fp_cost) * pi1
        grad_profit = coeff_tpr * grad_tpr - coeff_fpr * grad_fpr

        value = float(-profits[best_idx])
        gradient = np.asarray(-grad_profit, dtype=np.float64)

        coef = w[start_coef:]
        if self.l1_ratio == 0.0:
            gradient[start_coef:] += coef / self.C
            value += 0.5 * float(np.dot(coef, coef)) / self.C
        elif self.l1_ratio == 1.0:
            gradient[start_coef:] += np.sign(coef) / self.C
            value += float(np.sum(np.abs(coef))) / self.C
        else:
            gradient[start_coef:] += ((1.0 - self.l1_ratio) * coef + self.l1_ratio * np.sign(coef)) / self.C
            value += (
                (1.0 - self.l1_ratio) * 0.5 * float(np.dot(coef, coef)) + self.l1_ratio * float(np.sum(np.abs(coef)))
            ) / self.C

        return value, gradient
