from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import sympy

from ....._types import FloatNDArray, IntNDArray
from ...._cy_convex_hull import max_profit_scan
from ....common import classification_threshold
from ..._compile import _safe_lambdify, _safe_run_lambda
from ..._confusion import _compute_confusion_matrix
from ..._parameter_domain import _check_parameters
from .common import _BaseMaxProfitLogitObjective, _smooth_step_derivatives


def _calculate_profits_deterministic(
    y_true: IntNDArray,
    y_score: FloatNDArray,
    calculate_profit: Callable[..., float],
    profit_function: sympy.Expr,
    **kwargs: Any,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray, float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0

    # not taking convex hull here since it is cheaper to just check every point than computing the convex hull
    n_pos = float(np.sum(y_true))
    n_neg = y_true.shape[0] - n_pos
    confusion_counts, _, _ = _compute_confusion_matrix(y_true, y_score)
    if n_pos > 0 and n_neg > 0:
        tprs = confusion_counts[0] / n_pos
        fprs = confusion_counts[1] / n_neg
    else:
        # Without positives (or without negatives) the true (false) positive rate is 0/0. It is
        # weighted by a class prior of zero, so any value gives the same profit: take the rate of
        # the class that is present, so that both equal the fraction of samples targeted.
        targeted = (confusion_counts[0] + confusion_counts[1]) / y_true.shape[0]
        tprs, fprs = targeted, targeted.copy()

    eval_params = {'pi_0': pi0, 'pi_1': pi1, 'F_0': tprs, 'F_1': fprs, **kwargs}
    profits = np.asarray(_safe_run_lambda(calculate_profit, profit_function, **eval_params), dtype=np.float64)
    if profits.shape != tprs.shape:
        # profit_function doesn't actually depend on F_0 and/or F_1 (e.g. a degenerate cost
        # matrix where the tp/fn or tn/fp terms cancel out), so the lambdified call collapsed
        # to a scalar. Broadcast it back out to one value per convex-hull point.
        profits = np.full(tprs.shape, float(profits), dtype=np.float64)

    return profits, tprs, fprs, pi0, pi1


class _BaseMaxProfitDeterministic:
    """Shared setup for MaxProfit's fully-deterministic (no stochastic variable) score/rate.

    Subclasses differ only in which of the two results of the maximization they report:
    :class:`MaxProfitScoreDeterministic` the highest profit, :class:`MaxProfitRateDeterministic` the
    predicted-positive rate that reaches it.

    Given the four class terms the profit is built from, the maximization runs in the compiled
    :func:`~empulse.metrics._cy_convex_hull.max_profit_scan`. Without them the profit function is
    evaluated at every threshold instead.
    """

    def __init__(
        self,
        profit_function: sympy.Expr,
        deterministic_symbols: Iterable[sympy.Symbol],
        class_terms: tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr] | None = None,
    ) -> None:
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = _safe_lambdify(profit_function)
        self.class_terms = class_terms
        self._class_term_fns = None if class_terms is None else tuple(_safe_lambdify(term) for term in class_terms)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters((*self.deterministic_symbols,), kwargs)
        return self._maximize(y_true, y_score, kwargs)

    def _sample_scorer(self, y_true: IntNDArray, **kwargs: Any) -> Callable[[FloatNDArray], float]:
        """
        Prepare the score of fixed labels, for parameter values fixed across many calls.

        The parameters are checked, and the labels converted, once here. The returned function
        takes the scores and gives the same result as calling this score function.
        """
        _check_parameters((*self.deterministic_symbols,), kwargs)
        class_values = self._class_values(kwargs)
        if class_values is None:
            y_true = np.asarray(y_true).reshape(-1)

            def score(y_score: FloatNDArray) -> float:
                return self._reduce(
                    *_calculate_profits_deterministic(
                        y_true, y_score, self.calculate_profit, self.profit_function, **kwargs
                    )
                )

            return score

        labels = np.ascontiguousarray(y_true, dtype=np.int32).reshape(-1)

        def scan_score(y_score: FloatNDArray) -> float:
            return self._pick(max_profit_scan(labels, np.asarray(y_score, dtype=np.float64), *class_values))

        return scan_score

    def _maximize(self, y_true: IntNDArray, y_score: FloatNDArray, kwargs: dict[str, Any]) -> float:
        class_values = self._class_values(kwargs)
        if class_values is None:
            profits, tprs, fprs, pi0, pi1 = _calculate_profits_deterministic(
                y_true, y_score, self.calculate_profit, self.profit_function, **kwargs
            )
            return self._reduce(profits, tprs, fprs, pi0, pi1)
        result = max_profit_scan(
            np.asarray(y_true, dtype=np.int32).reshape(-1),
            np.asarray(y_score, dtype=np.float64).reshape(-1),
            *class_values,
        )
        return self._pick(result)

    def _class_values(self, kwargs: dict[str, Any]) -> tuple[float, float, float, float] | None:
        """Evaluate the class terms, or ``None`` when they are unknown or not all scalars."""
        # Instances pickled before the class terms were kept do not carry them.
        class_term_fns = getattr(self, '_class_term_fns', None)
        class_terms = getattr(self, 'class_terms', None)
        if class_term_fns is None or class_terms is None:
            return None
        values = []
        for function, term in zip(class_term_fns, class_terms, strict=True):
            value = np.asarray(_safe_run_lambda(function, term, **kwargs), dtype=np.float64)
            if value.size != 1:
                return None
            values.append(float(value.reshape(-1)[0]))
        return values[0], values[1], values[2], values[3]

    def _pick(self, result: tuple[float, float, float]) -> float:
        """Choose this score's number from the (profit, rate, threshold) of the compiled maximization."""
        raise NotImplementedError

    def _reduce(self, profits: FloatNDArray, tprs: FloatNDArray, fprs: FloatNDArray, pi0: float, pi1: float) -> float:
        raise NotImplementedError


class MaxProfitScoreDeterministic(_BaseMaxProfitDeterministic):
    """Compute the maximum profit for all deterministic variables."""

    def _pick(self, result: tuple[float, float, float]) -> float:
        return result[0]

    def _reduce(self, profits: FloatNDArray, tprs: FloatNDArray, fprs: FloatNDArray, pi0: float, pi1: float) -> float:
        return float(profits.max())


class MaxProfitRateDeterministic(_BaseMaxProfitDeterministic):
    """Compute the maximum profit for all deterministic variables."""

    def _pick(self, result: tuple[float, float, float]) -> float:
        return result[1]

    def _reduce(self, profits: FloatNDArray, tprs: FloatNDArray, fprs: FloatNDArray, pi0: float, pi1: float) -> float:
        best_index = np.argmax(profits)
        return float(tprs[best_index] * pi0 + fprs[best_index] * pi1)


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

        _, factor1, factor2 = _smooth_step_derivatives(y_score_arr - threshold, alpha)
        sigma_prime = alpha * factor1
        sigma_second = alpha**2 * factor2

        c_pos = -((self.tp_benefit + self.fn_cost) * pi0) / self.n_pos
        c_neg = ((self.tn_benefit + self.fp_cost) * pi1) / self.n_neg
        instance_weight = np.where(self.y_true == 1, c_pos, c_neg)

        gradient = instance_weight * sigma_prime
        hessian = np.abs(instance_weight * sigma_second)
        return gradient, hessian


class MaxProfitLogitGradientDeterministic(_BaseMaxProfitLogitObjective):
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
        fit_intercept: bool,
        alpha: float,
        objective_scale: float = 1.0,
        tp_benefit: float,
        tn_benefit: float,
        fp_cost: float,
        fn_cost: float,
        parameters: dict[str, FloatNDArray | float],
    ) -> None:
        super().__init__(
            features=features,
            y_true=y_true,
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            alpha=alpha,
            objective_scale=objective_scale,
        )
        self.profit_function = profit_function
        self.deterministic_symbols = deterministic_symbols
        self.calculate_profit = _safe_lambdify(profit_function)
        self.tp_benefit = tp_benefit
        self.tn_benefit = tn_benefit
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.parameters = parameters

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated max-profit objective and gradient for minimization."""
        start_coef = self._start_coef
        w = np.asarray(weights, dtype=np.float64)
        alpha = self.alpha

        y_score = self._compute_y_score(w)
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

        _, dsig_pos = _smooth_step_derivatives(s_pos - threshold, alpha, order=1)
        _, dsig_neg = _smooth_step_derivatives(s_neg - threshold, alpha, order=1)

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
        value += self._regularization_value(coef)
        gradient[start_coef:] += self._regularization_gradient(coef)

        return value, gradient

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the negated deterministic EMP objective and gradient.  Delegates to ``__call__``."""
        return self(weights)

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Return only the scalar negated EMP loss (no gradient computation).

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Negated EMP loss plus regularization.
        """
        value, _ = self(weights)
        return value

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the gradient of the negated EMP objective.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient vector matched in shape to *weights*.
        """
        _, gradient = self(weights)
        return gradient
