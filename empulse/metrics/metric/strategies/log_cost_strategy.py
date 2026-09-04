import copy
from typing import Any, Self

import numpy as np
import sympy
from scipy.special import expit

from ...._types import Float64Array, FloatNDArray, IntNDArray
from ..common import (
    Direction,
    MetricFn,
    RateFn,
    ThresholdFn,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda,
    _safe_run_lambda_array,
    replace_random_var_with_mean,
)
from .cost_strategy import CostOptimalRate, CostOptimalThreshold
from .metric_strategy import LogitObjective, MetricStrategy


def _soft_threshold_weights(weights: Float64Array, C: float, start_coef: int) -> Float64Array:
    """Apply soft-thresholding to the non-intercept coefficients of *weights*."""
    thresholded = weights.copy()
    coef = thresholded[start_coef:]
    absolute_val = np.abs(coef)
    thresholded[start_coef:] = np.sign(coef) * np.clip(absolute_val - C, a_min=0.0, a_max=None)
    return thresholded


def _apply_elastic_net_penalty(
    loss: float,
    gradient: Float64Array,
    weights: Float64Array,
    C: float,
    l1_ratio: float,
    start_coef: int,
) -> tuple[float, Float64Array]:
    """Add the elastic-net penalty (and its gradient) to *loss* and *gradient* in-place."""
    penalized = weights[start_coef:]
    if l1_ratio == 0.0:  # L2 regularization penalty
        gradient[start_coef:] += penalized / C
        loss += 0.5 * float(np.sum(penalized**2)) / C
    elif l1_ratio == 1.0:  # L1 regularization penalty
        gradient[start_coef:] += np.sign(penalized) / C
        loss += float(np.sum(np.abs(penalized))) / C
    else:  # elastic net regularization penalty
        gradient[start_coef:] += ((1 - l1_ratio) * penalized + l1_ratio * np.sign(penalized)) / C
        loss += float(np.sum((1 - l1_ratio) * 0.5 * penalized**2 + l1_ratio * np.abs(penalized))) / C
    return loss, gradient


class LogCostLogitObjective(LogitObjective):
    """
    Precomputed log-cost objective for logistic regression.

    Holds the constants derived from the data and exposes the :class:`~empulse.metrics.LogitObjective`
    interface. Unlike :class:`~empulse.metrics.Cost`'s objective, the per-sample loss is not linear in the
    predicted probability (it involves ``log(s)`` and ``log(1 - s)``), so the gradient depends on the
    current predictions and is recomputed (cheaply, in pure numpy) on every call rather than reduced to a
    single precomputed constant.
    """

    def __init__(
        self,
        *,
        tp_benefit: Float64Array,
        tn_benefit: Float64Array,
        fp_cost: Float64Array,
        fn_cost: Float64Array,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
    ) -> None:
        loss_const1 = y_true * -tp_benefit + (1 - y_true) * fp_cost
        loss_const2 = y_true * fn_cost - (1 - y_true) * tn_benefit

        self.loss_const1: Float64Array = np.asarray(loss_const1, dtype=np.float64).reshape(-1)
        self.loss_const2: Float64Array = np.asarray(loss_const2, dtype=np.float64).reshape(-1)
        self.features: Float64Array = np.asarray(features, dtype=np.float64)
        self.C = C
        self.l1_ratio = l1_ratio
        self.soft_threshold = soft_threshold
        self.fit_intercept = fit_intercept

    def with_indices(self, indices: FloatNDArray) -> 'LogCostLogitObjective':
        """Return a new objective restricted to the sample subset given by *indices*.

        Parameters
        ----------
        indices : array-like of int
            Row indices into the full training set.

        Returns
        -------
        LogCostLogitObjective
            A new objective for the selected samples.
        """
        obj = copy.copy(self)
        obj.loss_const1 = self.loss_const1[indices]
        obj.loss_const2 = self.loss_const2[indices]
        obj.features = self.features[indices]
        return obj

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return ``(loss, gradient)`` for *weights*."""
        start_coef = 1 if self.fit_intercept else 0
        w: Float64Array = np.asarray(weights, dtype=np.float64)
        if self.soft_threshold:
            w = _soft_threshold_weights(w, self.C, start_coef)

        n_samples = self.features.shape[0]
        logits = self.features @ w
        s: Float64Array = expit(logits)
        epsilon = np.finfo(s.dtype).eps
        s_clipped = np.clip(s, epsilon, 1 - epsilon)

        loss = float(np.mean(np.log(s_clipped) * self.loss_const1 + np.log(1 - s_clipped) * self.loss_const2))
        per_sample_grad = self.loss_const1 * (1 - s) - self.loss_const2 * s
        gradient: Float64Array = (self.features.T @ per_sample_grad) / n_samples

        # note: the intercept term (index 0, when fit_intercept=True) is excluded from `start_coef`
        # onwards and is therefore left unregularized, matching Cost's cy_logit_loss_gradient kernel.
        loss, gradient = _apply_elastic_net_penalty(loss, gradient, w, self.C, self.l1_ratio, start_coef)
        return loss, gradient

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Return only the scalar loss for *weights*."""
        return self.logit_loss_gradient(weights)[0]

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the gradient vector for *weights*."""
        return self.logit_loss_gradient(weights)[1]


class LogCostBoostGradient:
    """Class to compute the gradient and hessian of the log-cost metric for gradient boosting."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        y = sympy.symbols('y')
        self.loss_const1_eq = y * -tp_benefit + (1 - y) * fp_cost
        self.loss_const2_eq = y * fn_cost - (1 - y) * tn_benefit
        self.loss_const1_fn = _safe_lambdify(self.loss_const1_eq)
        self.loss_const2_fn = _safe_lambdify(self.loss_const2_eq)

    def __call__(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the log-cost loss wrt gradient boosting raw scores."""
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        _check_parameters(
            (self.loss_const1_eq.free_symbols | self.loss_const2_eq.free_symbols) - {sympy.symbols('y')}, parameters
        )
        loss_const1 = _safe_run_lambda_array(
            self.loss_const1_fn, self.loss_const1_eq, shape=y_true.shape[0], y=y_true, **parameters
        )
        loss_const2 = _safe_run_lambda_array(
            self.loss_const2_fn, self.loss_const2_eq, shape=y_true.shape[0], y=y_true, **parameters
        )

        y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
        s: FloatNDArray = expit(y_score)
        gradient: FloatNDArray = loss_const1 * (1 - s) - loss_const2 * s
        hessian: FloatNDArray = np.abs(loss_const1 + loss_const2) * s * (1 - s)
        return gradient, hessian


class LogCost(MetricStrategy):
    """
    Strategy for the Expected Log Cost metric.

    The expected log cost is closely related to (weighted) cross-entropy / log loss: it replaces
    the predicted probability ``s`` in :class:`~empulse.metrics.Cost`'s linear cost function with
    ``log(s)`` and ``log(1 - s)``. When ``tp_cost = tn_cost = -1`` and ``fp_cost = fn_cost = 0``,
    the expected log cost reduces to the standard log loss.

    .. seealso::
        :func:`~empulse.metrics.expected_log_cost_loss` : the underlying metric function.
    """

    _name: str = 'log cost'
    _direction: Direction = Direction.MINIMIZE

    def __init__(self) -> None:
        super().__init__(name=self._name, direction=self._direction)

    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""
        tp_benefit, tn_benefit, fp_cost, fn_cost = replace_random_var_with_mean(
            tp_benefit, tn_benefit, fp_cost, fn_cost
        )
        self._tp_benefit: sympy.Expr = tp_benefit
        self._tn_benefit: sympy.Expr = tn_benefit
        self._fp_cost: sympy.Expr = fp_cost
        self._fn_cost: sympy.Expr = fn_cost

        self._score_function: MetricFn = LogCostLoss(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self._optimal_threshold: ThresholdFn = CostOptimalThreshold(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        self._optimal_rate: RateFn = CostOptimalRate(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        self._boost_gradient = LogCostBoostGradient(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        return self

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the metric expected log cost loss.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted (calibrated) probabilities.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score: float
            The expected log cost loss.
        """
        return self._score_function(y_true, y_score, **parameters)

    def optimal_threshold(
        self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> float | FloatNDArray:
        """
        Compute the classification threshold(s) to optimize the metric value.

        The optimal threshold only depends on the (linear) cost matrix, not on the loss used to train
        the classifier, so it is identical to :class:`~empulse.metrics.Cost`'s optimal threshold.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted (calibrated) probabilities.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_threshold: float | FloatNDArray
            The optimal classification threshold(s).
        """
        return self._optimal_threshold(y_true, y_score, **parameters)

    def optimal_rate(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the predicted positive rate to optimize the metric value.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted (calibrated) probabilities.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_rate: float
            The optimal predicted positive rate.
        """
        return self._optimal_rate(y_true, y_score, **parameters)

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogCostLogitObjective:
        """
        Build an object which computes the metric value and the gradient wrt logistic coefficients.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        C : float
            Regularization strength parameter. Smaller values specify stronger regularization.
        l1_ratio : float
            The Elastic-Net mixing parameter, with range 0 <= l1_ratio <= 1.
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1 penalty.
        soft_threshold : bool
            Indicator of whether soft thresholding is applied during optimization.
        fit_intercept : bool
            Specifies if an intercept should be included in the model.
        parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        logistic_objective : LogCostLogitObjective
            An object implementing the :class:`~empulse.metrics.LogitObjective` interface.
        """
        tp_val = _safe_run_lambda(_safe_lambdify(self._tp_benefit), self._tp_benefit, **parameters)
        fn_val = _safe_run_lambda(_safe_lambdify(self._fn_cost), self._fn_cost, **parameters)
        tn_val = _safe_run_lambda(_safe_lambdify(self._tn_benefit), self._tn_benefit, **parameters)
        fp_val = _safe_run_lambda(_safe_lambdify(self._fp_cost), self._fp_cost, **parameters)
        return LogCostLogitObjective(
            tp_benefit=tp_val,
            tn_benefit=tn_val,
            fp_cost=fp_val,
            fn_cost=fn_val,
            features=features,
            y_true=y_true,
            C=C,
            l1_ratio=l1_ratio,
            soft_threshold=soft_threshold,
            fit_intercept=fit_intercept,
        )

    def gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient and hessian of the metric with respect to gradient boosting instances.

        Unlike :class:`~empulse.metrics.Cost`, the log-cost per-sample loss is not linear in the
        predicted probability, so the gradient and hessian are recomputed directly from the current
        (raw) boosting scores on every call, rather than through a precomputed constant.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The current (raw, pre-sigmoid) boosting scores.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        gradient : NDArray of shape (n_samples,)
            The gradient of the metric loss with respect to the gradient boosting weights.
        hessian : NDArray of shape (n_samples,)
            The hessian of the metric loss with respect to the gradient boosting weights.
        """
        return self._boost_gradient(y_true, y_score, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _log_cost_loss_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


class LogCostLoss:
    """Class to compute the log-cost metric for binary classification."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_log_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.cost_function = _safe_lambdify(self.cost_equation)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the log-cost loss."""
        _check_parameters(self.cost_equation.free_symbols - set(sympy.symbols('y s')), kwargs)
        y_score = np.asarray(y_score, dtype=np.float64)
        epsilon = np.finfo(y_score.dtype).eps
        y_score = np.clip(y_score, epsilon, 1 - epsilon)
        return float(np.mean(_safe_run_lambda(self.cost_function, self.cost_equation, y=y_true, s=y_score, **kwargs)))


def _build_log_cost_equation(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    """Build the per-sample log-cost equation as a sympy expression in y (label) and s (score).

    This is the single canonical form used both for numeric evaluation (via lambdify) and for LaTeX rendering.
    """
    y, s = sympy.symbols('y s')
    log_s = sympy.log(s)
    log_inv_s = sympy.log(1 - s)
    cost_function = y * (log_s * tp_cost + log_inv_s * fn_cost) + (1 - y) * (log_inv_s * tn_cost + log_s * fp_cost)
    return cost_function


def _log_cost_loss_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    i, N = sympy.symbols('i N')  # noqa: N806
    cost_function = (1 / N) * sympy.Sum(
        _build_log_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 0, N)
    )

    for symbol in cost_function.free_symbols:
        if symbol != N:
            cost_function = cost_function.subs(symbol, str(symbol) + '_i')

    output = latex(cost_function, mode='plain', order=None)

    return f'$\\displaystyle {output}$'
