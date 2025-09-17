import sys
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import numpy as np
import sympy

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ..._types import FloatNDArray
from .._loss import cy_logit_loss_gradient
from .common import (
    BoostGradientConst,
    Direction,
    LogitConsts,
    MetricFn,
    RateFn,
    SympyFnPickleMixin,
    ThresholdFn,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda,
    _safe_run_lambda_array,
)
from .metric_strategies import MetricStrategy


class Cost(MetricStrategy):
    """Strategy for the Expected Cost metric."""

    def __init__(self) -> None:
        super().__init__(name='cost', direction=Direction.MINIMIZE)

    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""
        all_symbols = tp_benefit.free_symbols | tn_benefit.free_symbols | fp_cost.free_symbols | fn_cost.free_symbols
        if any(sympy.stats.rv.is_random(symbol) for symbol in all_symbols):
            raise NotImplementedError('Random variables are not supported for the cost metric.')

        self._score_function: MetricFn = CostLoss(
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
        self._prepare_logit_objective: LogitConsts = CostLogitConsts(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self._prepare_boost_objective: BoostGradientConst = CostBoostGradientConst(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        return self

    def score(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the metric expected cost loss.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score: float
            The expected cost loss.
        """
        return self._score_function(y_true, y_score, **parameters)

    def optimal_threshold(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> float | FloatNDArray:
        """
        Compute the classification threshold(s) to optimize the metric value.

        i.e., the score threshold at which an observation should be classified as positive to optimize the metric.
        For instance-dependent costs and benefits, this will return an array of thresholds, one for each sample.
        For class-dependent costs and benefits, this will return a single threshold value.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

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

    def optimal_rate(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the predicted positive rate to optimize the metric value.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

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

    def prepare_logit_objective(
        self, features: FloatNDArray, y_true: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
        """
        Compute the constant term of the loss and gradient of the metric wrt logistic regression coefficients.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        gradient_const : NDArray of shape (n_samples, n_features)
            The constant term of the gradient.
        loss_const1 : NDArray of shape (n_features,)
            The first constant term of the loss function.
        loss_const2 : NDArray of shape (n_features,)
            The second constant term of the loss function.
        """
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=1)

        return self._prepare_logit_objective.prepare(features, y_true, **parameters)

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> Callable[[FloatNDArray], tuple[float, FloatNDArray]]:
        """
        Build a function which computes the metric value and the gradient of the metric w.r.t logistic coefficients.

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
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        logistic_objective : Callable[[NDArray], tuple[float, NDArray]]
            A function that takes logistic regression weights as input and returns the metric value and its gradient.
            The function signature is:
            ``logistic_objective(weights) -> (value, gradient)``
        """
        grad_const, loss_const1, loss_const2 = self.prepare_logit_objective(features, y_true, **parameters)
        loss_const1 = (
            loss_const1.reshape(-1)
            if isinstance(loss_const1, np.ndarray)
            else np.full(len(y_true), loss_const1, dtype=np.float64)
        )
        loss_const2 = (
            loss_const2.reshape(-1)
            if isinstance(loss_const2, np.ndarray)
            else np.full(len(y_true), loss_const2, dtype=np.float64)
        )
        return partial(
            cy_logit_loss_gradient,
            grad_const=grad_const,
            loss_const1=loss_const1,
            loss_const2=loss_const2,
            features=features,
            C=C,
            l1_ratio=l1_ratio,
            soft_threshold=soft_threshold,
            fit_intercept=fit_intercept,
        )

    def prepare_boost_objective(self, y_true: FloatNDArray, **parameters: FloatNDArray | float) -> FloatNDArray:
        """
        Compute the gradient's constant term of the metric wrt gradient boost.

        Parameters
        ----------
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        gradient_const : NDArray of shape (n_samples, n_features)
            The constant term of the gradient.
        """
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=1)
        return self._prepare_boost_objective(y_true, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _cost_loss_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


class CostLoss(SympyFnPickleMixin):
    """Class to compute the metric for binary classification."""

    _sympy_functions: ClassVar[dict[str, str]] = {'cost_function': 'cost_equation'}

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.cost_function = _safe_lambdify(self.cost_equation)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters(self.cost_equation.free_symbols - set(sympy.symbols('y s')), kwargs)
        return float(np.mean(_safe_run_lambda(self.cost_function, self.cost_equation, y=y_true, s=y_score, **kwargs)))


def _build_cost_equation(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    y, s = sympy.symbols('y s')
    cost_function = y * (s * tp_cost + (1 - s) * fn_cost) + (1 - y) * ((1 - s) * tn_cost + s * fp_cost)
    return cost_function


class CostLogitConsts(SympyFnPickleMixin):
    """Class to compute the constants of the cost metric for logistic regression."""

    _sympy_functions: ClassVar[dict[str, str]] = {
        'gradient_fn': 'gradient_const',
        'loss_const1_fn': 'loss_const1',
        'loss_const2_fn': 'loss_const2',
    }

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        y, x = sympy.symbols('y x')
        self.gradient_const = x * (y * (-tp_benefit - fn_cost) + (1 - y) * (fp_cost + tn_benefit))
        self.gradient_fn = _safe_lambdify(self.gradient_const)
        self.loss_const1 = y * -tp_benefit + (1 - y) * fp_cost
        self.loss_const1_fn = _safe_lambdify(self.loss_const1)
        self.loss_const2 = y * fn_cost - (1 - y) * tn_benefit
        self.loss_const2_fn = _safe_lambdify(self.loss_const2)

    def prepare(
        self, x: FloatNDArray, y_true: FloatNDArray, **kwargs: Any
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
        """Prepare the constant terms for the logistic regression objective."""
        gradient_const_value = _safe_run_lambda_array(
            self.gradient_fn, self.gradient_const, shape=x.shape, y=y_true, x=x, **kwargs
        )
        loss_const1_value = _safe_run_lambda_array(
            self.loss_const1_fn, self.loss_const1, shape=y_true.shape[0], y=y_true, **kwargs
        )
        loss_const2_value = _safe_run_lambda_array(
            self.loss_const2_fn, self.loss_const2, shape=y_true.shape[0], y=y_true, **kwargs
        )

        return gradient_const_value, loss_const1_value, loss_const2_value


class CostBoostGradientConst(SympyFnPickleMixin):
    """Class to compute the gradient constants of the cost metric for gradient boosting."""

    _sympy_functions: ClassVar[dict[str, str]] = {'gradient_const_fn': 'gradient_const_eq'}

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        y = sympy.symbols('y')
        self.gradient_const_eq = y * (-tp_benefit - fn_cost) + (1 - y) * (fp_cost + tn_benefit)
        self.gradient_const_fn = _safe_lambdify(self.gradient_const_eq)

    def __call__(self, y_true: FloatNDArray, **kwargs: Any) -> FloatNDArray:
        """Compute the gradient constants."""
        _check_parameters(self.gradient_const_eq.free_symbols - {sympy.symbols('y')}, kwargs)
        gradient_const_value = _safe_run_lambda_array(
            self.gradient_const_fn, self.gradient_const_eq, shape=y_true.shape[0], y=y_true, **kwargs
        )
        return gradient_const_value


class CostOptimalThreshold(SympyFnPickleMixin):
    """Class to compute the optimal threshold for the cost metric."""

    _sympy_functions: ClassVar[dict[str, str]] = {
        'calculate_denominator': 'denominator_expression',
        'calculate_numerator': 'numerator_expression',
    }

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
        self.numerator_expression = fp_cost + tn_benefit
        self.calculate_denominator = _safe_lambdify(self.denominator_expression)
        self.calculate_numerator = _safe_lambdify(self.numerator_expression)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any) -> FloatNDArray | float:
        """Compute the optimal threshold(s). `y_true` and `y_score` are unused and kept for API compatibility."""
        _check_parameters(self.denominator_expression.free_symbols, parameters)
        denominator = _safe_run_lambda(self.calculate_denominator, self.denominator_expression, **parameters)
        numerator = _safe_run_lambda(self.calculate_numerator, self.numerator_expression, **parameters)

        eps = float(np.finfo(np.float64).eps)
        if np.isscalar(denominator):
            if denominator == 0:
                denominator = eps
        else:
            denominator = np.clip(denominator, float(np.finfo(float).eps), denominator)

        optimal = numerator / denominator
        return float(optimal) if np.isscalar(optimal) else optimal  # type: ignore[arg-type]


class CostOptimalRate(SympyFnPickleMixin):
    """Class to compute the optimal predicted positive rate for the cost metric."""

    _sympy_functions: ClassVar[dict[str, str]] = {
        'calculate_denominator': 'denominator_expression',
        'calculate_numerator': 'numerator_expression',
    }

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
        self.numerator_expression = fp_cost + tn_benefit
        self.calculate_denominator = _safe_lambdify(self.denominator_expression)
        self.calculate_numerator = _safe_lambdify(self.numerator_expression)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any) -> float:
        """Compute the optimal predicted positive rate."""
        _check_parameters(self.denominator_expression.free_symbols, parameters)
        denominator = _safe_run_lambda(self.calculate_denominator, self.denominator_expression, **parameters)
        numerator = _safe_run_lambda(self.calculate_numerator, self.numerator_expression, **parameters)

        # Robust division to avoid divide-by-zero
        eps = float(np.finfo(np.float64).eps)
        if np.isscalar(denominator):
            denom_safe: FloatNDArray | float = denominator if denominator != 0 else eps  # type: ignore[assignment]
        else:
            denom_arr = np.asarray(denominator, dtype=np.float64)
            denom_safe = np.where(denom_arr == 0, eps, denom_arr)  # type: ignore[assignment]

        t_star = numerator / denom_safe

        # Normalize y\_score shape to 1D
        scores = np.asarray(y_score)
        if scores.ndim > 1:
            scores = scores.reshape(-1)

        # If t\* is scalar, compare against scalar; otherwise compare elementwise
        if np.isscalar(t_star):
            rate = float(np.mean(scores >= float(t_star)))  # type: ignore[arg-type]
        else:
            t_arr = np.asarray(t_star, dtype=np.float64).reshape(-1)
            rate = float(np.mean(scores >= t_arr))

        return rate


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
