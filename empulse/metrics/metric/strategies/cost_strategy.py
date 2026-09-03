import copy
from collections.abc import Generator
from typing import Any, Self

import numpy as np
import sympy

from ...._types import Float64Array, FloatNDArray, IntNDArray
from ..._loss import cy_logit_gradient, cy_logit_loss, cy_logit_loss_gradient
from ..common import (
    BoostGradientConst,
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
from .metric_strategy import MetricStrategy


class CostLogitObjective:
    """
    Precomputed cost-metric objective for logistic regression.

    Holds the constants derived from the data and exposes the same API as
    ``MaxProfitLogitGradientPiecewise``:

    * ``__call__(weights)`` – returns ``(value, gradient)``; delegates to :meth:`logit_loss_gradient`.
    * ``logit_loss_gradient(weights)`` – returns ``(value, gradient)``
    * ``logit_loss(weights)`` – returns only the scalar loss
    * ``logit_gradient(weights)`` – returns only the gradient vector
    * ``logit_gradient_steps(initial_weights)`` – generator that yields gradients
      while reusing the (fixed) precomputed constants across iterations

    The constants are computed once during construction and do not change, so
    the ``refresh`` signal in ``logit_gradient_steps`` is accepted for API
    compatibility but has no effect.
    """

    def __init__(
        self,
        *,
        tp_benefit: FloatNDArray,
        tn_benefit: FloatNDArray,
        fp_cost: FloatNDArray,
        fn_cost: FloatNDArray,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
    ) -> None:
        grad_const = features * (y_true * (-tp_benefit - fn_cost) + (1 - y_true) * (fp_cost + tn_benefit))
        loss_const1 = y_true * -tp_benefit + (1 - y_true) * fp_cost
        loss_const2 = y_true * fn_cost - (1 - y_true) * tn_benefit

        # Cast to float64 so Cython's double[:] memoryview accepts the arrays without a copy.
        self.grad_const: Float64Array = np.asarray(grad_const, dtype=np.float64)
        self.loss_const1: Float64Array = np.asarray(loss_const1, dtype=np.float64).reshape(-1)
        self.loss_const2: Float64Array = np.asarray(loss_const2, dtype=np.float64).reshape(-1)
        self.features: Float64Array = np.asarray(features, dtype=np.float64)
        self.C = C
        self.l1_ratio = l1_ratio
        self.soft_threshold = soft_threshold
        self.fit_intercept = fit_intercept

    def _common_kwargs(self) -> dict[str, Any]:
        return {
            'features': self.features,
            'C': self.C,
            'l1_ratio': self.l1_ratio,
            'soft_threshold': self.soft_threshold,
            'fit_intercept': self.fit_intercept,
        }

    def with_indices(self, indices: FloatNDArray) -> 'CostLogitObjective':
        """Return a new objective restricted to the sample subset given by *indices*.

        The pre-computed constant arrays (``grad_const``, ``loss_const1``,
        ``loss_const2``, ``features``) are sliced; all scalar attributes are
        shared.  This is very cheap compared to rebuilding from scratch.

        Parameters
        ----------
        indices : array-like of int
            Row indices into the full training set.

        Returns
        -------
        CostLogitObjective
            A new objective for the selected samples.
        """
        obj = copy.copy(self)
        obj.grad_const = self.grad_const[indices]
        obj.loss_const1 = self.loss_const1[indices]
        obj.loss_const2 = self.loss_const2[indices]
        obj.features = self.features[indices]
        return obj

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return ``(loss, gradient)`` for *weights*.  Delegates to :meth:`logit_loss_gradient`."""
        return self.logit_loss_gradient(weights)

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return ``(loss, gradient)`` for *weights*."""
        w: Float64Array = np.asarray(weights, dtype=np.float64)
        return cy_logit_loss_gradient(
            w,
            grad_const=self.grad_const,
            loss_const1=self.loss_const1,
            loss_const2=self.loss_const2,
            **self._common_kwargs(),
        )

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Return only the scalar loss for *weights*.

        No gradient is computed, so this is cheaper when only the objective
        value is needed (e.g. final fitness evaluation in a memetic algorithm).

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Loss value.
        """
        return float(
            cy_logit_loss(
                np.asarray(weights, dtype=np.float64),
                loss_const1=self.loss_const1,
                loss_const2=self.loss_const2,
                **self._common_kwargs(),
            )
        )

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the gradient vector for *weights*.

        No loss value is accumulated, so this is cheaper when only the
        gradient is needed (e.g. gradient-descent inner steps).

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient vector matched in shape to *weights*.
        """
        return cy_logit_gradient(  # type: ignore[return-value]
            np.asarray(weights, dtype=np.float64),
            grad_const=self.grad_const,
            **self._common_kwargs(),
        )

    def _logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """Yield gradients for successive weight vectors.

        Because the constants are derived from fixed data and parameters,
        there is no expensive state to reconstruct between steps.  The
        generator accepts the same send-protocol as
        ``MaxProfitLogitGradientPiecewise.logit_gradient_steps`` for API
        compatibility: passing ``(weights, refresh)`` works but ``refresh``
        is silently ignored.

        Parameters
        ----------
        initial_weights : ndarray
            Starting coefficient vector.

        Yields
        ------
        gradient : ndarray
            Gradient at the current weights.

        Receives (via ``send``)
        -----------------------
        weights : ndarray
            New coefficient vector for the next gradient step.
        (weights, refresh) : (ndarray, bool)
            ``refresh`` is accepted but ignored.

        """
        weights: FloatNDArray

        sent = yield

        while True:
            if sent is None:
                return
            if isinstance(sent, tuple):
                weights, _ = sent  # refresh ignored – constants are fixed
            else:
                weights = sent

            grad: Float64Array = cy_logit_gradient(  # type: ignore[assignment]
                np.asarray(weights, dtype=np.float64),
                grad_const=self.grad_const,
                **self._common_kwargs(),
            )
            sent = yield grad

    def logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients for successive weight vectors.

        Because the constants are derived from fixed data and parameters,
        there is no expensive state to reconstruct between steps.  The
        generator accepts the same send-protocol as ``MaxProfitLogitGradientPiecewise.logit_gradient_steps`` for API
        compatibility: passing ``(weights, refresh)`` works but ``refresh`` is silently ignored.

        Parameters
        ----------
        initial_weights : ndarray
            Starting coefficient vector.

        Yields
        ------
        gradient : ndarray
            Gradient at the current weights.

        Receives (via ``send``)
        -----------------------
        weights : ndarray
            New coefficient vector for the next gradient step.
        (weights, refresh) : (ndarray, bool)
            ``refresh`` is accepted but ignored.

        Examples
        --------
        >>> gen = objective.logit_gradient_steps()
        >>> grad = gen.send(theta)
        >>> gen.close()
        """
        generator = self._logit_gradient_steps()
        next(generator)
        return generator


class Cost(MetricStrategy):
    """Strategy for the Expected Cost metric."""

    _name: str = 'cost'
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
        self._prepare_boost_objective: BoostGradientConst = CostBoostGradientConst(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        return self

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
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
        self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
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

    def optimal_rate(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
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

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> CostLogitObjective:
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

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        logistic_objective : Callable[[NDArray], tuple[float, NDArray]]
            A function that takes logistic regression weights as input and returns the metric value and its gradient.
            The function signature is:
            ``logistic_objective(weights) -> (value, gradient)``
        """
        tp_val = _safe_run_lambda(_safe_lambdify(self._tp_benefit), self._tp_benefit, **parameters)
        fn_val = _safe_run_lambda(_safe_lambdify(self._fn_cost), self._fn_cost, **parameters)
        tn_val = _safe_run_lambda(_safe_lambdify(self._tn_benefit), self._tn_benefit, **parameters)
        fp_val = _safe_run_lambda(_safe_lambdify(self._fp_cost), self._fp_cost, **parameters)
        return CostLogitObjective(
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


class CostLoss:
    """Class to compute the metric for binary classification."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.cost_function = _safe_lambdify(self.cost_equation)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters(self.cost_equation.free_symbols - set(sympy.symbols('y s')), kwargs)
        return float(np.mean(_safe_run_lambda(self.cost_function, self.cost_equation, y=y_true, s=y_score, **kwargs)))


def _build_cost_equation(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    """Build the per-sample cost equation as a sympy expression in y (label) and s (score).

    This is the single canonical form used both for numeric evaluation (via lambdify) and for LaTeX rendering.
    """
    y, s = sympy.symbols('y s')
    cost_function = y * (s * tp_cost + (1 - s) * fn_cost) + (1 - y) * ((1 - s) * tn_cost + s * fp_cost)
    return cost_function


class CostBoostGradientConst:
    """Class to compute the gradient constants of the cost metric for gradient boosting."""

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


class CostOptimalThreshold:
    """Class to compute the optimal threshold for the cost metric."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
        self.numerator_expression = fp_cost + tn_benefit
        self.calculate_denominator = _safe_lambdify(self.denominator_expression)
        self.calculate_numerator = _safe_lambdify(self.numerator_expression)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: Any) -> FloatNDArray | float:
        """Compute the optimal threshold(s). `y_true` and `y_score` are unused and kept for API compatibility."""
        _check_parameters(self.denominator_expression.free_symbols, parameters)
        denominator = _safe_run_lambda(self.calculate_denominator, self.denominator_expression, **parameters)
        numerator = _safe_run_lambda(self.calculate_numerator, self.numerator_expression, **parameters)

        eps = float(np.finfo(np.float64).eps)
        if np.isscalar(denominator):
            if denominator == 0:
                denominator = eps
        else:
            denom_arr = np.asarray(denominator, dtype=np.float64)
            denominator = np.where(denom_arr == 0, eps, denom_arr)

        optimal: FloatNDArray | float = numerator / denominator  # type: ignore[operator, assignment]
        return float(optimal) if np.isscalar(optimal) else optimal  # type: ignore[arg-type]


class CostOptimalRate:
    """Class to compute the optimal predicted positive rate for the cost metric."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
        self.numerator_expression = fp_cost + tn_benefit
        self.calculate_denominator = _safe_lambdify(self.denominator_expression)
        self.calculate_numerator = _safe_lambdify(self.numerator_expression)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: Any) -> float:
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

        t_star: FloatNDArray | float = numerator / denom_safe  # type: ignore[operator, assignment]

        scores = np.asarray(y_score)
        if scores.ndim > 1:
            scores = scores.reshape(-1)

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
        _build_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 0, N)
    )

    for symbol in cost_function.free_symbols:
        if symbol != N:
            cost_function = cost_function.subs(symbol, str(symbol) + '_i')

    output = latex(cost_function, mode='plain', order=None)

    return f'$\\displaystyle {output}$'
