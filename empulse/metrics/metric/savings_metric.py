import sys
from typing import Any, ClassVar

import numpy as np
import sympy

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ..._types import FloatNDArray
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
)
from .cost_metric import (
    CostBoostGradientConst,
    CostLogitConsts,
    CostLoss,
    CostOptimalRate,
    CostOptimalThreshold,
    _build_cost_equation,
    _format_cost_function,
)
from .metric_strategies import MetricStrategy


class Savings(MetricStrategy):
    """Strategy for the Expected Savings metric."""

    def __init__(self) -> None:
        super().__init__(name='savings', direction=Direction.MAXIMIZE)

    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""
        self._score_function: MetricFn = SavingsScore(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
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
        self._score_logit_function = CostLoss(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        self._prepare_logit_objective: LogitConsts = CostLogitConsts(
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

    def score(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the metric expected savings score.

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
            The expected savings score.
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
        return _savings_score_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


class SavingsScore(SympyFnPickleMixin):
    """Class to compute the metric for binary classification."""

    _sympy_functions: ClassVar[dict[str, str]] = {'cost_function': 'cost_equation'}

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        if any(sympy.stats.rv.is_random(symbol) for symbol in self.cost_equation.free_symbols):
            raise NotImplementedError('Random variables are not supported for the savings metric.')
        self.all_zero_equation, self.all_one_equation = _build_naive_cost_functions(self.cost_equation)

        self.cost_func = _safe_lambdify(self.cost_equation)
        self.all_zero_function = _safe_lambdify(self.all_zero_equation)
        self.all_one_function = _safe_lambdify(self.all_one_equation)

    def __call__(self, y_true: FloatNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the savings score."""
        all_symbols = (
            self.cost_equation.free_symbols | self.all_zero_equation.free_symbols | self.all_one_equation.free_symbols
        )
        _check_parameters(all_symbols - {*sympy.symbols('y s')}, kwargs)
        all_zero_score = float(
            np.mean(_safe_run_lambda(self.all_zero_function, self.all_zero_equation, y=y_true, **kwargs))
        )
        all_one_score = float(
            np.mean(_safe_run_lambda(self.all_one_function, self.all_one_equation, y=y_true, **kwargs))
        )
        cost_base = min(all_zero_score, all_one_score)
        cost = _safe_run_lambda(self.cost_func, self.cost_equation, y=y_true, s=y_score, **kwargs)
        return float(1 - np.mean(cost) / cost_base)


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
