from typing import Any, Self

import numpy as np
import sympy

from ...._types import FloatNDArray, IntNDArray
from ...common import classification_threshold
from ..common import (
    Direction,
    MetricFn,
    RateFn,
    ThresholdFn,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda_array,
    replace_random_var_with_mean,
)
from .auepc_strategy import _build_delta_equation
from .metric_strategy import MetricStrategy


class EmpiricalMaxProfitScore:
    """Class to compute the maximum profit found by ranking samples by predicted score."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.delta_equation = _build_delta_equation(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.delta_function = _safe_lambdify(self.delta_equation)

    def cumulative_profits(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> tuple[FloatNDArray, int]:
        """
        Compute the cumulative profit curve of targeting the top-ranked fraction of samples.

        Samples are ranked by *y_score* (descending); a leading zero represents targeting nobody.

        Returns
        -------
        cumulative_profits : NDArray of shape (n_samples + 1,)
            Cumulative profit after targeting the top ``k`` samples, for ``k = 0, ..., n_samples``.
        n_samples : int
            Number of samples.
        """
        _check_parameters(self.delta_equation.free_symbols - {sympy.symbols('y')}, kwargs)

        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
        n_samples = y_true.shape[0]

        delta: FloatNDArray = np.asarray(
            _safe_run_lambda_array(self.delta_function, self.delta_equation, shape=n_samples, y=y_true, **kwargs),
            dtype=np.float64,
        )

        sorted_indices = np.argsort(y_score)[::-1]
        cumulative_profits: FloatNDArray = np.cumsum(delta[sorted_indices])
        # Prepend a zero to represent the policy of targeting nobody.
        cumulative_profits = np.insert(cumulative_profits, 0, 0.0)
        return cumulative_profits, n_samples

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the empirical maximum profit."""
        cumulative_profits, _ = self.cumulative_profits(y_true, y_score, **kwargs)
        return float(np.max(cumulative_profits))


class EmpiricalMaxProfitOptimalRate:
    """Class to compute the optimal predicted positive rate found by ranking samples."""

    def __init__(self, score_function: EmpiricalMaxProfitScore):
        self.score_function = score_function

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the fraction of samples that should be targeted to maximize profit."""
        cumulative_profits, n_samples = self.score_function.cumulative_profits(y_true, y_score, **kwargs)
        optimal_index = int(np.argmax(cumulative_profits))
        return optimal_index / n_samples


class EmpiricalMaxProfitOptimalThreshold:
    """Class to compute the optimal classification threshold found by ranking samples."""

    def __init__(self, optimal_rate: EmpiricalMaxProfitOptimalRate):
        self.optimal_rate = optimal_rate

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the score threshold at which a sample should be targeted to maximize profit."""
        rate = self.optimal_rate(y_true, y_score, **kwargs)
        return classification_threshold(y_true, y_score, rate)


class EmpiricalMaxProfit(MetricStrategy):
    """
    Strategy for the (empirical) maximum profit found by ranking samples by predicted score.

    Unlike :class:`~empulse.metrics.MaxProfit`, which searches for the profit-maximizing
    threshold *inside* the integral over any stochastic variables (via a closed-form profit
    function of the population's true/false positive rates), ``EmpiricalMaxProfit`` first
    simplifies any stochastic variable to its mean, and only then searches for the
    profit-maximizing threshold, empirically: samples are ranked by predicted score, the
    cumulative profit of targeting (predicting positive for) the top-ranked fraction is tracked
    per sample (rather than through population-level true/false positive rates), and the maximum
    of that curve is the metric's score.

    Because the threshold search happens per-sample rather than through population aggregates,
    this strategy naturally supports instance-dependent (array-like) costs and benefits, unlike
    :class:`~empulse.metrics.MaxProfit`.

    ``EmpiricalMaxProfit`` does not support use as a model training objective (no
    ``logit_objective`` or ``gradient_boost_objective``): the profit-maximizing threshold is found
    via an empirical argmax over the ranked samples, which is piecewise-constant (and therefore not
    differentiable) in the predicted scores.

    .. seealso::
        :func:`~empulse.metrics.empb_score` : the underlying metric function.
    """

    _name: str = 'empirical max profit'
    _direction: Direction = Direction.MAXIMIZE

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

        self._score_function: MetricFn = EmpiricalMaxProfitScore(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self._optimal_rate: RateFn = EmpiricalMaxProfitOptimalRate(self._score_function)  # type: ignore[arg-type]
        self._optimal_threshold: ThresholdFn = EmpiricalMaxProfitOptimalThreshold(self._optimal_rate)  # type: ignore[arg-type]
        return self

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the empirical maximum profit.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters;
            the stochastic variable is replaced by its mean before computing the metric.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score: float
            The empirical maximum profit.
        """
        return self._score_function(y_true, y_score, **parameters)

    def optimal_rate(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the fraction of samples that should be targeted (predicted positive) to maximize profit.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_rate: float
            The optimal predicted positive rate.
        """
        return self._optimal_rate(y_true, y_score, **parameters)

    def optimal_threshold(
        self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> float | FloatNDArray:
        """
        Compute the score threshold above which a sample should be targeted to maximize profit.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_threshold: float
            The optimal classification threshold.
        """
        return self._optimal_threshold(y_true, y_score, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _empirical_max_profit_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


def _empirical_max_profit_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    delta_equation = _build_delta_equation(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    for symbol in delta_equation.free_symbols:
        delta_equation = delta_equation.subs(symbol, str(symbol) + '_i')
    delta_latex = latex(delta_equation, mode='plain', order=None)

    formula = (
        r'\max_{k \in \{0, ..., N\}} \sum_{i=1}^{k} \Delta_{\pi(i)}' + r'\quad\text{where }\Delta_i = ' + delta_latex
    )

    return f'$\\displaystyle {formula}$'
