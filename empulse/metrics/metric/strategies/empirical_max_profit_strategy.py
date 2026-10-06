from collections.abc import Callable
from typing import Any, ClassVar, Literal, Self

import numpy as np
import sympy

from ...._common._objective import ElasticNetPenalty, RankingLogitValueObjective
from ...._types import FloatNDArray, IntNDArray
from ...common import classification_threshold
from .._compile import MetricFn, RateFn, ThresholdFn, _safe_lambdify, _safe_run_lambda_array
from .._direction import Direction
from .._parameter_domain import _check_parameters
from .._stochastic import replace_random_var_with_mean
from .._symbolic import _latex
from ..capabilities import Capability
from .auepc_strategy import _build_delta_equation, _ranked_profit_curve
from .metric_strategy import MetricStrategy


class EmpiricalMaxProfitScore:
    """Class to compute the maximum profit per sample found by ranking samples by predicted score."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.delta_equation = _build_delta_equation(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.delta_function = _safe_lambdify(self.delta_equation)

    def cumulative_profits(
        self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any
    ) -> tuple[IntNDArray, FloatNDArray, int]:
        """
        Compute the cumulative profit curve of targeting the top-ranked fraction of samples.

        Samples are ranked by *y_score* (descending). Samples with equal scores cannot be separated
        by a threshold, so the curve only has points at the edges of each group of tied scores.

        Returns
        -------
        n_targeted : NDArray of shape (n_points,)
            Number of samples targeted at each point, from 0 (nobody) to ``n_samples``.
        cumulative_profits : NDArray of shape (n_points,)
            Cumulative profit at each point.
        n_samples : int
            Number of samples.
        """
        y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
        delta = self.profit_differential(y_true, **kwargs)
        n_targeted, cumulative_profits = _ranked_profit_curve(delta, y_score)
        return n_targeted, cumulative_profits, delta.size

    def profit_differential(self, y_true: IntNDArray, **kwargs: Any) -> FloatNDArray:
        """Return how much targeting each sample earns over not targeting it."""
        _check_parameters(self.delta_equation.free_symbols - {sympy.symbols('y')}, kwargs)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        delta: FloatNDArray = np.asarray(
            _safe_run_lambda_array(self.delta_function, self.delta_equation, shape=y_true.size, y=y_true, **kwargs),
            dtype=np.float64,
        )
        return delta

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the empirical maximum profit per sample."""
        _, cumulative_profits, n_samples = self.cumulative_profits(y_true, y_score, **kwargs)
        return float(np.max(cumulative_profits)) / n_samples

    def _sample_scorer(self, y_true: IntNDArray, **kwargs: Any) -> Callable[[FloatNDArray], float]:
        """
        Prepare the score of fixed labels, for parameter values fixed across many calls.

        The profit differential is computed once here. The returned function takes the scores and
        gives the same result as calling this score function.
        """
        delta = self.profit_differential(y_true, **kwargs)

        def score(y_score: FloatNDArray) -> float:
            _, cumulative_profits = _ranked_profit_curve(delta, np.asarray(y_score, dtype=np.float64).reshape(-1))
            return float(np.max(cumulative_profits)) / delta.size

        return score


class EmpiricalMaxProfitOptimalRate:
    """Class to compute the optimal predicted positive rate found by ranking samples."""

    def __init__(self, score_function: EmpiricalMaxProfitScore):
        self.score_function = score_function

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the fraction of samples that should be targeted to maximize profit."""
        n_targeted, cumulative_profits, n_samples = self.score_function.cumulative_profits(y_true, y_score, **kwargs)
        return int(n_targeted[np.argmax(cumulative_profits)]) / n_samples


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
    of that curve, divided by the number of samples, is the metric's score. Like
    :class:`~empulse.metrics.MaxProfit`, it is a profit per sample, relative to targeting nobody.

    Because the threshold search happens per-sample rather than through population aggregates,
    this strategy naturally supports instance-dependent (array-like) costs and benefits, unlike
    :class:`~empulse.metrics.MaxProfit`.

    The profit-maximizing threshold is found via an empirical argmax over the ranked samples, which
    is piecewise-constant (and therefore not differentiable) in the predicted scores. Models that
    train without gradients can optimize it, such as :class:`~empulse.models.ProfTreeClassifier`,
    :class:`~empulse.models.ProfSRClassifier` and :class:`~empulse.models.ProfLogitClassifier` with
    an optimizer that does not use gradients, but gradient-based training (no ``logit_objective``
    or ``gradient_boost_objective``) is not supported.

    .. seealso::
        :func:`~empulse.metrics.empb_score` : the underlying metric function.
    """

    _capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.RANKING})
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
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters;
            the stochastic variable is replaced by its mean before computing the metric.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score : float
            The empirical maximum profit per sample.
        """
        return self._score_function(y_true, y_score, **parameters)

    def logit_value_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> RankingLogitValueObjective:
        """
        Build the logit objective for an optimizer that needs only its value, not its gradient.

        The objective is the negated empirical maximum profit of the model, plus an elastic-net
        penalty measured against the average gain of deciding a sample rightly instead of wrongly.

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
        fit_intercept : bool
            Specifies if an intercept should be included in the model.
        **parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

        Returns
        -------
        logistic_objective : RankingLogitValueObjective
            The objective, whose ``logit_loss`` is the regularized negated score.
        """
        y_true = np.asarray(y_true).reshape(-1)
        parameters = {
            name: value.reshape(-1) if isinstance(value, np.ndarray) else value for name, value in parameters.items()
        }
        delta = self._score_function.profit_differential(y_true, **parameters)  # type: ignore[attr-defined]
        return RankingLogitValueObjective(
            score=self._score_function._sample_scorer(y_true, **parameters),  # type: ignore[attr-defined]
            features=features,
            penalty=ElasticNetPenalty.from_scale(
                objective_scale=float(np.mean(np.abs(delta))),
                C=C,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                n_samples=features.shape[0],
            ),
        )

    def optimal_rate(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the fraction of samples that should be targeted (predicted positive) to maximize profit.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_rate : float
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
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        optimal_threshold : float
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


class EmpiricalMinCost(EmpiricalMaxProfit):
    """
    Strategy for the Empirical Minimum Cost metric.

    The cost phrasing of :class:`EmpiricalMaxProfit`: it walks the same ranking and reports the
    value at the same optimal cutoff negated, as a cost to minimize rather than a profit to
    maximize. Which of the two you use is a presentation choice -- models train identically on
    either, because they optimize :meth:`~empulse.metrics.BaseMetric._loss`, which removes the sign
    difference.

    .. seealso::
        :class:`EmpiricalMaxProfit` : The profit phrasing of the same metric.
    """

    _name: str = 'empirical min cost'
    _direction: Direction = Direction.MINIMIZE

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the empirical minimum cost score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score : float
            The empirical minimum cost per sample.
        """
        return -super().score(y_true, y_score, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        # Negating all four inputs negates the per-sample delta; minimizing the negated cumulative
        # sum is the same cutoff that maximizes the original one.
        return _empirical_max_profit_to_latex(-tp_benefit, -tn_benefit, -fp_cost, -fn_cost, operator='min')


def _empirical_max_profit_to_latex(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    operator: Literal['max', 'min'] = 'max',
) -> str:
    delta_equation = _build_delta_equation(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    for symbol in delta_equation.free_symbols:
        delta_equation = delta_equation.subs(symbol, str(symbol) + '_i')
    delta_latex = _latex(delta_equation)

    formula = (
        rf'\frac{{1}}{{N}} \{operator}_{{k \in \{{0, ..., N\}}}} \sum_{{i=1}}^{{k}} \Delta_{{\pi(i)}}'
        r'\quad\text{where }\Delta_i = ' + delta_latex
    )

    return f'$\\displaystyle {formula}$'
