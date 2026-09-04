from typing import Any, Self

import numpy as np
import sympy

from ...._types import FloatNDArray, IntNDArray
from ..common import (
    Direction,
    MetricFn,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda_array,
    replace_random_var_with_mean,
)
from .metric_strategy import MetricStrategy


def _build_delta_equation(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    """Build the per-sample "targeting" profit differential as a sympy expression in y (label).

    This is the marginal profit gained by predicting a sample positive (targeting it) instead of
    negative: ``y * (tp_benefit + fn_cost) - (1 - y) * (fp_cost + tn_benefit)``. Ranking samples by
    this quantity, in descending order, is the policy that maximizes cumulative profit at *every*
    targeted fraction simultaneously -- i.e. it is the oracle ("perfect model") ranking that AUEPC
    compares the classifier's own ranking against.

    This is the single canonical form used both for numeric evaluation (via lambdify) and for LaTeX
    rendering.
    """
    y = sympy.symbols('y')
    return y * (tp_benefit + fn_cost) - (1 - y) * (fp_cost + tn_benefit)


class AUEPCScore:
    """Class to compute the Area Under the Expected Profit Curve (AUEPC) for binary classification."""

    def __init__(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
        normalize: bool = True,
    ) -> None:
        self.delta_equation = _build_delta_equation(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.delta_function = _safe_lambdify(self.delta_equation)
        self.normalize = normalize

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the AUEPC score."""
        _check_parameters(self.delta_equation.free_symbols - {sympy.symbols('y')}, kwargs)

        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
        n_samples = y_true.shape[0]

        delta: FloatNDArray = np.asarray(
            _safe_run_lambda_array(self.delta_function, self.delta_equation, shape=n_samples, y=y_true, **kwargs),
            dtype=np.float64,
        )

        # Oracle ranking: sorting by the profit differential maximizes cumulative profit at every
        # targeted fraction simultaneously, so this is the "perfect model" curve.
        perfect_order = np.argsort(delta)[::-1]
        perfect_profits = np.cumsum(delta[perfect_order])

        # The classifier's own ranking.
        model_order = np.argsort(y_score)[::-1]
        profits = np.cumsum(delta[model_order])

        # Stop at the point where the oracle's cumulative profit becomes negative: beyond this
        # point, even the best possible policy is losing money by targeting further samples.
        stop_index: int = (
            int(np.argmax(perfect_profits < 0)) if np.any(perfect_profits < 0) else n_samples  # type: ignore[assignment]
        )

        score = float(np.trapezoid(profits[:stop_index] / perfect_profits[:stop_index], dx=1 / n_samples))  # type: ignore[attr-defined]
        if self.normalize:
            score /= (stop_index - 1) / n_samples
        return score


class AUEPC(MetricStrategy):
    """
    Strategy for the Area Under the Expected Profit Curve (AUEPC) metric.

    AUEPC ranks samples by their predicted score and tracks the cumulative profit of targeting
    (predicting positive for) the top-ranked fraction of samples, using the same benefits/costs as
    the other strategies (:class:`~empulse.metrics.Cost`, :class:`~empulse.metrics.MaxProfit`, ...).
    This cumulative-profit curve is then compared against the curve of an oracle ranking that
    maximizes cumulative profit at every targeted fraction -- the ratio of the two, integrated over
    the targeted fraction, is the AUEPC score. A perfect model (whose ranking matches the oracle)
    scores 1.0 (when ``normalize=True``); an unhelpful/random ranking scores lower.

    Unlike :class:`~empulse.metrics.Cost` and :class:`~empulse.metrics.MaxProfit`, AUEPC does not
    support use as a model training objective (no ``logit_objective`` or
    ``gradient_boost_objective``): it evaluates a full ranking, not the outcome of a single sample.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to normalize the AUEPC score so that a perfect model scores 1.0.
        This is only useful when part of the expected profit curve is negative.

    .. seealso::
        :func:`~empulse.metrics.auepc_score` : the underlying metric function.
    """

    _name: str = 'auepc'
    _direction: Direction = Direction.MAXIMIZE

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(name=self._name, direction=self._direction)
        self.normalize = normalize

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

        self._score_function: MetricFn = AUEPCScore(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost, normalize=self.normalize
        )
        return self

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the AUEPC score.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        parameters: float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score: float
            The AUEPC score.
        """
        return self._score_function(y_true, y_score, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _auepc_score_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


def _auepc_score_to_latex(
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
        r'\frac{1}{N}\sum_{k=1}^{N}'
        r'\frac{\sum_{i=1}^{k}\Delta_{\pi(i)}}{\sum_{i=1}^{k}\Delta_{\pi^{*}(i)}}'
        r'\quad\text{where }\Delta_i = ' + delta_latex
    )

    return f'$\\displaystyle {formula}$'
