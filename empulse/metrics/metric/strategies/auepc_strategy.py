from collections.abc import Callable
from typing import Any, ClassVar, Self

import numpy as np
import sympy
from scipy.integrate import trapezoid

from ...._common._objective import ElasticNetPenalty, RankingLogitValueObjective
from ...._types import Float64Array, FloatNDArray, IntNDArray
from .._compile import MetricFn, _safe_lambdify, _safe_run_lambda_array
from .._direction import Direction
from .._parameter_domain import _check_parameters
from .._stochastic import replace_random_var_with_mean
from .._symbolic import _latex
from ..capabilities import Capability
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


def _ranked_profit_curve(delta: FloatNDArray, y_score: FloatNDArray) -> tuple[IntNDArray, FloatNDArray]:
    """Cumulative profit of targeting the top-ranked samples, at every point a threshold can reach.

    Samples with equal scores cannot be separated by any threshold, so each group of tied scores is
    targeted all at once: the curve only has points at the edges of those groups. Taking a fraction
    of a group, e.g. by random tie-breaking, earns in expectation the straight line between its two
    edges, which is how a caller needing the curve at every sample should fill it in.

    Returns
    -------
    n_targeted : NDArray of shape (n_points,)
        Number of samples targeted at each point, ascending, from 0 (nobody) to ``n_samples``.
    profits : NDArray of shape (n_points,)
        Cumulative profit at each point.
    """
    order = np.argsort(y_score)[::-1]
    cumulative = np.concatenate([[0.0], np.cumsum(delta[order])])
    group_ends = np.flatnonzero(np.diff(y_score[order]) != 0) + 1
    n_targeted = np.concatenate([[0], group_ends, [y_score.size]])
    return n_targeted, cumulative[n_targeted]


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
        return self._sample_scorer(y_true, **kwargs)(y_score)

    def _sample_scorer(self, y_true: IntNDArray, **kwargs: Any) -> Callable[[FloatNDArray], float]:
        """
        Prepare the score of fixed labels, for parameter values fixed across many calls.

        The profit differential and the oracle's curve are computed once here. The returned function
        takes the scores and gives the same result as calling this score function.
        """
        _check_parameters(self.delta_equation.free_symbols - {sympy.symbols('y')}, kwargs)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        delta: Float64Array = np.asarray(
            _safe_run_lambda_array(self.delta_function, self.delta_equation, shape=y_true.size, y=y_true, **kwargs),
            dtype=np.float64,
        )
        # Oracle ranking: sorting by the profit differential maximizes cumulative profit at every
        # targeted fraction simultaneously, so this is the "perfect model" curve.
        perfect_profits = np.cumsum(delta[np.argsort(delta)[::-1]])

        def score(y_score: FloatNDArray) -> float:
            return self._score(delta, perfect_profits, np.asarray(y_score, dtype=np.float64).reshape(-1))

        return score

    def _score(self, delta: Float64Array, perfect_profits: Float64Array, y_score: FloatNDArray) -> float:
        n_samples = delta.size
        # The classifier's own ranking, after targeting 1, ..., n samples. Within a group of tied
        # scores the curve is the expected profit under random tie-breaking, so the result does not
        # depend on the order the samples happen to be in.
        n_targeted, curve = _ranked_profit_curve(delta, y_score)
        profits = np.interp(np.arange(1, n_samples + 1), n_targeted, curve)

        # Stop at the point where the oracle's cumulative profit is no longer positive: beyond this
        # point, even the best possible policy is losing money by targeting further samples. The
        # oracle curve rises and then falls, so this also keeps every divisor below strictly positive.
        stop_index: int = (
            int(np.argmax(perfect_profits <= 0)) if np.any(perfect_profits <= 0) else n_samples  # type: ignore[assignment]
        )
        if stop_index == 0:
            # Not even the oracle's best single sample is profitable, so there is no curve.
            return 0.0

        ratios = profits[:stop_index] / perfect_profits[:stop_index]
        if stop_index == 1:
            # A single point spans no area; its mean ratio over that point is the ratio itself.
            return float(ratios[0]) if self.normalize else 0.0

        score = float(trapezoid(ratios, dx=1 / n_samples))
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

    AUEPC evaluates a full ranking, not the outcome of a single sample, so it has no gradient.
    Models that train without gradients can optimize it, such as
    :class:`~empulse.models.ProfTreeClassifier`, :class:`~empulse.models.ProfSRClassifier` and
    :class:`~empulse.models.ProfLogitClassifier` with an optimizer that does not use gradients, but
    gradient-based training (no ``logit_objective`` or ``gradient_boost_objective``) is not
    supported. The score is a ratio of profits, so the penalties of those models are not measured
    against the costs.

    .. seealso::
        :func:`~empulse.metrics.auepc_score` : the underlying metric function.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to normalize the AUEPC score so that a perfect model scores 1.0.
        This is only useful when part of the expected profit curve is negative.
    """

    _capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.RANKING})
    _unitless_score: ClassVar[bool] = True
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
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores used to rank the samples.

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score : float
            The AUEPC score.
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

        The objective is the negated AUEPC of the model plus an elastic-net penalty. AUEPC is a ratio
        of profits, so the penalty is not scaled by the costs.

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
        return RankingLogitValueObjective(
            score=self._score_function._sample_scorer(y_true, **parameters),  # type: ignore[attr-defined]
            features=features,
            penalty=ElasticNetPenalty.from_scale(
                objective_scale=1.0, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, n_samples=features.shape[0]
            ),
        )

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
    delta_equation = _build_delta_equation(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    for symbol in delta_equation.free_symbols:
        delta_equation = delta_equation.subs(symbol, str(symbol) + '_i')
    delta_latex = _latex(delta_equation)

    formula = (
        r'\frac{1}{N}\sum_{k=1}^{N}'
        r'\frac{\sum_{i=1}^{k}\Delta_{\pi(i)}}{\sum_{i=1}^{k}\Delta_{\pi^{*}(i)}}'
        r'\quad\text{where }\Delta_i = ' + delta_latex
    )

    return f'$\\displaystyle {formula}$'
