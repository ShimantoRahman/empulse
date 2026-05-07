import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Literal, Self

import numpy as np
import sympy
from sympy.stats import density, pspace
from sympy.stats.rv import is_random

from ....._types import FloatNDArray, IntNDArray
from ....common import classification_threshold
from ...common import Direction, MetricFn, RateFn, ThresholdFn, _check_parameters, _safe_lambdify, _safe_run_lambda
from ..metric_strategy import MetricStrategy
from .deterministic import (
    MaxProfitBoostGradientDeterministic,
    MaxProfitLogitGradientDeterministic,
    MaxProfitRateDeterministic,
    MaxProfitScoreDeterministic,
)
from .gradient_piecewise import MaxProfitBoostGradientPiecewise, MaxProfitLogitGradientPiecewise
from .monte_carlo import MaxProfitScoreMonteCarlo
from .piecewise import (
    BasePositiveDistribution,
    ComplexRootsError,
    _build_max_profit_rate_piecewise,
    _build_max_profit_score_piecewise,
)
from .quadrature import MaxProfitScoreQuad
from .quasi_monte_carlo import MaxProfitScoreQuasiMonteCarlo, _sympy_dist_to_scipy


def _aggregate_instance_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Replace instance-dependent array-like parameter values with their mean (float).

    Leaves scalar parameters unchanged.
    """
    for key, value in list(parameters.items()):
        if isinstance(value, np.ndarray):
            parameters[key] = float(np.mean(value))
    return parameters


class MaxProfit(MetricStrategy):
    """
    Strategy for the Expected Maximum Profit (EMP) metric.

    Parameters
    ----------
    integration_method: {'auto', 'quad', 'monte-carlo', 'quasi-monte-carlo'}, default='auto'
        The integration method to use when the metric has stochastic variables.

        - If ``'auto'``, the integration method is automatically chosen based on the number of stochastic variables,
          balancing accuracy with execution speed.
          For a single stochastic variable, piecewise integration is used. This is the most accurate method.
          For two stochastic variables, 'quad' is used,
          and for more than two stochastic variables, 'quasi-monte-carlo' is used if all distribution are supported.
          Otherwise, 'monte-carlo' is used.
        - If ``'quad'``, the metric is integrated using the quad function from scipy.
          Be careful, as this can be slow for more than 2 stochastic variables.
        - If ``'monte-carlo'``, the metric is integrated using a Monte Carlo simulation.
          The monte-carlo simulation is less accurate but faster than quad for many stochastic variables.
        - If ``'quasi-monte-carlo'``, the metric is integrated using a Quasi Monte Carlo simulation.
          The quasi-monte-carlo simulation is more accurate than monte-carlo but only supports a few distributions
          present in :mod:`sympy:sympy.stats`:

            - :class:`sympy.stats.Arcsin`
            - :class:`sympy.stats.Beta`
            - :class:`sympy.stats.BetaPrime`
            - :class:`sympy.stats.Chi`
            - :class:`sympy.stats.ChiSquared`
            - :class:`sympy.stats.Erlang`
            - :class:`sympy.stats.Exponential`
            - :class:`sympy.stats.ExGaussian`
            - :class:`sympy.stats.F`
            - :class:`sympy.stats.Gamma`
            - :class:`sympy.stats.GammaInverse`
            - :class:`sympy.stats.GaussianInverse`
            - :class:`sympy.stats.Laplace`
            - :class:`sympy.stats.Logistic`
            - :class:`sympy.stats.LogNormal`
            - :class:`sympy.stats.Lomax`
            - :class:`sympy.stats.Normal`
            - :class:`sympy.stats.Maxwell`
            - :class:`sympy.stats.Moyal`
            - :class:`sympy.stats.Nakagami`
            - :class:`sympy.stats.PowerFunction`
            - :class:`sympy.stats.StudentT`
            - :class:`sympy.stats.Trapezoidal`
            - :class:`sympy.stats.Triangular`
            - :class:`sympy.stats.Uniform`

    n_mc_samples_exp: int
        ``2**n_mc_samples_exp`` is the number of (Quasi-) Monte Carlo samples to use when
        ``integration_technique'monte-carlo'``.
        Increasing the number of samples improves the accuracy of the metric estimation but slows down the speed.
        This argument is ignored when the ``integration_technique='quad'``.

    random_state: int | np.random.Generator | None, default=None
        The random state to use when ``integration_technique='monte-carlo'`` or
        ``integration_technique='quasi-monte-carlo'``.
        Determines the points sampled from the distribution of the stochastic variables.
        This argument is ignored when ``integration_technique='quad'``.

    alpha: float, default=1.0
        Initial temperature used in the smooth sigmoid approximation for ``logit_objective``.

    alpha_growth: float, default=1.1
        Exponential growth factor :math:`\\gamma` of the annealing schedule.
        Set to ``1.0`` to keep a constant temperature.

    alpha_max: float, default=100.0
        Maximum value reached by the annealed temperature.
    """

    INTEGRATION_METHODS: ClassVar[list[Literal['auto', 'quad', 'quasi-monte-carlo', 'monte-carlo']]] = [
        'auto',
        'quad',
        'quasi-monte-carlo',
        'monte-carlo',
    ]

    def __init__(
        self,
        integration_method: Literal['auto', 'quad', 'quasi-monte-carlo', 'monte-carlo'] = 'auto',
        n_mc_samples_exp: int = 16,
        random_state: np.random.Generator | int | None = None,
        alpha: float = 1.0,
        alpha_growth: float = 1.1,
        alpha_max: float = 100.0,
    ):
        super().__init__(name='max profit', direction=Direction.MAXIMIZE)
        if integration_method not in self.INTEGRATION_METHODS:
            raise ValueError(
                f'Integration method {integration_method} is not supported. '
                f'Supported values are {self.INTEGRATION_METHODS}'
            )
        self.integration_method = integration_method
        self.n_mc_samples: int = 2**n_mc_samples_exp
        if alpha <= 0:
            raise ValueError('alpha must be strictly positive.')
        if alpha_growth < 1.0:
            raise ValueError('alpha_growth must be >= 1.0.')
        if alpha_max <= 0:
            raise ValueError('alpha_max must be strictly positive.')
        if alpha > alpha_max:
            raise ValueError('alpha must be <= alpha_max.')
        self.alpha = alpha
        self.alpha_growth = alpha_growth
        self.alpha_max = alpha_max
        self._boost_epoch = 0
        self._boost_objective: MaxProfitBoostGradientDeterministic | None = None
        self._boost_signature: tuple[tuple[int, ...], tuple[tuple[str, float], ...]] | None = None
        if isinstance(random_state, np.random.Generator):
            self._rng: np.random.Generator = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    def _current_boost_alpha(self) -> float:
        """Compute annealed temperature for the current boosting objective evaluation."""
        try:
            alpha = self.alpha * (self.alpha_growth**self._boost_epoch)
        except OverflowError:
            alpha = self.alpha_max
        return float(min(self.alpha_max, alpha))

    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""
        self._score_function = _build_max_profit_score(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            integration_method=self.integration_method,
            n_mc_samples=self.n_mc_samples,
            rng=self._rng,
        )
        self._optimal_rate: RateFn = _build_max_profit_optimal_rate(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            integration_method=self.integration_method,
            n_mc_samples=self.n_mc_samples,
            rng=self._rng,
        )
        # Store symbolic expressions for gradient computation in logit_objective
        self._tp_benefit = tp_benefit
        self._tn_benefit = tn_benefit
        self._fp_cost = fp_cost
        self._fn_cost = fn_cost
        self._boost_epoch = 0
        self._boost_objective = None
        self._boost_signature = None
        return self

    def _prepare_boost_deterministic_objective(
        self, y_true: FloatNDArray, **parameters: FloatNDArray | float
    ) -> MaxProfitBoostGradientDeterministic:
        if not isinstance(self._score_function, MaxProfitScoreDeterministic):
            raise NotImplementedError(
                'gradient_boost_objective is only supported for deterministic MaxProfit metrics. '
                'The metric must not contain any stochastic variables.'
            )

        f0, f1 = sympy.symbols('F_0 F_1')
        try:
            profit_poly = sympy.Poly(sympy.expand(self._score_function.profit_function), f0, f1)
        except sympy.polys.polyerrors.PolynomialError as exc:
            raise NotImplementedError(
                'gradient_boost_objective currently supports profit functions linear in F_0 and F_1 only.'
            ) from exc
        if profit_poly.total_degree() > 1:
            raise NotImplementedError(
                'gradient_boost_objective currently supports profit functions linear in F_0 and F_1 only.'
            )

        _check_parameters(self._score_function.deterministic_symbols, parameters)
        agg_params = _aggregate_instance_parameters(dict(parameters))

        tp_val = float(_safe_run_lambda(_safe_lambdify(self._tp_benefit), self._tp_benefit, **agg_params))
        fn_val = float(_safe_run_lambda(_safe_lambdify(self._fn_cost), self._fn_cost, **agg_params))
        tn_val = float(_safe_run_lambda(_safe_lambdify(self._tn_benefit), self._tn_benefit, **agg_params))
        fp_val = float(_safe_run_lambda(_safe_lambdify(self._fp_cost), self._fp_cost, **agg_params))

        return MaxProfitBoostGradientDeterministic(
            profit_function=self._score_function.profit_function,
            deterministic_symbols=self._score_function.deterministic_symbols,
            y_true=np.asarray(y_true),
            tp_benefit=tp_val,
            tn_benefit=tn_val,
            fp_cost=fp_val,
            fn_cost=fn_val,
            parameters=agg_params,
        )

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the maximum profit score.

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
            The maximum profit score.
        """
        parameters = _aggregate_instance_parameters(parameters)
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
        rate = self.optimal_rate(y_true, y_score, **parameters)
        return classification_threshold(y_true, y_score, rate)

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
        parameters = _aggregate_instance_parameters(parameters)
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
    ) -> Callable[[FloatNDArray], tuple[float, FloatNDArray]]:
        """
        Build a function which computes the metric value and the gradient of the metric w.r.t logistic coefficients.

        Uses the Envelope Theorem combined with a smooth sigmoid approximation of the ROC curve to derive
        an analytically differentiable proxy for the Expected Maximum Profit.

        Only supported for deterministic metrics
        or metrics with one stochastic variable following a distribution with positive support.

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
            A function that takes logistic regression weights as input and returns the negated metric value
            and its gradient (negated for minimization).
            The function signature is:
            ``logistic_objective(weights) -> (value, gradient)``

        Raises
        ------
        NotImplementedError
            If the metric contains stochastic variables (only deterministic case is supported).
        """
        _check_parameters(self._score_function.deterministic_symbols, parameters)
        agg_params = _aggregate_instance_parameters(dict(parameters))

        # 1. Deterministic Route
        if isinstance(self._score_function, MaxProfitScoreDeterministic):
            tp_val = float(_safe_run_lambda(_safe_lambdify(self._tp_benefit), self._tp_benefit, **agg_params))
            fn_val = float(_safe_run_lambda(_safe_lambdify(self._fn_cost), self._fn_cost, **agg_params))
            tn_val = float(_safe_run_lambda(_safe_lambdify(self._tn_benefit), self._tn_benefit, **agg_params))
            fp_val = float(_safe_run_lambda(_safe_lambdify(self._fp_cost), self._fp_cost, **agg_params))

            return MaxProfitLogitGradientDeterministic(
                profit_function=self._score_function.profit_function,
                deterministic_symbols=self._score_function.deterministic_symbols,
                features=features,
                y_true=y_true,
                C=C,
                l1_ratio=l1_ratio,
                soft_threshold=soft_threshold,
                fit_intercept=fit_intercept,
                alpha_0=self.alpha,
                alpha_growth=self.alpha_growth,
                alpha_max=self.alpha_max,
                tp_benefit=tp_val,
                tn_benefit=tn_val,
                fp_cost=fp_val,
                fn_cost=fn_val,
                parameters=agg_params,
            )

        # 2. Stochastic Piecewise Route (e.g., Beta, Gamma, Pareto)
        elif isinstance(self._score_function, BasePositiveDistribution):
            return MaxProfitLogitGradientPiecewise(
                score_function=self._score_function,
                features=features,
                y_true=y_true,
                C=C,
                l1_ratio=l1_ratio,
                soft_threshold=soft_threshold,
                fit_intercept=fit_intercept,
                alpha_0=self.alpha,
                alpha_growth=self.alpha_growth,
                alpha_max=self.alpha_max,
                parameters=agg_params,
            )
        else:
            raise NotImplementedError(
                'logit_objective is currently only supported for Deterministic and '
                'BasePositiveDistribution stochastic metrics.'
            )

    def gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute gradient and hessian for boosting with MaxProfit.

        Automatically handles deterministic and stochastic piecewise integrals.
        """
        alpha = self._current_boost_alpha()
        self._boost_epoch += 1

        y_true_arr = np.asarray(y_true).reshape(-1)
        agg_params = _aggregate_instance_parameters(dict(parameters))
        param_signature = tuple(sorted((k, float(v)) for k, v in agg_params.items()))
        signature = (y_true_arr.shape, param_signature)

        # Build the objective once and cache it based on the signature
        if self._boost_objective is None or self._boost_signature != signature:
            # Route: Deterministic Linear EMP
            if isinstance(self._score_function, MaxProfitScoreDeterministic):
                self._boost_objective = self._prepare_boost_deterministic_objective(y_true_arr, **agg_params)

            # Route: Stochastic Piecewise EMP
            elif isinstance(self._score_function, BasePositiveDistribution):
                self._boost_objective = self._prepare_boost_piecewise_objective(y_true_arr, **agg_params)

            else:
                raise NotImplementedError(
                    'gradient_boost_objective is currently only supported for Deterministic '
                    'and BasePositiveDistribution stochastic metrics.'
                )

            self._boost_signature = signature

        # Evaluate and return gradient and hessian for GBDT
        return self._boost_objective(np.asarray(y_score), alpha)

    def _prepare_boost_piecewise_objective(
        self, y_true: FloatNDArray, **parameters: FloatNDArray | float
    ) -> MaxProfitBoostGradientPiecewise:

        _check_parameters(self._score_function.deterministic_symbols, parameters)
        agg_params = _aggregate_instance_parameters(dict(parameters))

        return MaxProfitBoostGradientPiecewise(
            score_function=self._score_function,
            y_true=y_true,
            parameters=agg_params,
        )

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _max_profit_score_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(direction={self.direction}'
            f', integration_method={self.integration_method!r}, n_mc_samples={self.n_mc_samples}'
            f', random_state={self._rng})'
        )


def _build_max_profit_score(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> MetricFn:
    random_symbols, deterministic_symbols = _identify_symbols(tp_benefit, tn_benefit, fp_cost, fn_cost)
    n_random = len(random_symbols)

    profit_function = _build_profit_function(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    if n_random == 0:
        max_profit_score: MetricFn = MaxProfitScoreDeterministic(profit_function, deterministic_symbols)
    else:
        max_profit_score = _build_max_profit_stochastic(
            profit_function,
            random_symbols,
            deterministic_symbols,
            integration_method=integration_method,
            n_mc_samples=n_mc_samples,
            rng=rng,
        )
    return max_profit_score


def _to_threshold_function(rate_function: RateFn) -> ThresholdFn:
    def threshold_function(y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        rate = rate_function(y_true, y_score, **kwargs)
        return classification_threshold(y_true, y_score, rate)

    return threshold_function


def _build_max_profit_optimal_threshold(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> ThresholdFn:
    rate_fn = _build_max_profit_optimal_rate(
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        integration_method=integration_method,
        n_mc_samples=n_mc_samples,
        rng=rng,
    )
    return _to_threshold_function(rate_fn)


def _build_max_profit_optimal_rate(
    tp_benefit: sympy.Expr,
    tn_benefit: sympy.Expr,
    fp_cost: sympy.Expr,
    fn_cost: sympy.Expr,
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> RateFn:
    random_symbols, deterministic_symbols = _identify_symbols(tp_benefit, tn_benefit, fp_cost, fn_cost)
    n_random = len(random_symbols)

    profit_function = _build_profit_function(
        tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    rate_function = _build_rate_function()
    if n_random == 0:
        optimal_rate: MetricFn = MaxProfitRateDeterministic(profit_function, deterministic_symbols)
    else:
        optimal_rate = _build_max_profit_rate_stochastic(
            profit_function, rate_function, random_symbols, deterministic_symbols, integration_method, n_mc_samples, rng
        )
    return optimal_rate


def _identify_symbols(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    """Identify random and deterministic symbols in the profit function."""
    terms = tp_benefit + tn_benefit + fp_cost + fn_cost
    random_symbols = [symbol for symbol in terms.free_symbols if is_random(symbol)]
    deterministic_symbols = [symbol for symbol in terms.free_symbols if not is_random(symbol)]
    return random_symbols, deterministic_symbols


def _build_profit_function(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
    profit_function = (
        tp_benefit * pos_prior * tpr
        + tn_benefit * neg_prior * (1 - fpr)
        - fn_cost * pos_prior * (1 - tpr)
        - neg_prior * fp_cost * fpr
    )
    return profit_function


def _build_rate_function() -> sympy.Expr:
    pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
    return pos_prior * tpr + neg_prior * fpr


def _build_max_profit_rate_stochastic(
    profit_function: sympy.Expr,
    rate_function: sympy.Expr,
    random_symbols: Sequence[sympy.Symbol],
    deterministic_symbols: Iterable[sympy.Symbol],
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> RateFn:
    """Compute the maximum profit for one or more stochastic variables."""
    n_random = len(random_symbols)
    if integration_method == 'auto':
        if n_random == 1 and is_linear_in(profit_function, random_symbols[0]):
            return _build_max_profit_rate_piecewise(
                profit_function, rate_function, random_symbols[0], deterministic_symbols
            )
        elif n_random <= 2:
            return MaxProfitScoreQuad(profit_function, rate_function, random_symbols, deterministic_symbols)
        elif _support_all_distributions(random_symbols):
            return MaxProfitScoreQuasiMonteCarlo(
                profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
        else:
            return MaxProfitScoreMonteCarlo(
                profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
    elif integration_method == 'quad':
        return MaxProfitScoreQuad(profit_function, rate_function, random_symbols, deterministic_symbols)
    elif integration_method == 'monte-carlo':
        return MaxProfitScoreMonteCarlo(
            profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    elif integration_method == 'quasi-monte-carlo':
        return MaxProfitScoreQuasiMonteCarlo(
            profit_function, rate_function, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    else:
        raise ValueError(f'Integration method {integration_method} is not supported')


def _support_all_distributions(random_symbols: Iterable[sympy.Symbol]) -> bool:
    return all(pspace(r).distribution.__class__ in _sympy_dist_to_scipy for r in random_symbols)


def is_linear_in(expr: sympy.Expr, x: sympy.Symbol) -> bool:
    """Test whether the expression is linear in `x`."""
    expr = sympy.collect(sympy.factor(expr, x), x)
    try:
        poly = sympy.Poly(expr, x)
    except sympy.polys.polyerrors.PolynomialError:
        return False
    return poly.degree() <= 1  # type: ignore[no-any-return]


def is_polynomial_in(expr: sympy.Expr, x: sympy.Symbol) -> bool:
    """Test whether the expression is a valid polynomial in `x`."""
    # Expanding is generally safer than factoring for polynomial construction
    expr = sympy.collect(sympy.expand(expr), x)
    try:
        # If SymPy can construct a Poly, it contains only non-negative integer powers of x
        sympy.Poly(expr, x)
    except sympy.polys.polyerrors.PolynomialError:
        return False

    return True


def _build_max_profit_stochastic(
    profit_function: sympy.Expr,
    random_symbols: Sequence[sympy.Symbol],
    deterministic_symbols: Iterable[sympy.Symbol],
    integration_method: str,
    n_mc_samples: int,
    rng: np.random.Generator,
) -> MetricFn:
    """Compute the maximum profit for one or more stochastic variables."""
    n_random = len(random_symbols)
    if integration_method == 'auto':
        if n_random == 1 and is_polynomial_in(profit_function, random_symbols[0]):
            try:
                return _build_max_profit_score_piecewise(profit_function, random_symbols[0], deterministic_symbols)
            except ComplexRootsError:
                warnings.warn(
                    'The profit function polynomial has complex roots for the stochastic variable, '
                    'making piecewise integration inapplicable. '
                    'Falling back to numerical quadrature. To suppress this warning, pass '
                    "integration_method='quad' explicitly to MaxProfit().",
                    UserWarning,
                    stacklevel=2,
                )
                return MaxProfitScoreQuad(profit_function, None, random_symbols, deterministic_symbols)
        if n_random <= 2:
            return MaxProfitScoreQuad(profit_function, None, random_symbols, deterministic_symbols)
        elif _support_all_distributions(random_symbols):
            return MaxProfitScoreQuasiMonteCarlo(
                profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
        else:
            return MaxProfitScoreMonteCarlo(
                profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng
            )
    elif integration_method == 'quad':
        return MaxProfitScoreQuad(profit_function, None, random_symbols, deterministic_symbols)
    elif integration_method == 'monte-carlo':
        return MaxProfitScoreMonteCarlo(profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng)
    elif integration_method == 'quasi-monte-carlo':
        return MaxProfitScoreQuasiMonteCarlo(
            profit_function, None, random_symbols, deterministic_symbols, n_mc_samples, rng
        )
    else:
        raise ValueError(f'Integration method {integration_method} is not supported')


def _max_profit_score_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    profit_function = _build_profit_function(
        tp_benefit=-tp_benefit, tn_benefit=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
    )
    random_symbols = [symbol for symbol in profit_function.free_symbols if is_random(symbol)]

    if random_symbols:
        integrand = profit_function
        for random_symbol in random_symbols:
            integrand *= density(random_symbol).pdf(random_symbol)
        integral = integrand
        for random_symbol in random_symbols:
            lower_bound, upper_bound = pspace(random_symbol).domain.set.args[:2]
            integral = sympy.Integral(integral, (random_symbol, lower_bound, upper_bound))

        output = latex(integral, mode='plain', order=None)
    else:
        output = latex(profit_function, mode='plain', order=None)
    return f'$\\displaystyle {output}$'
