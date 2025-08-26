import sys
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import sympy

from ..._types import FloatNDArray
from .common import Direction, RateFn, ThresholdFn
from .max_profit_metric import (
    _build_max_profit_optimal_rate,
    _build_max_profit_optimal_threshold,
    _build_max_profit_score,
    _max_profit_score_to_latex,
)


class MetricStrategy(ABC):
    """
    Abstract base class for metric strategies.

    This class defines the interface for metric strategies.
    Metric strategies are used to compute the metric value, gradient, and hessian.
    """

    def __init__(self, name: str, direction: Direction):
        self.name = name
        self.direction = direction

    @abstractmethod
    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""

    @abstractmethod
    def score(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the metric score or loss.

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
            The computed metric score or loss.
        """

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
        raise NotImplementedError(f'Optimal threshold is not defined for the {self.name} strategy')

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
        raise NotImplementedError(f'Optimal rate is not defined for the {self.name} strategy')

    def logit_objective(
        self, features: FloatNDArray, weights: FloatNDArray, y_true: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[float, FloatNDArray]:
        """
        Compute the metric value and the gradient of the metric with respect to logistic regression coefficients.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
        weights : NDArray of shape (n_features,)
            The weights of the logistic regression model.
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
        value : float
            The metric loss to be minimized.
        gradient : NDArray of shape (n_features,)
            The gradient of the metric loss with respect to the logistic regression weights.
        """
        raise NotImplementedError(f'Gradient of the logit function is not defined for the {self.name} strategy')

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
        raise NotImplementedError(f'Gradient of the logit function is not defined for the {self.name} strategy')

    def gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient of the metric with respect to gradient boosting instances.

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
        gradient : NDArray of shape (n_samples,)
            The gradient of the metric loss with respect to the gradient boosting weights.
        hessian : NDArray of shape (n_samples,)
            The hessian of the metric loss with respect to the gradient boosting weights.
        """
        raise NotImplementedError(
            f'Gradient and Hessian of the gradient boosting function is not defined for the {self.name} strategy'
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
        raise NotImplementedError(
            f'Gradient and Hessian of the gradient boosting function is not defined for the {self.name} strategy'
        )

    @abstractmethod
    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(direction={self.direction})'


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

    random_state: int | np.random.RandomState | None, default=None
        The random state to use when ``integration_technique='monte-carlo'`` or
        ``integration_technique='quasi-monte-carlo'``.
        Determines the points sampled from the distribution of the stochastic variables.
        This argument is ignored when ``integration_technique='quad'``.
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
        random_state: np.random.RandomState | int | None = None,
    ):
        super().__init__(name='max profit', direction=Direction.MAXIMIZE)
        if integration_method not in self.INTEGRATION_METHODS:
            raise ValueError(
                f'Integration method {integration_method} is not supported. '
                f'Supported values are {self.INTEGRATION_METHODS}'
            )
        self.integration_method = integration_method
        self.n_mc_samples: int = 2**n_mc_samples_exp
        if isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState(random_state)

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
        self._optimal_threshold: ThresholdFn = _build_max_profit_optimal_threshold(
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
        return self

    def score(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
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
