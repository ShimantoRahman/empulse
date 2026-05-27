from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Self

import numpy as np
import sympy

from ...._types import FloatNDArray, IntNDArray
from ..common import Direction


class LogitObjective(ABC):
    """Class to compute the loss and gradient of a logistic regression objective."""

    @abstractmethod
    def logit_loss(self, weights: FloatNDArray) -> float:
        """
        Compute the loss for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Regularized loss.
        """

    @abstractmethod
    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """
        Compute the gradient of the loss for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Regularized gradient.
        """

    def _logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """Yield gradients for successive weight vectors.

        Because the constants are derived from fixed data and parameters,
        there is no expensive state to reconstruct between steps.  The
        generator accepts the same send-protocol as
        ``MaxProfitLogitGradientPiecewise.logit_gradient_steps`` for API
        compatibility: passing ``(weights, refresh)`` works but ``refresh``
        is silently ignored.

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
                weights, _ = sent
            else:
                weights = sent

            sent = yield self.logit_gradient(weights)

    def logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients for successive weight vectors.

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
        >>> grad = gen.send(new_theta)  # first time gradient is computed from scratch
        >>> grad = gen.send(new_theta)  # gradient computed from cached information
        >>> grad = gen.send((new_theta, True))  # gradient computed from scratch
        >>> gen.close()
        """
        generator = self._logit_gradient_steps()
        next(generator)
        return generator

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """
        Compute the loss and its gradient for minimization.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        loss : float
            Loss minus regularization.
        gradient : ndarray
            Regularized gradient.
        """
        return self.logit_loss(weights), self.logit_gradient(weights)

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """
        Compute the loss and its gradient for minimization.

        Here for backward compatibility.  Delegates to ``logit_loss_gradient``.
        """
        return self.logit_loss_gradient(weights)

    def set_alpha(self, alpha: float) -> None:  # noqa: B027
        """Override the smoothing parameter *alpha* (no-op for objectives without alpha annealing).

        Gradient optimizers with an ``alpha_schedule`` call this before each gradient
        computation to externally drive the annealing schedule.  Objectives that
        implement alpha annealing (e.g. :class:`MaxProfitLogitGradientPiecewise`)
        override this method; all others silently ignore the call.

        Parameters
        ----------
        alpha : float
            New alpha value to use for the next gradient computation.
        """
        # no-op: override in subclasses that support alpha annealing

    def with_indices(self, indices: np.ndarray) -> 'LogitObjective':
        """Return a new objective restricted to the sample subset given by *indices*.

        Used by gradient optimizers for mini-batch training.  The default
        implementation raises :exc:`NotImplementedError`; concrete objectives
        that store their data should override this method.

        Parameters
        ----------
        indices : ndarray of int
            Row indices into the full training set.

        Returns
        -------
        LogitObjective
            A new objective for the selected samples.

        Raises
        ------
        NotImplementedError
            If this objective does not support mini-batch slicing.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support mini-batch training. Override with_indices() to enable it.'
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

    @property
    def _extra_kwargs(self) -> set[str]:
        """Extra keyword arguments accepted by :meth:`score` beyond the cost-matrix parameters.

        Subclasses should override this to declare any additional keyword arguments that
        their :meth:`score` implementation accepts (e.g. ``{'baseline'}`` for
        :class:`~empulse.metrics.Savings`).  These are excluded from the unknown-parameter
        warning raised by :class:`~empulse.metrics.Metric`.
        """
        return set()

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
    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
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
        raise NotImplementedError(f'Optimal threshold is not defined for the {self.name} strategy')

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
        raise NotImplementedError(f'Optimal rate is not defined for the {self.name} strategy')

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """
        Compute the logit loss and its gradient with respect to the logistic regression weights.

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
        logistic_objective : LogitObjective
            A class that implements the logit loss and its gradient.
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
