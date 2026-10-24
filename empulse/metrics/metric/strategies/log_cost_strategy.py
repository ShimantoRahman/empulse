from functools import cached_property
from typing import Any, ClassVar, Self

import numpy as np
import sympy

from ...._types import Float64Array, FloatNDArray, IntNDArray
from ..._loss import (
    cy_log_cost_boost_grad_hess,
    cy_log_cost_gradient,
    cy_log_cost_loss,
    cy_log_cost_loss_gradient,
)
from .._compile import (
    MetricFn,
    _safe_lambdify,
    _safe_run_lambda,
    _safe_run_lambda_array,
)
from .._direction import Direction
from .._parameter_domain import LABEL_AND_SCORE_SYMBOLS, LABEL_SYMBOLS, _check_parameters
from .._stochastic import expectation_over_random_vars
from .._symbolic import _latex
from ..capabilities import Capability
from .cost_strategy import (
    CostOptimalThreshold,
    RateAtThreshold,
    _LogitFromClassTerms,
    _loss_constants,
    _LossConstantsObjective,
)
from .metric_strategy import MetricStrategy, _CompiledDecisions


class LogCostLogitObjective(_LossConstantsObjective):
    """
    Precomputed log-cost objective for logistic regression.

    Holds the constants derived from the data and exposes the :class:`~empulse.metrics.LogitObjective`
    interface. Unlike :class:`~empulse.metrics.Cost`'s objective, the per-sample loss is not linear in the
    predicted probability (it involves ``log(s)`` and ``log(1 - s)``), so the gradient depends on the
    current predictions and is recomputed on every call rather than reduced to a single precomputed
    constant.

    As with :class:`~empulse.metrics.Cost`'s objective, the loss, the gradient and the penalty are all
    computed by a compiled kernel, which takes the penalty as the scalars cached on the objective. The
    penalty is scaled as for :class:`~empulse.metrics.Cost`: the per-sample gradient
    ``c1 (1 - s) - c2 s`` has magnitude ``|c1 - c2| / 2`` at ``s = 0.5``.
    """

    def _kernel_arguments(self, weights: FloatNDArray) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]:
        return np.ascontiguousarray(weights, dtype=np.float64), self.features, self.loss_const1, self.loss_const2

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the regularized ``(loss, gradient)`` for *weights*."""
        return cy_log_cost_loss_gradient(  # type: ignore[no-any-return]
            *self._kernel_arguments(weights), self._l1_weight, self._l2_weight, self._start_coef, self.n_threads
        )

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Return only the regularized scalar loss for *weights*, without computing the gradient."""
        return float(
            cy_log_cost_loss(
                *self._kernel_arguments(weights),
                self._l1_weight,
                self._l2_weight,
                self._start_coef,
                self.n_threads,
            )
        )

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the regularized gradient for *weights*, without computing the loss."""
        return cy_log_cost_gradient(  # type: ignore[return-value]
            *self._kernel_arguments(weights), self._l1_weight, self._l2_weight, self._start_coef, self.n_threads
        )

    def data_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the unpenalized ``(loss, gradient)`` for *weights*.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        loss : float
            Loss of the data term alone.
        gradient : ndarray
            Gradient of the data term alone.
        """
        return cy_log_cost_loss_gradient(  # type: ignore[no-any-return]
            *self._kernel_arguments(weights), n_threads=self.n_threads
        )

    def data_loss(self, weights: FloatNDArray) -> float:
        """Return only the unpenalized scalar loss for *weights*.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        float
            Loss of the data term alone.
        """
        return float(cy_log_cost_loss(*self._kernel_arguments(weights), n_threads=self.n_threads))

    def data_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the unpenalized gradient vector for *weights*.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient of the data term alone.
        """
        return cy_log_cost_gradient(  # type: ignore[return-value]
            *self._kernel_arguments(weights), n_threads=self.n_threads
        )


class LogCostBoostGradient:
    """Class to compute the gradient and hessian of the log-cost metric for gradient boosting."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.loss_const1_eq, self.loss_const2_eq = _loss_constants(
            sympy.Symbol('y'), tp_benefit, tn_benefit, fp_cost, fn_cost
        )
        self.loss_const1_fn = _safe_lambdify(self.loss_const1_eq)
        self.loss_const2_fn = _safe_lambdify(self.loss_const2_eq)

    @cached_property
    def _parameter_symbols(self) -> set[sympy.Symbol]:
        return set((self.loss_const1_eq.free_symbols | self.loss_const2_eq.free_symbols) - LABEL_SYMBOLS)

    def __call__(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """Compute the gradient and hessian of the log-cost loss wrt gradient boosting raw scores."""
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        _check_parameters(self._parameter_symbols, parameters)
        loss_const1 = _safe_run_lambda_array(
            self.loss_const1_fn, self.loss_const1_eq, shape=y_true.shape[0], y=y_true, **parameters
        )
        loss_const2 = _safe_run_lambda_array(
            self.loss_const2_fn, self.loss_const2_eq, shape=y_true.shape[0], y=y_true, **parameters
        )

        return cy_log_cost_boost_grad_hess(  # type: ignore[no-any-return]
            np.ascontiguousarray(y_score, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(loss_const1, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(loss_const2, dtype=np.float64).reshape(-1),
        )


class LogCost(_CompiledDecisions, _LogitFromClassTerms, MetricStrategy):
    r"""
    Strategy for the Expected Log Cost metric.

    The expected log cost is closely related to (weighted) cross-entropy / log loss: it replaces
    the predicted probability ``s`` in :class:`~empulse.metrics.Cost`'s linear cost function with
    ``log(s)`` and ``log(1 - s)``, giving the per-sample loss

    .. math::

        y \left( C_{TP} \log s + C_{FN} \log (1 - s) \right)
        + (1 - y) \left( C_{TN} \log (1 - s) + C_{FP} \log s \right)

    Each cost weights the log of the probability assigned to the outcome it names, so the terms
    that reproduce log loss are the *correct*-classification ones. Setting
    ``tp_cost = tn_cost = -1`` (equivalently, a true-positive and true-negative benefit of ``1``)
    with ``fp_cost = fn_cost = 0`` leaves exactly the standard log loss,
    :math:`-y \log s - (1 - y) \log (1 - s)`.

    Fitting :class:`~empulse.models.CSLogitClassifier` on that metric with ``l1_ratio=0`` and
    ``b=1`` reproduces :class:`~sklearn:sklearn.linear_model.LogisticRegression` at the same ``C``,
    up to solver tolerance.

    .. code-block:: python

        from empulse.metrics import CostMatrix, LogCost, Metric

        log_loss = Metric(CostMatrix().add_tp_benefit('b').add_tn_benefit('b'), LogCost())

    .. warning::

        Weighting the *mis*-classification terms instead, with ``add_fp_cost`` and ``add_fn_cost``,
        is a different objective rather than a reweighted log loss. It puts the log on the
        probability assigned to the wrong outcome, so a positive sample contributes
        :math:`\log (1 - s)`, whose gradient grows without bound as that sample is classified
        *correctly* instead of flattening out. Weight cross-entropy per class through
        ``tp_cost``/``tn_cost`` (or their benefits).

    .. seealso::
        :func:`~empulse.metrics.expected_log_cost_loss` : the underlying metric function.
    """

    _name: str = 'log cost'
    _direction: Direction = Direction.MINIMIZE
    #: `optimal_threshold`/`optimal_rate` depend only on the cost values, like `Cost`'s.
    _capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.COST_ONLY_DECISION})
    _logit_objective_type: ClassVar[type[_LossConstantsObjective]] = LogCostLogitObjective

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
        tp_benefit, tn_benefit, fp_cost, fn_cost = expectation_over_random_vars(
            tp_benefit, tn_benefit, fp_cost, fn_cost
        )
        self._compile_class_terms(tp_benefit, tn_benefit, fp_cost, fn_cost)

        self._score_function: MetricFn = LogCostLoss(
            tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self._optimal_threshold = LogCostOptimalThreshold(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        self._optimal_rate = RateAtThreshold(self._optimal_threshold)
        self._boost_gradient = LogCostBoostGradient(
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        return self

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the metric expected log cost loss.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted (calibrated) probabilities.

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score : float
            The expected log cost loss.
        """
        return self._score_function(y_true, y_score, **parameters)

    def gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient and hessian of the metric with respect to gradient boosting instances.

        Unlike :class:`~empulse.metrics.Cost`, the log-cost per-sample loss is not linear in the
        predicted probability, so the gradient and hessian are recomputed directly from the current
        (raw) boosting scores on every call, rather than through a precomputed constant.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The current (raw, pre-sigmoid) boosting scores.

        **parameters : float or array-like of shape (n_samples,)
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
        return self._boost_gradient(y_true, y_score, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _log_cost_loss_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


class LogCostOptimalThreshold(CostOptimalThreshold):
    """
    Bayes-optimal threshold for scores that minimize the expected log cost.

    With positive-class probability ``q``, the expected log cost of score ``s`` is
    ``A log s + B log(1 - s)``, where ``A`` and ``B`` are the expected linear costs of a positive and a
    negative decision. Its minimizer is ``s = A / (A + B)``, which is 0.5 exactly where ``A == B``, the
    probability at which the linear decision flips.
    """

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: Any) -> FloatNDArray | float:
        """Compute the optimal threshold(s). `y_true` and `y_score` are unused and kept for API compatibility."""
        probability_threshold = super().__call__(y_true, y_score, **parameters)
        if np.isscalar(probability_threshold):
            return 0.5 if 0.0 < probability_threshold < 1.0 else float(probability_threshold)  # type: ignore[operator, arg-type]
        threshold = np.asarray(probability_threshold, dtype=np.float64)
        return np.where((threshold > 0.0) & (threshold < 1.0), 0.5, threshold)


class LogCostLoss:
    """Class to compute the log-cost metric for binary classification."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_log_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.cost_function = _safe_lambdify(self.cost_equation)

    @cached_property
    def _parameter_symbols(self) -> set[sympy.Symbol]:
        return set(self.cost_equation.free_symbols - LABEL_AND_SCORE_SYMBOLS)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the log-cost loss."""
        _check_parameters(self._parameter_symbols, kwargs)
        y_score = np.asarray(y_score, dtype=np.float64)
        epsilon = np.finfo(y_score.dtype).eps
        y_score = np.clip(y_score, epsilon, 1 - epsilon)
        return float(np.mean(_safe_run_lambda(self.cost_function, self.cost_equation, y=y_true, s=y_score, **kwargs)))


def _build_log_cost_equation(
    tp_cost: sympy.Expr, tn_cost: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> sympy.Expr:
    """Build the per-sample log-cost equation as a sympy expression in y (label) and s (score).

    This is the single canonical form used both for numeric evaluation (via lambdify) and for LaTeX rendering.
    """
    y, s = sympy.symbols('y s')
    log_s = sympy.log(s)
    log_inv_s = sympy.log(1 - s)
    cost_function = y * (log_s * tp_cost + log_inv_s * fn_cost) + (1 - y) * (log_inv_s * tn_cost + log_s * fp_cost)
    return cost_function


def _log_cost_loss_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    i, N = sympy.symbols('i N')  # ruff: ignore[non-lowercase-variable-in-function]
    cost_function = (1 / N) * sympy.Sum(
        _build_log_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 1, N)
    )

    for symbol in cost_function.free_symbols:
        if symbol != N:
            cost_function = cost_function.subs(symbol, str(symbol) + '_i')

    output = _latex(cost_function)

    return f'$\\displaystyle {output}$'
