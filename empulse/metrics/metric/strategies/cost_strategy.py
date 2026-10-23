import copy
import warnings
from collections.abc import Generator
from typing import Any, ClassVar, Self

import numpy as np
import sympy

from ...._common._objective import ElasticNetPenalty, LogitObjective
from ...._types import Float64Array, FloatNDArray, IntNDArray
from ..._loss import cy_logit_gradient, cy_logit_loss, cy_logit_loss_gradient
from .._compile import (
    BoostGradientConst,
    MetricFn,
    RateFn,
    ThresholdFn,
    _safe_lambdify,
    _safe_run_lambda,
    _safe_run_lambda_array,
)
from .._direction import Direction
from .._parameter_domain import LABEL_AND_SCORE_SYMBOLS, LABEL_SYMBOLS, _check_parameters
from .._stochastic import expectation_over_random_vars
from .._symbolic import _latex
from ..capabilities import Capability
from ._class_terms import ClassTerms
from ._training_signal import warn_if_no_training_signal
from .metric_strategy import MetricStrategy


class _CachedPenaltyWeights:
    """
    Keep an objective's penalty resolved into the three scalars the Cython kernels take.

    The penalty rarely changes, but the ``logit_*`` methods are re-entered hundreds of times by
    L-BFGS-B and thousands of times by the genetic optimizers, so resolving it on every call is
    measurable overhead on small problems. The scalars are refreshed whenever :attr:`penalty` is
    assigned, since it is replaced after construction: a :class:`~empulse.metrics.MixtureMetric`
    strips its components' penalties and applies a single one of its own.
    """

    fit_intercept: bool
    _penalty: ElasticNetPenalty | None = None
    _l1_weight: float = 0.0
    _l2_weight: float = 0.0
    _start_coef: int = 1

    @property
    def penalty(self) -> ElasticNetPenalty | None:
        """Penalty applied on top of the data term, see :attr:`~empulse.metrics.LogitObjective.penalty`."""
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: ElasticNetPenalty | None) -> None:
        self._penalty = penalty
        if penalty is None or not penalty.is_active:
            self._l1_weight = 0.0
            self._l2_weight = 0.0
            self._start_coef = 1 if self.fit_intercept else 0
        else:
            self._l1_weight = penalty.l1_weight
            self._l2_weight = penalty.l2_weight
            self._start_coef = penalty.start_coef

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Objectives pickled by older versions stored the penalty as `penalty`, and the cost
        # objective a `grad_const` array the kernels no longer take.
        if 'penalty' in state:
            state['_penalty'] = state.pop('penalty')
        state.pop('grad_const', None)
        self.__dict__.update(state)


class CostLogitObjective(_CachedPenaltyWeights, LogitObjective):
    """
    Precomputed cost-metric objective for logistic regression.

    Holds the constants derived from the data and exposes the
    :class:`~empulse.metrics.LogitObjective` interface. The expected cost is linear in the
    predicted probability, so the derivative of the cost with respect to the probability is a
    per-sample constant, ``loss_const1 - loss_const2``.

    The regularized ``logit_*`` methods add the penalty inside the Cython kernel rather than
    through :class:`~empulse.metrics.ElasticNetPenalty` in numpy. The penalty is only
    ``O(n_features)`` against the data term at ``O(n_samples * n_features)``, but a numpy round
    trip costs a fixed ~9 microseconds per evaluation, which is over half of a whole evaluation at
    moderate sample sizes -- and these objectives are re-entered hundreds of times by L-BFGS-B and
    thousands of times by the genetic optimizers. The kernel therefore takes the two already-scaled
    penalty weights, so the scaling policy stays in Python and only the arithmetic is compiled.
    """

    def __init__(
        self,
        *,
        tp_benefit: Float64Array,
        tn_benefit: Float64Array,
        fp_cost: Float64Array,
        fn_cost: Float64Array,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
    ) -> None:
        loss_const1 = y_true * -tp_benefit + (1 - y_true) * fp_cost
        loss_const2 = y_true * fn_cost - (1 - y_true) * tn_benefit
        # Derivative of the expected cost wrt the predicted probability, per sample. The kernels form
        # it from the two loss constants; here it is the magnitude the penalty is scaled against.
        grad_factor = np.asarray(loss_const1 - loss_const2, dtype=np.float64)
        warn_if_no_training_signal(grad_factor, 'expected cost')

        # The Cython kernels take C-contiguous float64 arrays, so convert them once here rather than
        # on every call. Features given in Fortran order would otherwise be rejected.
        self.loss_const1: Float64Array = np.ascontiguousarray(loss_const1, dtype=np.float64).reshape(-1)
        self.loss_const2: Float64Array = np.ascontiguousarray(loss_const2, dtype=np.float64).reshape(-1)
        self.features: Float64Array = np.ascontiguousarray(features, dtype=np.float64)
        self.fit_intercept = fit_intercept
        self.penalty = ElasticNetPenalty.from_scale(
            objective_scale=float(np.mean(np.abs(grad_factor))),
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            n_samples=self.features.shape[0],
        )

    def with_indices(self, indices: FloatNDArray) -> 'CostLogitObjective':
        """Return a new objective restricted to the sample subset given by *indices*.

        The pre-computed constant arrays (``loss_const1``, ``loss_const2``,
        ``features``) are sliced; all scalar attributes are shared.  This is
        very cheap compared to rebuilding from scratch.

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
        obj.loss_const1 = self.loss_const1[indices]
        obj.loss_const2 = self.loss_const2[indices]
        obj.features = self.features[indices]
        return obj

    def __call__(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return ``(loss, gradient)`` for *weights*.  Delegates to :meth:`logit_loss_gradient`."""
        return self.logit_loss_gradient(weights)

    def logit_loss_gradient(self, weights: FloatNDArray) -> tuple[float, FloatNDArray]:
        """Return the regularized ``(loss, gradient)`` for *weights*."""
        return cy_logit_loss_gradient(  # type: ignore[no-any-return]
            np.ascontiguousarray(weights, dtype=np.float64),
            self.features,
            self.loss_const1,
            self.loss_const2,
            self._l1_weight,
            self._l2_weight,
            self._start_coef,
            self.n_threads,
        )

    def logit_loss(self, weights: FloatNDArray) -> float:
        """Return only the regularized scalar loss for *weights*.

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
                np.ascontiguousarray(weights, dtype=np.float64),
                self.features,
                self.loss_const1,
                self.loss_const2,
                self._l1_weight,
                self._l2_weight,
                self._start_coef,
                self.n_threads,
            )
        )

    def logit_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the regularized gradient vector for *weights*.

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
            np.ascontiguousarray(weights, dtype=np.float64),
            self.features,
            self.loss_const1,
            self.loss_const2,
            self._l1_weight,
            self._l2_weight,
            self._start_coef,
            self.n_threads,
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
        return cy_logit_loss_gradient(  # type: ignore[no-any-return]
            np.ascontiguousarray(weights, dtype=np.float64),
            self.features,
            self.loss_const1,
            self.loss_const2,
            n_threads=self.n_threads,
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
        return float(
            cy_logit_loss(
                np.ascontiguousarray(weights, dtype=np.float64),
                self.features,
                self.loss_const1,
                self.loss_const2,
                n_threads=self.n_threads,
            )
        )

    def data_gradient(self, weights: FloatNDArray) -> FloatNDArray:
        """Return only the unpenalized gradient vector for *weights*.

        Parameters
        ----------
        weights : ndarray
            Coefficient vector.

        Returns
        -------
        ndarray
            Gradient of the data term alone, matched in shape to *weights*.
        """
        return cy_logit_gradient(  # type: ignore[return-value]
            np.ascontiguousarray(weights, dtype=np.float64),
            self.features,
            self.loss_const1,
            self.loss_const2,
            n_threads=self.n_threads,
        )

    def _logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """Yield gradients for successive weight vectors.

        Because the constants are derived from fixed data and parameters,
        there is no expensive state to reconstruct between steps.  The
        generator accepts the same send-protocol as
        ``MaxProfitLogitGradientPiecewise.logit_gradient_steps`` for API
        compatibility: send in either a ``weights`` vector or a
        ``(weights, refresh)`` tuple, where ``refresh`` is silently ignored.

        Yields
        ------
        gradient : ndarray
            Gradient at the weights last sent in.
        """
        weights: FloatNDArray

        sent = yield  # type: ignore[misc]

        while True:
            if sent is None:
                return
            if isinstance(sent, tuple):
                weights, _ = sent  # refresh ignored - the constants are fixed
            else:
                weights = sent

            sent = yield self.logit_gradient(weights)

    def logit_gradient_steps(self) -> Generator[FloatNDArray, FloatNDArray | tuple[FloatNDArray, bool] | None, None]:
        """
        Yield gradients for successive weight vectors.

        Because the constants are derived from fixed data and parameters,
        there is no expensive state to reconstruct between steps.  The
        generator accepts the same send-protocol as ``MaxProfitLogitGradientPiecewise.logit_gradient_steps`` for API
        compatibility. Send in either a ``weights`` vector or a ``(weights, refresh)`` tuple;
        ``refresh`` is accepted but ignored here.

        Yields
        ------
        gradient : ndarray
            Gradient at the weights last sent in.

        Examples
        --------
        Driving the generator by hand (``objective`` is a built objective,
        ``theta`` a coefficient vector)::

            gen = objective.logit_gradient_steps()
            grad = gen.send(theta)
            gen.close()
        """
        generator = self._logit_gradient_steps()
        next(generator)
        return generator


class Cost(ClassTerms, MetricStrategy):
    """Strategy for the Expected Cost metric."""

    _name: str = 'cost'
    _direction: Direction = Direction.MINIMIZE
    #: `optimal_threshold`/`optimal_rate` are algebraic solves of the cost equation, so they
    #: depend only on the cost values, not on the actual `y_true`/`y_score`. Inherited by `Profit`
    #: and `Savings`, for which the same is true, as is `EXPECTED_COST`: savings are a decreasing affine
    #: map of the expected cost.
    _capabilities: ClassVar[frozenset[Capability]] = frozenset({
        Capability.COST_ONLY_DECISION,
        Capability.EXPECTED_COST,
    })

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
        optimal_threshold : float | FloatNDArray
            The optimal classification threshold(s).
        """
        return self._optimal_threshold(y_true, y_score, **parameters)

    def optimal_rate(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the predicted positive rate to optimize the metric value.

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
        optimal_rate : float
            The optimal predicted positive rate.
        """
        return self._optimal_rate(y_true, y_score, **parameters)

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
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
        fit_intercept : bool
            Specifies if an intercept should be included in the model.
        **parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        logistic_objective : Callable[[NDArray], tuple[float, NDArray]]
            A function that takes logistic regression weights as input and returns the metric value and its gradient.
            The function signature is:
            ``logistic_objective(weights) -> (value, gradient)``.
        """
        tp_val, tn_val, fp_val, fn_val = self._evaluate_class_terms(**parameters)
        return CostLogitObjective(
            tp_benefit=tp_val,
            tn_benefit=tn_val,
            fp_cost=fp_val,
            fn_cost=fn_val,
            features=features,
            y_true=y_true,
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
        )

    def prepare_boost_objective(self, y_true: FloatNDArray, **parameters: FloatNDArray | float) -> FloatNDArray:
        """
        Compute the gradient's constant term of the metric wrt gradient boost.

        Parameters
        ----------
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        **parameters : float or NDArray of shape (n_samples,)
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


class Profit(Cost):
    """
    Strategy for the Expected Profit metric.

    The profit phrasing of :class:`Cost`: it computes the same quantity and reports it negated, so
    that a higher score is a better model. Which of the two you use is a presentation choice --
    models train identically on either, because they optimize
    :meth:`~empulse.metrics.BaseMetric._loss`, which removes the sign difference.

    .. seealso::
        :class:`Cost` : The cost phrasing of the same metric.
    """

    _name: str = 'profit'
    _direction: Direction = Direction.MAXIMIZE

    def score(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float) -> float:
        """
        Compute the expected profit score.

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
            The expected profit score.
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
        # The rendered expression is linear in the four inputs, so negating all four renders the
        # negation of the cost formula, i.e. the profit formula.
        return _cost_loss_to_latex(-tp_benefit, -tn_benefit, -fp_cost, -fn_cost)


class CostLoss:
    """Class to compute the metric for binary classification."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.cost_function = _safe_lambdify(self.cost_equation)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **kwargs: Any) -> float:
        """Compute the cost loss."""
        _check_parameters(self.cost_equation.free_symbols - LABEL_AND_SCORE_SYMBOLS, kwargs)
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
        _check_parameters(self.gradient_const_eq.free_symbols - LABEL_SYMBOLS, kwargs)
        gradient_const_value = _safe_run_lambda_array(
            self.gradient_const_fn, self.gradient_const_eq, shape=y_true.shape[0], y=y_true, **kwargs
        )
        warn_if_no_training_signal(gradient_const_value, 'expected cost')
        return gradient_const_value


def _solve_optimal_threshold(
    numerator: FloatNDArray | float, denominator: FloatNDArray | float
) -> FloatNDArray | float:
    """
    Solve ``numerator / denominator`` for the optimal threshold, guarding against degenerate cases.

    A sample is targeted when its probability ``p`` satisfies ``p * denominator > numerator``. Where
    the denominator is zero, the costs do not depend on ``p``, so the sample is always targeted when
    that costs less (a negative numerator, threshold ``0``) and otherwise never (threshold ``1``),
    which includes a sample for which both decisions cost the same. A threshold outside ``[0, 1]``
    is clipped into it.

    Both an always-or-never decision implied by unequal costs and a clipped threshold warn, since
    they usually mean the cost matrix or its parameters are not what was intended.

    Parameters
    ----------
    numerator : float or ndarray
        ``fp_cost + tn_benefit``, evaluated for the given parameters.
    denominator : float or ndarray
        ``fp_cost + tn_benefit + fn_cost + tp_benefit``, evaluated for the given parameters.

    Returns
    -------
    threshold : float or ndarray
        The optimal threshold(s), each in ``[0, 1]``.
    """
    is_scalar = np.ndim(numerator) == 0 and np.ndim(denominator) == 0
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    )
    constant_cost = denominator == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        threshold = np.where(constant_cost, np.where(numerator < 0, 0.0, 1.0), numerator / denominator)

    always_or_never = constant_cost & (numerator != 0)
    if np.any(always_or_never | (threshold < 0) | (threshold > 1)):
        warnings.warn(
            'The optimal threshold computed from the cost matrix fell outside [0, 1] and was '
            'clipped. This means the cost matrix implies an always-positive or always-negative '
            'decision rule for (at least some of) the given parameters - double check the cost '
            'matrix and the parameter values passed in.',
            UserWarning,
            stacklevel=3,
        )
    threshold = np.clip(threshold, 0.0, 1.0)
    return float(threshold) if is_scalar else threshold


class CostOptimalThreshold:
    """Class to compute the optimal threshold for the cost metric."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.denominator_expression = fp_cost + tn_benefit + fn_cost + tp_benefit
        self.numerator_expression = fp_cost + tn_benefit
        self.calculate_denominator = _safe_lambdify(self.denominator_expression)
        self.calculate_numerator = _safe_lambdify(self.numerator_expression)

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: Any) -> FloatNDArray | float:
        """Compute the optimal threshold(s). `y_true` and `y_score` are unused and kept for API compatibility."""
        all_symbols = self.denominator_expression.free_symbols | self.numerator_expression.free_symbols
        _check_parameters(all_symbols, parameters)
        denominator = _safe_run_lambda(self.calculate_denominator, self.denominator_expression, **parameters)
        numerator = _safe_run_lambda(self.calculate_numerator, self.numerator_expression, **parameters)
        return _solve_optimal_threshold(numerator, denominator)


class CostOptimalRate(CostOptimalThreshold):
    """Class to compute the optimal predicted positive rate for the cost metric."""

    def __call__(self, y_true: IntNDArray, y_score: FloatNDArray, **parameters: Any) -> float:  # type: ignore[override]
        """Compute the optimal predicted positive rate."""
        return _rate_at_threshold(y_score, super().__call__(y_true, y_score, **parameters))


def _rate_at_threshold(y_score: FloatNDArray, threshold: FloatNDArray | float) -> float:
    """Return the fraction of *y_score* at or above the (per-sample) *threshold*."""
    scores = np.asarray(y_score).reshape(-1)
    return float(np.mean(scores >= np.asarray(threshold, dtype=np.float64).reshape(-1)))


def _cost_loss_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    i, N = sympy.symbols('i N')  # ruff: ignore[non-lowercase-variable-in-function]
    cost_function = (1 / N) * sympy.Sum(
        _build_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 1, N)
    )

    for symbol in cost_function.free_symbols:
        if symbol != N:
            cost_function = cost_function.subs(symbol, str(symbol) + '_i')

    output = _latex(cost_function)

    return f'$\\displaystyle {output}$'
