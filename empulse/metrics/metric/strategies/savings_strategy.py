import warnings
from functools import cached_property
from typing import Any, Literal, Self

import numpy as np
import sympy
from sklearn.exceptions import UndefinedMetricWarning

from ...._types import FloatArrayLike, FloatNDArray, IntNDArray
from .._compile import MetricFn, _safe_lambdify, _safe_run_lambda
from .._direction import Direction
from .._parameter_domain import LABEL_AND_SCORE_SYMBOLS, _check_parameters
from .._symbolic import _latex
from .cost_strategy import (
    Cost,
    _build_cost_equation,
)


class Savings(Cost):
    """Strategy for the Expected Savings metric."""

    _name: str = 'savings'
    _direction: Direction = Direction.MAXIMIZE
    _label_dependent_cost_weight = True

    @property
    def _extra_kwargs(self) -> set[str]:
        # 'baseline' is consumed by Savings.score and is not a cost-matrix symbol.
        return {'baseline'}

    def _cost_weight(
        self,
        y_true: IntNDArray | FloatNDArray,
        baseline: Literal['zero_one', 'zero', 'one', 'prior'] | FloatArrayLike = 'zero_one',
        **parameters: Any,
    ) -> float:
        """
        Return one over the magnitude of the baseline's cost, which the savings divide the cost by.

        Infinite when the baseline costs nothing, for then the savings are undefined.
        """
        cost_base = self._score_function.baseline_cost(y_true, baseline, **parameters)  # type: ignore[attr-defined]
        return float('inf') if cost_base == 0.0 else 1.0 / abs(cost_base)

    def build(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> Self:
        """Build the metric strategy."""
        super().build(tp_benefit, tn_benefit, fp_cost, fn_cost)
        self._score_function: MetricFn = SavingsScore(  # type: ignore[assignment]
            tp_benefit=self._tp_benefit,
            tn_benefit=self._tn_benefit,
            fp_cost=self._fp_cost,
            fn_cost=self._fn_cost,
        )
        return self

    def score(  # type: ignore[override]
        self,
        y_true: IntNDArray,
        y_score: FloatNDArray,
        baseline: Literal['zero_one', 'zero', 'one', 'prior'] | FloatArrayLike = 'zero_one',
        **parameters: FloatNDArray | float,
    ) -> float:
        """
        Compute the metric expected savings score.

        .. note::
            This method extends the :meth:`MetricStrategy.score` interface with an
            additional ``baseline`` argument (defaulting to ``'zero_one'``).
            Callers using the abstract :class:`MetricStrategy` interface will always
            receive the default baseline behaviour, which is consistent and safe.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground truth labels.

        y_score : array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

        baseline : {'zero_one', 'zero', 'one', 'prior'}, float or array-like of shape (n_samples,), default='zero_one'
            The model that savings are measured against; the score is
            ``(cost_baseline - cost) / |cost_baseline|``.

            - If ``'zero_one'``, whichever of "predict all zeros" or "predict all ones" costs less.
            - If ``'zero'``, predict all zeros.
            - If ``'one'``, predict all ones.
            - If ``'prior'``, predict the positive class prior for every sample.
            - If ``float`` or ``array-like``, the predicted labels or calibrated probabilities of a reference model.

            If the baseline costs nothing, the savings are undefined and ``nan`` is returned.

        **parameters : float or array-like of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        score : float
            The expected savings score.
        """
        return self._score_function(y_true, y_score, baseline=baseline, **parameters)

    def to_latex(
        self,
        tp_benefit: sympy.Expr,
        tn_benefit: sympy.Expr,
        fp_cost: sympy.Expr,
        fn_cost: sympy.Expr,
    ) -> str:
        """Return the LaTeX representation of the metric."""
        return _savings_score_to_latex(tp_benefit, tn_benefit, fp_cost, fn_cost)


class SavingsScore:
    """Class to compute the metric for binary classification."""

    def __init__(self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr):
        self.cost_equation = _build_cost_equation(
            tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
        )
        self.all_zero_equation, self.all_one_equation = _build_naive_cost_functions(self.cost_equation)

        self.cost_func = _safe_lambdify(self.cost_equation)
        self.all_zero_function = _safe_lambdify(self.all_zero_equation)
        self.all_one_function = _safe_lambdify(self.all_one_equation)

    @cached_property
    def _parameter_symbols(self) -> set[sympy.Symbol]:
        equations = (self.cost_equation, self.all_zero_equation, self.all_one_equation)
        return set().union(*(equation.free_symbols for equation in equations)) - LABEL_AND_SCORE_SYMBOLS

    def __call__(
        self,
        y_true: IntNDArray,
        y_score: FloatNDArray,
        baseline: FloatArrayLike | Literal['zero_one', 'zero', 'one', 'prior'],
        **kwargs: Any,
    ) -> float:
        """Compute the savings score."""
        cost_base = self.baseline_cost(y_true, baseline, **kwargs)
        cost = float(np.mean(_safe_run_lambda(self.cost_func, self.cost_equation, y=y_true, s=y_score, **kwargs)))
        return _relative_savings(cost, cost_base)

    def baseline_cost(
        self,
        y_true: IntNDArray | FloatNDArray,
        baseline: FloatArrayLike | Literal['zero_one', 'zero', 'one', 'prior'],
        **kwargs: Any,
    ) -> float:
        """Compute the average cost of the *baseline* classifier, which the savings are relative to."""
        _check_parameters(self._parameter_symbols, kwargs)

        if not isinstance(baseline, str):
            baseline_score = np.broadcast_to(np.asarray(baseline, dtype=np.float64), y_true.shape)
            cost_base = float(
                np.mean(_safe_run_lambda(self.cost_func, self.cost_equation, y=y_true, s=baseline_score, **kwargs))
            )
        elif baseline == 'zero_one':
            all_zero_score = float(
                np.mean(_safe_run_lambda(self.all_zero_function, self.all_zero_equation, y=y_true, **kwargs))
            )
            all_one_score = float(
                np.mean(_safe_run_lambda(self.all_one_function, self.all_one_equation, y=y_true, **kwargs))
            )
            cost_base = min(all_zero_score, all_one_score)
        elif baseline == 'zero':
            cost_base = float(
                np.mean(_safe_run_lambda(self.all_zero_function, self.all_zero_equation, y=y_true, **kwargs))
            )
        elif baseline == 'one':
            cost_base = float(
                np.mean(_safe_run_lambda(self.all_one_function, self.all_one_equation, y=y_true, **kwargs))
            )
        elif baseline == 'prior':
            prior = np.full(y_true.shape, np.mean(y_true), dtype=np.float64)
            cost_base = float(
                np.mean(_safe_run_lambda(self.cost_func, self.cost_equation, y=y_true, s=prior, **kwargs))
            )
        else:
            raise ValueError("Invalid baseline. Must be 'zero_one', 'zero', 'one', 'prior', or an array-like.")
        return cost_base


def _relative_savings(cost: float, cost_base: float) -> float:
    """
    Return the cost saved relative to the baseline's cost.

    Dividing by the magnitude keeps a lower cost scoring higher when the baseline cost is negative,
    as it is for a matrix of benefits.
    """
    if cost_base == 0.0:
        warnings.warn(
            'The baseline has zero cost, so the savings are undefined. Returning nan.',
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float('nan')
    return float((cost_base - cost) / abs(cost_base))


def _build_naive_cost_functions(cost_function: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    all_zero_function = cost_function.subs('s', 0)
    all_one_function = cost_function.subs('s', 1)
    return all_zero_function, all_one_function


def _savings_score_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    i, N, c0, c1 = sympy.symbols('i N Cost_{0} Cost_{1}')  # ruff: ignore[non-lowercase-variable-in-function]
    baseline_cost = sympy.Min(c0, c1)
    cost = (1 / N) * sympy.Sum(
        _build_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 1, N)
    )
    savings_function = (baseline_cost - cost) / sympy.Abs(baseline_cost)

    for symbol in savings_function.free_symbols:
        if symbol not in {N, c0, c1}:
            savings_function = savings_function.subs(symbol, str(symbol) + '_i')

    output = _latex(savings_function)

    return f'$\\displaystyle {output}$'
