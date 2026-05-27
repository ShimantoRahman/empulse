from typing import Any, Literal, Self

import numpy as np
import sympy

from ...._types import FloatNDArray, IntNDArray
from ..common import (
    Direction,
    MetricFn,
    _check_parameters,
    _safe_lambdify,
    _safe_run_lambda,
)
from .cost_strategy import (
    Cost,
    _build_cost_equation,
)


class Savings(Cost):
    """Strategy for the Expected Savings metric."""

    _name: str = 'savings'
    _direction: Direction = Direction.MAXIMIZE

    @property
    def _extra_kwargs(self) -> set[str]:
        # 'baseline' is consumed by Savings.score and is not a cost-matrix symbol.
        return {'baseline'}

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
            tp_benefit=tp_benefit,
            tn_benefit=tn_benefit,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        return self

    def score(  # type: ignore[override]
        self,
        y_true: IntNDArray,
        y_score: FloatNDArray,
        baseline: Literal['zero_one', 'zero', 'one', 'prior'] | FloatNDArray = 'zero_one',
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
        if any(sympy.stats.rv.is_random(symbol) for symbol in self.cost_equation.free_symbols):
            raise NotImplementedError('Random variables are not supported for the savings metric.')
        self.all_zero_equation, self.all_one_equation = _build_naive_cost_functions(self.cost_equation)

        self.cost_func = _safe_lambdify(self.cost_equation)
        self.all_zero_function = _safe_lambdify(self.all_zero_equation)
        self.all_one_function = _safe_lambdify(self.all_one_equation)

    def __call__(
        self,
        y_true: IntNDArray,
        y_score: FloatNDArray,
        baseline: FloatNDArray | Literal['zero_one', 'zero', 'one', 'prior'],
        **kwargs: Any,
    ) -> float:
        """Compute the savings score."""
        all_symbols = (
            self.cost_equation.free_symbols | self.all_zero_equation.free_symbols | self.all_one_equation.free_symbols
        )
        _check_parameters(all_symbols - {*sympy.symbols('y s')}, kwargs)

        if isinstance(baseline, np.ndarray):
            cost_base = float(
                np.mean(_safe_run_lambda(self.cost_func, self.cost_equation, y=y_true, s=baseline, **kwargs))
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
            prior = np.mean(y_true)
            cost_base = float(
                np.mean(
                    _safe_run_lambda(
                        self.cost_func, self.cost_equation, y=y_true, s=np.full_like(y_true, prior), **kwargs
                    )
                )
            )
        else:
            raise ValueError("Invalid baseline. Must be 'zero_one', 'zero', 'one', 'prior', or an array-like.")

        if cost_base == 0.0:
            cost_base = float(np.finfo(float).eps)
        cost = _safe_run_lambda(self.cost_func, self.cost_equation, y=y_true, s=y_score, **kwargs)
        return float(1 - np.mean(cost) / cost_base)


def _build_naive_cost_functions(cost_function: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    all_zero_function = cost_function.subs('s', 0)
    all_one_function = cost_function.subs('s', 1)
    return all_zero_function, all_one_function


def _savings_score_to_latex(
    tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
) -> str:
    from sympy.printing.latex import latex

    i, N, c0, c1 = sympy.symbols('i N Cost_{0} Cost_{1}')  # noqa: N806
    savings_function = (1 / (N * sympy.Min(c0, c1))) * sympy.Sum(
        _build_cost_equation(tp_cost=-tp_benefit, tn_cost=-tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost), (i, 0, N)
    )

    for symbol in savings_function.free_symbols:
        if symbol not in {N, c0, c1}:
            savings_function = savings_function.subs(symbol, str(symbol) + '_i')

    output = latex(savings_function, mode='plain', order=None)

    return f'$\\displaystyle {output}$'
