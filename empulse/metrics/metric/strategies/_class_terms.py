"""The four class terms of a cost matrix, compiled once and evaluated at parameter values."""

from typing import Any

import sympy

from .._compile import PicklableLambda, _safe_lambdify, _safe_run_lambda


class ClassTerms:
    """
    Mixin for a strategy that evaluates the benefit and cost terms of its cost matrix.

    :meth:`_compile_class_terms` keeps the four terms and compiles them, once, when the strategy is
    built. :meth:`_evaluate_class_terms` then evaluates them as often as needed.
    """

    _tp_benefit: sympy.Expr
    _tn_benefit: sympy.Expr
    _fp_cost: sympy.Expr
    _fn_cost: sympy.Expr
    _tp_benefit_fn: PicklableLambda
    _tn_benefit_fn: PicklableLambda
    _fp_cost_fn: PicklableLambda
    _fn_cost_fn: PicklableLambda

    def _compile_class_terms(
        self, tp_benefit: sympy.Expr, tn_benefit: sympy.Expr, fp_cost: sympy.Expr, fn_cost: sympy.Expr
    ) -> None:
        self._tp_benefit, self._tn_benefit, self._fp_cost, self._fn_cost = tp_benefit, tn_benefit, fp_cost, fn_cost
        self._tp_benefit_fn = _safe_lambdify(tp_benefit)
        self._tn_benefit_fn = _safe_lambdify(tn_benefit)
        self._fp_cost_fn = _safe_lambdify(fp_cost)
        self._fn_cost_fn = _safe_lambdify(fn_cost)

    def _evaluate_class_terms(self, **parameters: Any) -> tuple[Any, Any, Any, Any]:
        """Return the values of ``tp_benefit``, ``tn_benefit``, ``fp_cost`` and ``fn_cost``, in that order."""
        return (
            _safe_run_lambda(self._tp_benefit_fn, self._tp_benefit, **parameters),
            _safe_run_lambda(self._tn_benefit_fn, self._tn_benefit, **parameters),
            _safe_run_lambda(self._fp_cost_fn, self._fp_cost, **parameters),
            _safe_run_lambda(self._fn_cost_fn, self._fn_cost, **parameters),
        )
