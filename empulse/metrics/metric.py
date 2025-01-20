from itertools import islice, pairwise
from typing import Literal

import numpy as np
import sympy
from numpy.typing import ArrayLike
from scipy.integrate import quad
from sympy import solve
from sympy.stats import density, pspace
from sympy.stats.rv import is_random
from sympy.utilities import lambdify

from ._convex_hull import _compute_convex_hull


class Metric:
    def __init__(self, kind: Literal['max profit', 'profit', 'min cost', 'cost', 'savings'] = 'max profit'):
        self.kind = kind
        self._tp_benefit = 0
        self._tn_benefit = 0
        self._fp_cost = 0
        self._fn_cost = 0
        self._aliases = {}

    @property
    def tp_benefit(self):
        return self._tp_benefit

    @property
    def tn_benefit(self):
        return self._tn_benefit

    @property
    def fp_benefit(self):
        return -self._fp_cost

    @property
    def fn_benefit(self):
        return -self._fn_cost

    @property
    def tp_cost(self):
        return -self._tp_benefit

    @property
    def tn_cost(self):
        return -self._tn_benefit

    @property
    def fp_cost(self):
        return self._fp_cost

    @property
    def fn_cost(self):
        return self._fn_cost

    def add_tp_benefit(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tp_benefit += term
        return self

    def add_tn_benefit(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tn_benefit += term
        return self

    def add_fp_benefit(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fp_cost -= term
        return self

    def add_fn_benefit(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fn_cost -= term
        return self

    def add_tp_cost(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tp_benefit -= term
        return self

    def add_tn_cost(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tn_benefit -= term
        return self

    def add_fp_cost(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fp_cost += term
        return self

    def add_fn_cost(self, term: sympy.Symbol | sympy.Expr | str) -> 'Metric':
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fn_cost += term
        return self

    def alias(self, alias: str | dict, symbol: sympy.Symbol = None) -> 'Metric':
        if isinstance(alias, dict):
            self._aliases.update(alias)
        else:
            self._aliases[alias] = symbol
        return self

    def build(self, kind: Literal['metric', 'objective'] = 'metric') -> 'Metric':
        if self.kind == 'max profit':
            self.profit_function = self._build_max_profit()
            random_symbols = [symbol for symbol in self.profit_function.free_symbols if is_random(symbol)]
            n_random = len(random_symbols)

            if n_random == 0:
                self._score_function = self._compute_deterministic(self.profit_function)
            elif n_random == 1:
                self._score_function = self._compute_one_stochastic(self.profit_function, random_symbols[0])
            else:
                raise NotImplementedError('Only zero or one random variable is supported')
        elif self.kind == 'cost':
            y, s = sympy.symbols('y s')
            cost_function = y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * (
                (1 - s) * self.tn_cost + s * self.fp_cost
            )
            cost_funct = lambdify(list(cost_function.free_symbols), cost_function)

            def cost_loss(y_true, y_score, **kwargs):
                return np.mean(cost_funct(y=y_true, s=y_score, **kwargs))

            self._score_function = cost_loss

        elif self.kind == 'savings':
            y, s = sympy.symbols('y s')
            cost_function = y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * (
                (1 - s) * self.tn_cost + s * self.fp_cost
            )
            all_zero_function = cost_function.subs(s, 0)
            all_one_function = cost_function.subs(s, 1)

            cost_func = lambdify(list(cost_function.free_symbols), cost_function)
            all_zero_func = lambdify(list(all_zero_function.free_symbols), all_zero_function)
            all_one_func = lambdify(list(all_one_function.free_symbols), all_one_function)

            def savings(y_true, y_score, **kwargs):
                # it is possible that with the substitution of the symbols, the function becomes a constant
                all_zero_score = np.mean(all_zero_func(y=y_true, **kwargs)) if all_zero_function != 0 else 0
                all_one_score = np.mean(all_one_func(y=y_true, **kwargs)) if all_one_function != 0 else 0
                cost_base = min(all_zero_score, all_one_score)
                return 1 - np.mean(cost_func(y=y_true, s=y_score, **kwargs)) / cost_base

            self._score_function = savings
        else:
            raise NotImplementedError(f'Kind {self.kind} is not supported')
        return self

    def __call__(self, y_true: ArrayLike, y_score: ArrayLike, **kwargs) -> float:
        # Map aliases to the appropriate symbols
        for alias, symbol in self._aliases.items():
            if alias in kwargs:
                kwargs[symbol] = kwargs.pop(alias)
        return self._score_function(y_true, y_score, **kwargs)

    def _compute_deterministic(self, profit_function):
        calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

        def score_function(y_true: ArrayLike, y_score: ArrayLike, **kwargs) -> float:
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)

            pi0 = float(np.mean(y_true))
            pi1 = 1 - pi0

            tprs, fprs = _compute_convex_hull(y_true, y_score)

            profits = np.zeros_like(tprs)
            for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
                profits[i] = calculate_profit(pi_0=pi0, pi_1=pi1, F_0=tpr, F_1=fpr, **kwargs)

            return profits.max()

        return score_function

    def _compute_one_stochastic(self, profit_function, random_symbol):
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        bound_eq = solve(profit_function - profit_prime, random_symbol)[0]

        compute_bounds = lambdify(list(bound_eq.free_symbols), bound_eq)
        random_var_bounds = pspace(random_symbol).domain.set.args
        distribution_args = pspace(random_symbol).distribution.args

        integrand = profit_function * density(random_symbol).pdf(random_symbol)

        def compute_integral(integrand, lower_bound, upper_bound, tpr, fpr, random_var):
            integrand = integrand.subs('F_0', tpr).subs('F_1', fpr).evalf()
            integrand_func = lambdify(random_var, integrand)
            result, _ = quad(integrand_func, lower_bound, upper_bound)
            return result

        def score_function(y_true: ArrayLike, y_score: ArrayLike, **kwargs) -> float:
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)

            pi0 = float(np.mean(y_true))
            pi1 = 1 - pi0

            f0, f1 = _compute_convex_hull(y_true, y_score)

            dist_vals = {str(key): kwargs.pop(str(key)) for key in distribution_args}
            bounds = []
            for (tpr0, fpr0), (tpr1, fpr1) in islice(pairwise(zip(f0, f1)), len(f0) - 2):
                bounds.append(compute_bounds(F_0=tpr0, F_1=fpr0, F_2=tpr1, F_3=fpr1, pi_0=pi0, pi_1=pi1, **kwargs))
            bounds.append(random_var_bounds[1])
            bounds.insert(0, random_var_bounds[0])

            integrand_ = integrand.subs(kwargs).subs(dist_vals).subs('pi_0', pi0).subs('pi_1', pi1)
            score = 0
            for (lower_bound, upper_bound), tpr, fpr in zip(pairwise(bounds), f0, f1):
                score += compute_integral(integrand_, lower_bound, upper_bound, tpr, fpr, random_symbol)
            return score

        return score_function

    def _build_max_profit(self):
        pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
        profit_function = (
            self._tp_benefit * pos_prior * tpr
            + self._tn_benefit * neg_prior * (1 - fpr)
            - self._fn_cost * pos_prior * (1 - tpr)
            - neg_prior * self.fp_cost * fpr
        )
        return profit_function

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(tp_benefit={self.tp_benefit}, '
            f'tn_benefit={self.tn_benefit}, fp_cost={self.fp_cost}, '
            f'fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self):
        from sympy.printing.latex import latex

        if self.kind == 'max profit':
            profit_function = self._build_max_profit()
            random_symbols = [symbol for symbol in profit_function.free_symbols if is_random(symbol)]

            if random_symbols:
                random_symbol = random_symbols[0]
                h = sympy.Function('h')(random_symbol)
                integrand = profit_function * h
                lower_bound, upper_bound = pspace(random_symbol).domain.set.args[:2]
                integral = sympy.Integral(integrand, (random_symbol, lower_bound, upper_bound))
                s = latex(integral, mode='plain', order=None)
            else:
                s = latex(profit_function, mode='plain', order=None)

            s = s.replace('F_{0}', 'F_{0}(T)').replace('F_{1}', 'F_{1}(T)')
            return '$\\displaystyle %s$' % s
        elif self.kind == 'cost':
            y, s, i, N = sympy.symbols('y s i N')
            cost_function = (1 / N) * sympy.Sum(
                y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * ((1 - s) * self.tn_cost + s * self.fp_cost),
                (i, 0, N),
            )

            for symbol in cost_function.free_symbols:
                if symbol != N:
                    cost_function = cost_function.subs(symbol, str(symbol) + '_i')

            s = latex(cost_function, mode='plain', order=None)
            return '$\\displaystyle %s$' % s
        elif self.kind == 'savings':
            y, s, i, N, c0, c1 = sympy.symbols('y s i N Cost_{0} Cost_{1}')
            cost_function = (1 / (N * sympy.Min(c0, c1))) * sympy.Sum(
                y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * ((1 - s) * self.tn_cost + s * self.fp_cost),
                (i, 0, N),
            )

            for symbol in cost_function.free_symbols:
                if symbol not in (N, c0, c1):
                    cost_function = cost_function.subs(symbol, str(symbol) + '_i')

            s = latex(cost_function, mode='plain', order=None)
            return '$\\displaystyle %s$' % s
