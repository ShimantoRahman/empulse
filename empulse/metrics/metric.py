from itertools import islice, pairwise
from numbers import Real
from typing import Any, ClassVar, Literal

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
    """
    Class to create a custom value/cost-sensitive metric.

    The Metric class uses the Builder pattern to create a custom metric.
    You start by specifying the kind of metric you want to compute and then add the terms that make up the metric.
    These terms come from the cost-benefits matrix of the classification problem.
    After you have added all the terms, you can call the :meth:`~empulse.metrics.Metric.build`
    method to create the metric function.
    Then you can call the metric function with the true labels and predicted probabilities to compute the metric value.

    The costs and benefits are specified using sympy symbols or expressions.
    Stochastic variables are supported, and can be specified using sympy.stats random variables.
    Make sure that you add the parameters of the random variables as keyword arguments when calling the metric function.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    Parameters
    ----------
    kind : {'max profit', 'cost', 'savings'}
        The kind of metric to compute.

    Attributes
    ----------
    tp_benefit : sympy.Expr
        The benefit of a true positive.

    tn_benefit : sympy.Expr
        The benefit of a true negative.

    fp_benefit : sympy.Expr
        The benefit of a false positive.

    fn_benefit : sympy.Expr
        The benefit of a false negative.

    tp_cost : sympy.Expr
        The cost of a true positive.

    tn_cost : sympy.Expr
        The cost of a true negative.

    fp_cost : sympy.Expr
        The cost of a false positive.

    fn_cost : sympy.Expr
        The cost of a false negative.

    Examples
    --------
    Reimplementing :func:`~empulse.metrics.empc_score` using the :class:`Metric` class.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import Metric

        clv, d, f, alpha, beta = sp.symbols(
            'clv d f alpha beta'
        )  # define deterministic variables
        gamma = sp.stats.Beta('gamma', alpha, beta)  # define gamma to follow a Beta distribution

        empc_score = (
            Metric(kind='max profit')
            .add_tp_benefit(gamma * (clv - d - f))  # when churner accepts offer
            .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
            .add_fp_cost(d + f)  # when you send an offer to a non-churner
            .alias({'incentive_cost': 'd', 'contact_cost': 'f'})
            .build()
        )

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]

        empc_score(y_true, y_proba, clv=100, incentive_cost=10, contact_cost=1, alpha=6, beta=14)

    Reimplementing :func:`~empulse.metrics.expected_cost_loss_churn` using the :class:`Metric` class.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import Metric

        clv, delta, f, gamma = sp.symbols('clv delta f gamma')

        cost_loss = (
            Metric(kind='cost')
            .add_tp_benefit(gamma * (clv - delta * clv - f))  # when churner accepts offer
            .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
            .add_fp_cost(delta * clv + f)  # when you send an offer to a non-churner
            .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
            .build()
        )

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]

        cost_loss(
            y_true, y_proba, clv=100, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
        )
    """

    METRIC_TYPES: ClassVar[list[str]] = ['max profit', 'cost', 'savings']

    def __init__(self, kind: Literal['max profit', 'cost', 'savings']) -> None:
        if kind not in self.METRIC_TYPES:
            raise ValueError(f'Kind {kind} is not supported. Supported values are {self.METRIC_TYPES}')
        self.kind = kind
        self._tp_benefit = sympy.core.numbers.Zero()
        self._tn_benefit = sympy.core.numbers.Zero()
        self._fp_cost = sympy.core.numbers.Zero()
        self._fn_cost = sympy.core.numbers.Zero()
        self._aliases: dict[str, str | sympy.Symbol] = {}
        self._defaults: dict[str, Any] = {}
        self._built = False

    @property
    def tp_benefit(self) -> sympy.Expr:  # noqa: D102
        return self._tp_benefit

    @property
    def tn_benefit(self) -> sympy.Expr:  # noqa: D102
        return self._tn_benefit

    @property
    def fp_benefit(self) -> sympy.Expr:  # noqa: D102
        return -self._fp_cost

    @property
    def fn_benefit(self) -> sympy.Expr:  # noqa: D102
        return -self._fn_cost

    @property
    def tp_cost(self) -> sympy.Expr:  # noqa: D102
        return -self._tp_benefit

    @property
    def tn_cost(self) -> sympy.Expr:  # noqa: D102
        return -self._tn_benefit

    @property
    def fp_cost(self) -> sympy.Expr:  # noqa: D102
        return self._fp_cost

    @property
    def fn_cost(self):  # noqa: D102
        return self._fn_cost

    def add_tp_benefit(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the benefit of classifying a true positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a true positive.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tp_benefit += term
        return self

    def add_tn_benefit(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the benefit of classifying a true negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a true negative.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tn_benefit += term
        return self

    def add_fp_benefit(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the benefit of classifying a false positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a false positive.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fp_cost -= term
        return self

    def add_fn_benefit(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the benefit of classifying a false negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the benefit of classifying a false negative.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fn_cost -= term
        return self

    def add_tp_cost(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the cost of classifying a true positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a true positive.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tp_benefit -= term
        return self

    def add_tn_cost(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the cost of classifying a true negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a true negative.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._tn_benefit -= term
        return self

    def add_fp_cost(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the cost of classifying a false positive.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a false positive.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fp_cost += term
        return self

    def add_fn_cost(self, term: sympy.Expr | str) -> 'Metric':
        """
        Add a term to the cost of classifying a false negative.

        Parameters
        ----------
        term: sympy.Expr | str
            The term to add to the cost of classifying a false negative.

        Returns
        -------
        Metric
        """
        if isinstance(term, str):
            term = sympy.sympify(term)
        self._fn_cost += term
        return self

    def alias(self, alias: str | dict[str, sympy.Symbol | str], symbol: sympy.Symbol | None = None) -> 'Metric':
        """
        Add an alias for a symbol.

        Parameters
        ----------
        alias: str | dict[str, sympy.Symbol | str]
            The alias to add. If a dictionary is passed, the keys are the aliases and the values are the symbols.
        symbol: sympy.Symbol, optional
            The symbol to alias to.

        Returns
        -------
        Metric

        Examples
        --------

        .. code-block:: python

            import sympy as sp
            from empulse.metrics import Metric

            clv, delta, f, gamma = sp.symbols('clv delta f gamma')
            cost_loss = (
                Metric(kind='cost')
                .add_tp_benefit(gamma * (clv - delta * clv - f))  # when churner accepts offer
                .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
                .add_fp_cost(delta * clv + f)  # when you send an offer to a non-churner
                .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
                .build()
            )

            y_true = [1, 0, 1, 0, 1]
            y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
            cost_loss(
                y_true, y_proba, clv=100, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
            )
        """
        if isinstance(alias, dict):
            self._aliases.update(alias)
        elif symbol is not None:
            self._aliases[alias] = symbol
        else:
            raise ValueError('Either a dictionary or both an alias and a symbol should be provided')
        return self

    def set_default(self, **defaults: float) -> 'Metric':
        """
        Set default values for symbols or their aliases.

        Parameters
        ----------
        defaults: float
            Default values for symbols or their aliases.
            These default values will be used if not provided in __call__.

        Returns
        -------
        Metric

        Examples
        --------

        .. code-block:: python

            import sympy as sp
            from empulse.metrics import Metric

            clv, delta, f, gamma = sp.symbols('clv delta f gamma')
            cost_loss = (
                Metric(kind='cost')
                .add_tp_benefit(gamma * (clv - delta * clv - f))  # when churner accepts offer
                .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
                .add_fp_cost(delta * clv + f)  # when you send an offer to a non-churner
                .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
                .set_default(incentive_fraction=0.05, contact_cost=1, accept_rate=0.3)
                .build()
            )

            y_true = [1, 0, 1, 0, 1]
            y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]
            cost_loss(y_true, y_proba, clv=100, incentive_fraction=0.1)

        """
        self._defaults.update(defaults)
        return self

    def build(self) -> 'Metric':
        """
        Build the metric function.

        This function should be called last after adding all the terms.
        After calling this function, the metric function can be called with the true labels and predicted probabilities.

        Returns
        -------
        Metric
        """
        self._built = True
        terms = self.tp_cost + self.tn_cost + self.fp_cost + self.fn_cost
        random_symbols = [symbol for symbol in terms.free_symbols if is_random(symbol)]
        n_random = len(random_symbols)

        if self.kind in {'cost', 'savings'} and n_random > 0:
            raise NotImplementedError('Random variables are not supported for cost and savings metrics')

        if self.kind == 'max profit':
            self.profit_function = self._build_max_profit()
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

    def __call__(self, y_true: ArrayLike, y_score: ArrayLike, **kwargs: ArrayLike | float) -> float:
        """
        Compute the metric score or loss.

        The :meth:`empulse.metrics.Metric.build` method should be called before calling this method.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities or decision scores (based on the chosen metric).

        kwargs: float or array-like of shape (n_samples,)
            The values of the costs and benefits defined in the metric.
            Can either be their symbols or their aliases.

        Returns
        -------
        score: float
            The computed metric score or loss.
        """
        if not self._built:
            raise ValueError('The metric function has not been built. Call the build method before calling the metric')

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Use default values if not provided in kwargs
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        for key, value in kwargs.items():
            if not isinstance(value, Real):
                kwargs[key] = np.asarray(value)

        # Map aliases to the appropriate symbols
        for alias, symbol in self._aliases.items():
            if alias in kwargs:
                kwargs[symbol] = kwargs.pop(alias)

        return self._score_function(y_true, y_score, **kwargs)

    def _compute_deterministic(self, profit_function):
        calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

        def score_function(y_true: ArrayLike, y_score: ArrayLike, **kwargs: Any) -> float:
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)

            pi0 = float(np.mean(y_true))
            pi1 = 1 - pi0

            tprs, fprs = _compute_convex_hull(y_true, y_score)

            profits = np.zeros_like(tprs)
            for i, (tpr, fpr) in enumerate(zip(tprs, fprs, strict=False)):
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
            if not integrand.free_symbols:  # if the integrand is constant
                if integrand == 0:
                    return 0
                return float(integrand * (upper_bound - lower_bound))
            integrand_func = lambdify(random_var, integrand)
            result, _ = quad(integrand_func, lower_bound, upper_bound)
            return result

        def score_function(y_true: ArrayLike, y_score: ArrayLike, **kwargs: Any) -> float:
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)

            pi0 = float(np.mean(y_true))
            pi1 = 1 - pi0

            f0, f1 = _compute_convex_hull(y_true, y_score)

            dist_vals = {str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs}
            bounds = []
            for (tpr0, fpr0), (tpr1, fpr1) in islice(pairwise(zip(f0, f1, strict=False)), len(f0) - 2):
                bounds.append(compute_bounds(F_0=tpr0, F_1=fpr0, F_2=tpr1, F_3=fpr1, pi_0=pi0, pi_1=pi1, **kwargs))
            if isinstance(upper_bound := random_var_bounds[1], sympy.Symbol | sympy.Expr):
                upper_bound = upper_bound.subs(dist_vals)
            bounds.append(upper_bound)
            if isinstance(lower_bound := random_var_bounds[0], sympy.Symbol | sympy.Expr):
                lower_bound = lower_bound.subs(dist_vals)
            bounds.insert(0, lower_bound)

            integrand_ = integrand.subs(kwargs).subs(dist_vals).subs('pi_0', pi0).subs('pi_1', pi1)
            score = 0
            for (lower_bound, upper_bound), tpr, fpr in zip(pairwise(bounds), f0, f1, strict=False):
                score += compute_integral(integrand_, lower_bound, upper_bound, tpr, fpr, random_symbol)
            return score

        return score_function

    def _build_max_profit(self) -> sympy.Expr:
        pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
        profit_function = (
            self._tp_benefit * pos_prior * tpr
            + self._tn_benefit * neg_prior * (1 - fpr)
            - self._fn_cost * pos_prior * (1 - tpr)
            - neg_prior * self.fp_cost * fpr
        )
        return profit_function

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(tp_benefit={self.tp_benefit}, '
            f'tn_benefit={self.tn_benefit}, fp_cost={self.fp_cost}, '
            f'fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self) -> str:
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
            return f'$\\displaystyle {s}$'
        elif self.kind == 'cost':
            y, s, i, N = sympy.symbols('y s i N')  # noqa: N806
            cost_function = (1 / N) * sympy.Sum(
                y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * ((1 - s) * self.tn_cost + s * self.fp_cost),
                (i, 0, N),
            )

            for symbol in cost_function.free_symbols:
                if symbol != N:
                    cost_function = cost_function.subs(symbol, str(symbol) + '_i')

            s = latex(cost_function, mode='plain', order=None)
            return f'$\\displaystyle {s}$'
        elif self.kind == 'savings':
            y, s, i, N, c0, c1 = sympy.symbols('y s i N Cost_{0} Cost_{1}')  # noqa: N806
            cost_function = (1 / (N * sympy.Min(c0, c1))) * sympy.Sum(
                y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * ((1 - s) * self.tn_cost + s * self.fp_cost),
                (i, 0, N),
            )

            for symbol in cost_function.free_symbols:
                if symbol not in {N, c0, c1}:
                    cost_function = cost_function.subs(symbol, str(symbol) + '_i')

            s = latex(cost_function, mode='plain', order=None)
            return f'$\\displaystyle {s}$'
