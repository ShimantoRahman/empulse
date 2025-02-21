from collections.abc import Callable
from itertools import islice, pairwise
from numbers import Real
from typing import Any, ClassVar, Final, Literal, Protocol

import numpy as np
import scipy
import sympy
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import dblquad, nquad, quad, tplquad
from scipy.stats._qmc import Sobol
from sympy import solve
from sympy.stats import density, pspace
from sympy.stats.rv import is_random
from sympy.utilities import lambdify

from ._convex_hull import _compute_convex_hull


class MetricFn(Protocol):  # noqa: D101
    def __call__(self, y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float: ...  # noqa: D102


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
    Make sure that you add the parameters of the stochastic variables
    as keyword arguments when calling the metric function.
    Stochastic variables are assumed to be independent of each other.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    Parameters
    ----------
    kind : {'max profit', 'cost', 'savings'}
        The kind of metric to compute.

        - If ``max profit``, the metric computes the maximum profit that can be achieved by a classifier.
          The metric determines the optimal threshold that maximizes the profit.
          This metric supports the use of stochastic variables.
        - If ``'cost'``, the metric computes the expected cost loss of a classifier.
          This metric supports passing instance-dependent costs in the form of array-likes.
          This metric does not support stochastic variables.
        - If ``'savings'``, the metric computes the savings that can be achieved by a classifier
          over a naive classifier which always predicts 0 or 1 (whichever is better).
          This metric supports passing instance-dependent costs in the form of array-likes.
          This metric does not support stochastic variables.

    integration_method : {'auto', 'quad', 'monte-carlo'}, default='auto'
        The integration method to use when the metric has stochastic variables.

        - If ``'auto'``, the integration method is automatically chosen based on the number of stochastic variables,
          balancing accuracy with execution speed.
          For a single stochastic variables, piecewise integration is used. This is the most accurate method.
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

            - :class:`sympy:sympy.stats.Normal`
            - :class:`sympy:sympy.stats.Beta`
            - :class:`~sympy:sympy.stats.ChiSquared`
            - :class:`~sympy:sympy.stats.Exponential`
            - :class:`~sympy:sympy.stats.Gamma`
            - :class:`~sympy:sympy.stats.Laplace`
            - :class:`~sympy:sympy.stats.LogNormal`

    n_mc_samples_exp : int, default=16 (-> 2**16 = 65536)
        ``2**n_mc_samples_exp`` is the number of (Quasi-) Monte Carlo samples to use when
        ``integration_technique'monte-carlo'``.
        Increasing the number of samples improves the accuracy of the metric estimation, but slows down the speed.
        This argument is ignored when the ``integration_technique='quad'``.

    random_state : int | np.random.RandomState | None, default=None
        The random state to use when ``integration_technique='monte-carlo'`` or
        ``integration_technique='quasi-monte-carlo'``.
        Determines the points sampled from the distribution of the stochastic variables.
        This argument is ignored when ``integration_technique='quad'``.

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
    INTEGRATION_METHODS: ClassVar[list[str]] = ['auto', 'quad', 'quasi-monte-carlo', 'monte-carlo']
    _sympy_dist_to_scipy: ClassVar[
        dict[
            sympy.stats.crv_types.SingleContinuousDistribution | sympy.stats.drv_types.SingleDiscreteDistribution,
            scipy.stats.rv_continuous | scipy.stats.rv_discrete,
        ]
    ] = {
        sympy.stats.crv_types.NormalDistribution: scipy.stats.norm,
        sympy.stats.crv_types.BetaDistribution: scipy.stats.beta,
        # sympy.stats.drv_types.NegativeBinomialDistribution: scipy.stats.binom,
        sympy.stats.crv_types.ChiSquaredDistribution: scipy.stats.chi2,
        sympy.stats.crv_types.ExponentialDistribution: scipy.stats.expon,
        sympy.stats.crv_types.GammaDistribution: scipy.stats.gamma,
        sympy.stats.crv_types.LaplaceDistribution: scipy.stats.laplace,
        sympy.stats.crv_types.LogNormalDistribution: scipy.stats.lognorm,
        sympy.stats.drv_types.PoissonDistribution: scipy.stats.poisson,
        sympy.stats.crv_types.UniformDistribution: scipy.stats.uniform,
    }

    def __init__(
        self,
        kind: Literal['max profit', 'cost', 'savings'],
        *,
        integration_method: Literal['auto', 'quad', 'quasi-monte-carlo', 'monte-carlo'] = 'auto',
        n_mc_samples_exp: int = 16,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if kind not in self.METRIC_TYPES:
            raise ValueError(f'Kind {kind} is not supported. Supported values are {self.METRIC_TYPES}')
        if integration_method not in self.INTEGRATION_METHODS:
            raise ValueError(
                f'Integration technique {integration_method} is not supported. '
                f'Supported values are {self.INTEGRATION_METHODS}'
            )
        self.kind = kind
        self.integration_method = integration_method
        self.n_mc_samples: Final[int] = 2**n_mc_samples_exp
        self._rng = np.random.RandomState(random_state)
        self._tp_benefit: sympy.Expr = sympy.core.numbers.Zero()
        self._tn_benefit: sympy.Expr = sympy.core.numbers.Zero()
        self._fp_cost: sympy.Expr = sympy.core.numbers.Zero()
        self._fn_cost: sympy.Expr = sympy.core.numbers.Zero()
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
        deterministic_symbols = [symbol for symbol in terms.free_symbols if not is_random(symbol)]
        n_random = len(random_symbols)

        if self.kind in {'cost', 'savings'} and n_random > 0:
            raise NotImplementedError('Random variables are not supported for cost and savings metrics')

        if self.kind == 'max profit':
            self.profit_function = self._build_profit_function()
            if n_random == 0:
                self._score_function = self._build_max_profit_deterministic(self.profit_function, deterministic_symbols)
            else:
                self._score_function = self._build_max_profit_stochastic(
                    self.profit_function, random_symbols, deterministic_symbols
                )
        elif self.kind == 'cost':
            self._score_function = self._build_cost_loss()

        elif self.kind == 'savings':
            self._score_function = self._build_savings_score()
        else:
            raise NotImplementedError(f'Kind {self.kind} is not supported')
        return self

    def _build_max_profit_deterministic(self, profit_function, deterministic_symbols):
        """Compute the maximum profit for all deterministic variables."""
        calculate_profit = lambdify(list(profit_function.free_symbols), profit_function)

        @_check_parameters(*deterministic_symbols)
        def score_function(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            pi0 = float(np.mean(y_true))
            pi1 = 1 - pi0
            tprs, fprs = _compute_convex_hull(y_true, y_score)

            profits = np.zeros_like(tprs)
            for i, (tpr, fpr) in enumerate(zip(tprs, fprs, strict=False)):
                profits[i] = calculate_profit(pi_0=pi0, pi_1=pi1, F_0=tpr, F_1=fpr, **kwargs)

            return float(profits.max())

        return score_function

    def _support_all_distributions(self, random_symbols):
        return all(pspace(r).distribution.__class__ in self._sympy_dist_to_scipy for r in random_symbols)

    def _build_max_profit_stochastic(self, profit_function, random_symbols, deterministic_symbols):
        """Compute the maximum profit for one or more stochastic variables."""
        n_random = len(random_symbols)
        if self.integration_method == 'auto':
            if n_random == 1:
                return self._build_max_profit_stochastic_piecewise(
                    self.profit_function, random_symbols[0], deterministic_symbols
                )
            elif n_random == 2:
                return self._build_max_profit_stochastic_quad(profit_function, random_symbols, deterministic_symbols)
            elif self._support_all_distributions(random_symbols):
                return self._build_max_profit_stochastic_qmc(profit_function, random_symbols, deterministic_symbols)
            else:
                return self._build_max_profit_stochastic_mc(profit_function, random_symbols, deterministic_symbols)
        elif self.integration_method == 'quad':
            return self._build_max_profit_stochastic_quad(profit_function, random_symbols, deterministic_symbols)
        elif self.integration_method == 'monte-carlo':
            return self._build_max_profit_stochastic_mc(profit_function, random_symbols, deterministic_symbols)
        elif self.integration_method == 'quasi-monte-carlo':
            return self._build_max_profit_stochastic_qmc(profit_function, random_symbols, deterministic_symbols)
        else:
            raise ValueError(f'Integration method {self.integration_method} is not supported')

    def _build_max_profit_stochastic_piecewise(self, profit_function, random_symbol, deterministic_symbols):
        """
        Compute the maximum profit for a single stochastic variable using piecewise integration.

        For each convex hull segment, the bounds of the random variable are computed
        in which that decision threshold is optimal.
        For each segment, the profit is integrated over the bounds of the random variable.
        """
        profit_prime = profit_function.subs('F_0', 'F_2').subs('F_1', 'F_3')
        bound_eq = solve(profit_function - profit_prime, random_symbol)[0]
        compute_bounds = lambdify(list(bound_eq.free_symbols), bound_eq)

        random_var_bounds = pspace(random_symbol).domain.set.args
        distribution_args = pspace(random_symbol).distribution.args

        integrand = profit_function * density(random_symbol).pdf(random_symbol)

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
            dist_params = []
        else:
            dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

        def compute_integral(integrand, lower_bound, upper_bound, true_positive_rate, false_positive_rate, random_var):
            integrand = integrand.subs('F_0', true_positive_rate).subs('F_1', false_positive_rate).evalf()
            if not integrand.free_symbols:  # if the integrand is constant, no need to call quad
                if integrand == 0:  # need this separate path since sometimes upper or lower bound can be infinite
                    return 0
                return float(integrand * (upper_bound - lower_bound))
            integrand_fn = lambdify(random_var, integrand)
            result, _ = quad(integrand_fn, lower_bound, upper_bound)
            return result

        @_check_parameters(*deterministic_symbols, *dist_params)
        def score_function(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            positive_class_prior = float(np.mean(y_true))
            negative_class_prior = 1 - positive_class_prior
            true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

            distribution_parameters = {  # distribution parameters of the random variable
                str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
            }
            bounds = []
            for (tpr0, fpr0), (tpr1, fpr1) in islice(
                pairwise(zip(true_positive_rates, false_positive_rates, strict=False)), len(true_positive_rates) - 2
            ):
                bounds.append(
                    compute_bounds(
                        F_0=tpr0,
                        F_1=fpr0,
                        F_2=tpr1,
                        F_3=fpr1,
                        pi_0=positive_class_prior,
                        pi_1=negative_class_prior,
                        **kwargs,
                    )
                )

            # bounds of the random variable can be parameterized by the user
            # if so substitute the parameters in the bounds with the user provided values
            if isinstance(upper_bound := random_var_bounds[1], sympy.Expr):
                upper_bound = upper_bound.subs(distribution_parameters)
            bounds.append(upper_bound)
            if isinstance(lower_bound := random_var_bounds[0], sympy.Expr):
                lower_bound = lower_bound.subs(distribution_parameters)
            bounds.insert(0, lower_bound)

            integrand_ = (
                integrand.subs(kwargs)
                .subs(distribution_parameters)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
            score = 0
            for (lower_bound, upper_bound), tpr, fpr in zip(
                pairwise(bounds), true_positive_rates, false_positive_rates, strict=False
            ):
                score += compute_integral(integrand_, lower_bound, upper_bound, tpr, fpr, random_symbol)
            return score

        return score_function

    def _build_max_profit_stochastic_quad(self, profit_function, random_symbols, deterministic_symbols):
        """
        Compute the maximum profit for one or more stochastic variables using quad integration.

        This method is very slow for more than 2 stochastic variables.
        It is recommended to use Quasi Monte Carlo integration for more than 2 stochastic variables.
        """
        n_random = len(random_symbols)
        random_variables_bounds = [pspace(random_symbol).domain.set.args for random_symbol in random_symbols]
        random_variables_bounds = [(lb, up) for (lb, up, *_) in random_variables_bounds]
        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        distribution_args = [arg for args in distributions_args for arg in args]

        integrand = profit_function
        for random_symbol in random_symbols:
            integrand *= density(random_symbol).pdf(random_symbol)

        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
            dist_params = []
        else:
            dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

        def compute_integral(integrand, bounds, true_positive_rates, false_positive_rates, random_variables):
            integrands = [
                lambdify(random_variables, integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
                for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
            ]

            def integrand_fn(*random_vars):
                return max(integrand(*reversed(random_vars)) for integrand in integrands)

            if n_random == 1:
                result, _ = quad(integrand_fn, *bounds)
            elif n_random == 2:
                result, _ = dblquad(integrand_fn, *bounds)
            elif n_random == 3:
                result, _ = tplquad(integrand_fn, *bounds)
            else:
                result, _ = nquad(integrand_fn, *bounds)
            return result

        @_check_parameters(*deterministic_symbols, *dist_params)
        def score_function(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            positive_class_prior = float(np.mean(y_true))
            negative_class_prior = 1 - positive_class_prior
            true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

            # certain distributions determine the bounds of the integral (e.g., uniform)
            # for those distributions we have to fill in the parameters of the distribution
            distribution_parameters = {
                str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
            }
            bounds = [bound for bounds in random_variables_bounds for bound in bounds]
            bounds = [
                bounds.subs(distribution_parameters) if isinstance(bounds, sympy.Expr) else bounds for bounds in bounds
            ]

            integrand_ = (
                integrand.subs(kwargs)
                .subs(distribution_parameters)
                .subs('pi_0', positive_class_prior)
                .subs('pi_1', negative_class_prior)
            )
            return compute_integral(integrand_, bounds, true_positive_rates, false_positive_rates, random_symbols)

        return score_function

    def _build_max_profit_stochastic_mc(self, profit_function, random_symbols, deterministic_symbols):
        """
        Compute the maximum profit for one or more stochastic variables using Monte Carlo (MC) integration.

        This method is less accurate than quad integration but faster for many stochastic variables.
        The QMC method is preferred over the MC due to better accuracy.
        This method should only be used if there is no mapping of sympy distributions to scipy distributions.
        """
        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        distribution_args = [arg for args in distributions_args for arg in args]
        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
            param_grid_needs_recompute = False
            param_grid = [
                sympy.stats.sample(random_var, size=(self.n_mc_samples,), seed=self._rng)
                for random_var in random_symbols
            ]
            dist_params = []
        else:
            cached_dist_params = {str(arg): arg for arg in distribution_args}
            param_grid_needs_recompute = True
            param_grid = None
            dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

        @_check_parameters(*deterministic_symbols, *dist_params)
        def score_function(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            positive_class_prior = float(np.mean(y_true))
            negative_class_prior = 1 - positive_class_prior
            true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

            nonlocal param_grid
            nonlocal param_grid_needs_recompute
            nonlocal cached_dist_params
            if param_grid_needs_recompute:
                distribution_parameters = {  # distribution parameters of the random variable
                    str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
                }
                if cached_dist_params != distribution_parameters:
                    cached_dist_params = distribution_parameters
                    param_grid = [
                        sympy.stats.sample(
                            random_var.subs(cached_dist_params), size=(self.n_mc_samples,), seed=self._rng
                        )
                        for random_var in random_symbols
                    ]

                integrand = (
                    profit_function.subs(kwargs)
                    .subs(cached_dist_params)
                    .subs('pi_0', positive_class_prior)
                    .subs('pi_1', negative_class_prior)
                )
            else:
                integrand = (
                    profit_function.subs(kwargs).subs('pi_0', positive_class_prior).subs('pi_1', negative_class_prior)
                )
            integrands = [
                lambdify(random_symbols, integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
                for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
            ]

            results = np.empty((len(integrands), self.n_mc_samples))
            for i, integrand in enumerate(integrands):
                results[i, :] = integrand(*param_grid)
            result = results.max(axis=0).mean()

            return float(result)

        return score_function

    def _build_max_profit_stochastic_qmc(self, profit_function, random_symbols, deterministic_symbols):
        distributions_args = [pspace(random_symbol).distribution.args for random_symbol in random_symbols]
        distribution_args = [arg for args in distributions_args for arg in args]
        # Generate a Sobol sequence for QMC sampling
        sobol = Sobol(d=len(random_symbols), scramble=True, seed=self._rng)
        sobol_samples = sobol.random(2**16)
        if all(isinstance(arg, sympy.core.numbers.Integer) for arg in distribution_args):
            # If all distribution parameters are fixed, then the param grid can be pre-computed.
            param_grid_needs_recompute = False
            # convert to scipy distributions
            scipy_distributions = [
                self._sympy_dist_to_scipy[pspace(random_var).distribution.__class__](
                    *[float(arg) for arg in pspace(random_var).distribution.args]
                )
                for random_var in random_symbols
            ]
            param_grid = [dist.ppf(sobol_samples[:, i]) for i, dist in enumerate(scipy_distributions)]
            dist_params = []

        else:
            cached_dist_params = {str(arg): arg for arg in distribution_args}
            scipy_distributions = None
            param_grid_needs_recompute = True
            param_grid = None
            dist_params = [arg for arg in distribution_args if not isinstance(arg, sympy.core.numbers.Integer)]

        @_check_parameters(*deterministic_symbols, *dist_params)
        def score_function(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            positive_class_prior = float(np.mean(y_true))
            negative_class_prior = 1 - positive_class_prior
            true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)

            nonlocal param_grid
            nonlocal param_grid_needs_recompute
            nonlocal cached_dist_params
            nonlocal scipy_distributions
            if param_grid_needs_recompute:
                distribution_parameters = {  # distribution parameters of the random variable
                    str(key): kwargs.pop(str(key)) for key in distribution_args if str(key) in kwargs
                }
                if cached_dist_params != distribution_parameters:
                    cached_dist_params = distribution_parameters
                    scipy_distributions = [
                        self._sympy_dist_to_scipy[pspace(random_var).distribution.__class__](
                            *[float(arg) for arg in pspace(random_var.subs(cached_dist_params)).distribution.args]
                        )
                        for random_var in random_symbols
                    ]
                    param_grid = [dist.ppf(sobol_samples[:, i]) for i, dist in enumerate(scipy_distributions)]

                integrand = (
                    profit_function.subs(kwargs)
                    .subs(cached_dist_params)
                    .subs('pi_0', positive_class_prior)
                    .subs('pi_1', negative_class_prior)
                )
            else:
                integrand = (
                    profit_function.subs(kwargs).subs('pi_0', positive_class_prior).subs('pi_1', negative_class_prior)
                )

            integrands = [
                lambdify(random_symbols, integrand.subs('F_0', tpr).subs('F_1', fpr).evalf())
                for tpr, fpr in zip(true_positive_rates, false_positive_rates, strict=True)
            ]

            results = np.empty((len(integrands), len(param_grid[0])))
            for i, integrand in enumerate(integrands):
                results[i, :] = integrand(*param_grid)
            result = results.max(axis=0).mean()

            return float(result)

        return score_function

    def _build_profit_function(self) -> sympy.Expr:
        pos_prior, neg_prior, tpr, fpr = sympy.symbols('pi_0 pi_1 F_0 F_1')
        profit_function = (
            self._tp_benefit * pos_prior * tpr
            + self._tn_benefit * neg_prior * (1 - fpr)
            - self._fn_cost * pos_prior * (1 - tpr)
            - neg_prior * self.fp_cost * fpr
        )
        return profit_function

    def _build_savings_score(self) -> MetricFn:
        cost_function = self._build_cost_function()
        all_zero_function, all_one_function = self._build_naive_cost_functions(cost_function)

        cost_func = lambdify(list(cost_function.free_symbols), cost_function)
        all_zero_func = lambdify(list(all_zero_function.free_symbols), all_zero_function)
        all_one_func = lambdify(list(all_one_function.free_symbols), all_one_function)

        def savings(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            # it is possible that with the substitution of the symbols, the function becomes a constant
            all_zero_score: float = np.mean(all_zero_func(y=y_true, **kwargs)) if all_zero_function != 0 else 0  # type: ignore[assignment]
            all_one_score: float = np.mean(all_one_func(y=y_true, **kwargs)) if all_one_function != 0 else 0  # type: ignore[assignment]
            cost_base = min(all_zero_score, all_one_score)
            return float(1 - np.mean(cost_func(y=y_true, s=y_score, **kwargs)) / cost_base)

        return savings

    def _build_cost_loss(self) -> MetricFn:
        cost_function = self._build_cost_function()
        cost_funct = lambdify(list(cost_function.free_symbols), cost_function)

        def cost_loss(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            return float(np.mean(cost_funct(y=y_true, s=y_score, **kwargs)))

        return cost_loss

    def _build_cost_function(self) -> sympy.Expr:
        y, s = sympy.symbols('y s')
        cost_function = y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * (
            (1 - s) * self.tn_cost + s * self.fp_cost
        )
        return cost_function

    def _build_naive_cost_functions(self, cost_function: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
        all_zero_function = cost_function.subs('s', 0)
        all_one_function = cost_function.subs('s', 1)
        return all_zero_function, all_one_function

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(tp_benefit={self.tp_benefit}, '
            f'tn_benefit={self.tn_benefit}, fp_cost={self.fp_cost}, '
            f'fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self) -> str:
        from sympy.printing.latex import latex

        if self.kind == 'max profit':
            profit_function = self._build_profit_function()
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

            output = output.replace('F_{0}', 'F_{0}(T)').replace('F_{1}', 'F_{1}(T)')
        elif self.kind == 'cost':
            i, N = sympy.symbols('i N')  # noqa: N806
            cost_function = (1 / N) * sympy.Sum(self._format_cost_function(), (i, 0, N))

            for symbol in cost_function.free_symbols:
                if symbol != N:
                    cost_function = cost_function.subs(symbol, str(symbol) + '_i')

            output = latex(cost_function, mode='plain', order=None)
        elif self.kind == 'savings':
            i, N, c0, c1 = sympy.symbols('i N Cost_{0} Cost_{1}')  # noqa: N806
            savings_function = (1 / (N * sympy.Min(c0, c1))) * sympy.Sum(self._format_cost_function(), (i, 0, N))

            for symbol in savings_function.free_symbols:
                if symbol not in {N, c0, c1}:
                    savings_function = savings_function.subs(symbol, str(symbol) + '_i')

            output = latex(savings_function, mode='plain', order=None)
        else:
            return repr(self)
        return f'$\\displaystyle {output}$'

    def _format_cost_function(self):
        y, s = sympy.symbols('y s')
        cost_function = y * (s * self.tp_cost + (1 - s) * self.fn_cost) + (1 - y) * (
            (1 - s) * self.tn_cost + s * self.fp_cost
        )
        return cost_function


def _check_parameters(*parameters: str | sympy.Expr) -> Callable[[MetricFn], MetricFn]:
    """
    Check if all parameters are provided.

    In particular:
        - deterministic parameters
        - distribution parameters of stochastic variables
    """

    def decorator(func: MetricFn) -> MetricFn:
        def wrapper(y_true: NDArray, y_score: NDArray, **kwargs: Any) -> float:
            for value in parameters:
                if str(value) not in kwargs:
                    raise ValueError(f'Metric expected a value for {value}, did not receive it.')
            return func(y_true, y_score, **kwargs)

        return wrapper

    return decorator
