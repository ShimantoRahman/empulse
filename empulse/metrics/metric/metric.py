from numbers import Real
from typing import Any, ClassVar, Final, Literal, Protocol

import numpy as np
import scipy
import sympy
from numpy.typing import ArrayLike, NDArray

from .cost_metric import _build_cost_gradient_logit, _build_cost_loss, _cost_loss_to_latex
from .max_profit_metric import _build_max_profit_score, _max_profit_score_to_latex
from .savings_metric import _build_savings_score, _savings_score_to_latex


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
            - :class:`sympy.stats.Gompertz`
            - :class:`sympy.stats.Laplace`
            - :class:`sympy.stats.Levy`
            - :class:`sympy.stats.Logistic`
            - :class:`sympy.stats.LogNormal`
            - :class:`sympy.stats.Lomax`
            - :class:`sympy.stats.Normal`
            - :class:`sympy.stats.Maxwell`
            - :class:`sympy.stats.Moyal`
            - :class:`sympy.stats.Nakagami`
            - :class:`sympy.stats.Pareto`
            - :class:`sympy.stats.PowerFunction`
            - :class:`sympy.stats.StudentT`
            - :class:`sympy.stats.Trapezoidal`
            - :class:`sympy.stats.Triangular`
            - :class:`sympy.stats.Uniform`
            - :class:`sympy.stats.VonMises`

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

        def not_defined(*args, **kwargs):
            raise NotImplementedError

        self.build_gradient_logit = not_defined
        if kind == 'max profit':
            self.build_metric = _build_max_profit_score
            self.metric_to_latex = _max_profit_score_to_latex
        elif kind == 'cost':
            self.build_metric = _build_cost_loss
            self.build_gradient_logit = _build_cost_gradient_logit
            self.metric_to_latex = _cost_loss_to_latex
        elif kind == 'savings':
            self.build_metric = _build_savings_score
            self.metric_to_latex = _savings_score_to_latex
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
        self._score_function = self.build_metric(
            tp_benefit=self.tp_benefit,
            tn_benefit=self.tn_benefit,
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
            integration_method=self.integration_method,
            n_mc_samples=self.n_mc_samples,
            rng=self._rng,
        )
        self._gradient_logit_function = self.build_gradient_logit(
            tp_benefit=self.tp_benefit,
            tn_benefit=self.tn_benefit,
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )
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

    def _objective_logit(
        self, features: NDArray, weights: NDArray, y_true: NDArray, **kwargs: NDArray | float
    ) -> tuple[float | NDArray]:
        y_pred = scipy.special.expit(np.dot(weights, features.T))

        if y_pred.ndim == 1:
            y_pred = np.expand_dims(y_pred, axis=1)
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=1)

        value = self.__call__(y_true, y_pred, **kwargs)
        gradient = self._gradient_logit_function(features, y_true, y_pred, **kwargs)
        return value, gradient

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(tp_benefit={self.tp_benefit}, '
            f'tn_benefit={self.tn_benefit}, fp_cost={self.fp_cost}, '
            f'fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self) -> str:
        return self.metric_to_latex(
            tp_benefit=self.tp_benefit, tn_benefit=self.tn_benefit, fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )
