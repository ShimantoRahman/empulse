import sys
from collections.abc import MutableMapping
from numbers import Real
from types import TracebackType
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import scipy
import sympy

from ..._types import FloatArrayLike, FloatNDArray
from .common import Direction, _evaluate_expression
from .metric_strategies import MetricStrategy


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
    Stochastic variables are supported and can be specified using sympy.stats random variables.
    Make sure that you add the parameters of the stochastic variables
    as keyword arguments when calling the metric function.
    Stochastic variables are assumed to be independent of each other.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    Parameters
    ----------
    strategy : MetricStrategy
        The strategy to use for computing the metric.

        - If :class:`~empulse.metrics.MaxProfit`,
          the metric computes the maximum profit that can be achieved by a classifier.
          The metric determines the optimal threshold that maximizes the profit.
          This metric supports the use of stochastic variables.
        - If :class:`~empulse.metrics.Cost`, the metric computes the expected cost loss of a classifier.
          This metric supports passing instance-dependent costs in the form of array-likes.
          This metric does not support stochastic variables.
        - If :class:`~empulse.metrics.Savings`,
          the metric computes the savings that can be achieved by a classifier
          over a naive classifier which always predicts 0 or 1 (whichever is better).
          This metric supports passing instance-dependent costs in the form of array-likes.
          This metric does not support stochastic variables.

    Attributes
    ----------
    tp_benefit : sympy.Expr
        The benefit of a true positive.
        See :meth:`~empulse.metrics.Metric.add_tp_benefit` for more details.

    tn_benefit : sympy.Expr
        The benefit of a true negative.
        See :meth:`~empulse.metrics.Metric.add_tn_benefit` for more details.

    fp_benefit : sympy.Expr
        The benefit of a false positive.
        See :meth:`~empulse.metrics.Metric.add_fp_benefit` for more details.

    fn_benefit : sympy.Expr
        The benefit of a false negative.
        See :meth:`~empulse.metrics.Metric.add_fn_benefit` for more details.

    tp_cost : sympy.Expr
        The cost of a true positive.
        See :meth:`~empulse.metrics.Metric.add_tp_cost` for more details.

    tn_cost : sympy.Expr
        The cost of a true negative.
        See :meth:`~empulse.metrics.Metric.add_tn_cost` for more details.

    fp_cost : sympy.Expr
        The cost of a false positive.
        See :meth:`~empulse.metrics.Metric.add_fp_cost` for more details.

    fn_cost : sympy.Expr
        The cost of a false negative.
        See :meth:`~empulse.metrics.Metric.add_fn_cost` for more details.

    direction: Direction
        Whether the metric is to be maximized or minimized.

    Examples
    --------
    Reimplementing :func:`~empulse.metrics.empc_score` using the :class:`Metric` class.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import Metric, MaxProfit

        clv, d, f, alpha, beta = sp.symbols(
            'clv d f alpha beta'
        )  # define deterministic variables
        gamma = sp.stats.Beta('gamma', alpha, beta)  # define gamma to follow a Beta distribution

        empc_score = (
            Metric(MaxProfit())
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
        from empulse.metrics import Metric, Cost

        clv, delta, f, gamma = sp.symbols('clv delta f gamma')

        cost_loss = (
            Metric(Cost())
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

    Using the Metric class as a context manager (automatically builds after assembling the components).
    Also adding default values for some parameters.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import Metric, Cost

        clv, delta, f, gamma = sp.symbols('clv delta f gamma')

        with Metric(Cost()) as cost_loss:
            cost_loss.add_tp_benefit(gamma * (clv - delta * clv - f))
            cost_loss.add_tp_benefit((1 - gamma) * -f)
            cost_loss.add_fp_cost(delta * clv + f)
            cost_loss.alias('incentive_fraction', delta)
            cost_loss.alias('contact_cost', f)
            cost_loss.alias('accept_rate', gamma)
            cost_loss.set_default(incentive_fraction=0.05, contact_cost=1, accept_rate=0.3)

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]

        cost_loss(y_true, y_proba, clv=100)
    """

    def __init__(self, strategy: MetricStrategy) -> None:
        self.strategy = strategy
        self._tp_benefit: sympy.Expr = sympy.core.numbers.Zero()
        self._tn_benefit: sympy.Expr = sympy.core.numbers.Zero()
        self._fp_cost: sympy.Expr = sympy.core.numbers.Zero()
        self._fn_cost: sympy.Expr = sympy.core.numbers.Zero()
        self._aliases: MutableMapping[str, str | sympy.Symbol] = {}
        self._defaults: dict[str, Any] = {}
        self._outlier_sensitive_symbols: set[sympy.Symbol] = set()
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
    def fn_cost(self) -> sympy.Expr:  # noqa: D102
        return self._fn_cost

    @property
    def direction(self) -> Direction:  # noqa: D102
        return self.strategy.direction

    def add_tp_benefit(self, term: sympy.Expr | str) -> Self:
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

    def add_tn_benefit(self, term: sympy.Expr | str) -> Self:
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

    def add_fp_benefit(self, term: sympy.Expr | str) -> Self:
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

    def add_fn_benefit(self, term: sympy.Expr | str) -> Self:
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

    def add_tp_cost(self, term: sympy.Expr | str) -> Self:
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

    def add_tn_cost(self, term: sympy.Expr | str) -> Self:
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

    def add_fp_cost(self, term: sympy.Expr | str) -> Self:
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

    def add_fn_cost(self, term: sympy.Expr | str) -> Self:
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

    def alias(self, alias: str | MutableMapping[str, sympy.Symbol | str], symbol: sympy.Symbol | None = None) -> Self:
        """
        Add an alias for a symbol.

        Parameters
        ----------
        alias: str | MutableMapping[str, sympy.Symbol | str]
            The alias to add. If a MutableMapping (.e.g, dictionary) is passed,
            the keys are the aliases and the values are the symbols.
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
        if isinstance(alias, MutableMapping):
            self._aliases.update(alias)
        elif symbol is not None:
            self._aliases[alias] = str(symbol)
        else:
            raise ValueError('Either a dictionary or both an alias and a symbol should be provided')
        return self

    def set_default(self, **defaults: float) -> Self:
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

    def mark_outlier_sensitive(self, symbol: str | sympy.Symbol) -> Self:
        """
        Mark a symbol as outlier-sensitive.

        This is used to indicate that the symbol is sensitive to outliers.
        When the metric is used as a loss function or criterion for training a model,
        :class:`~empulse.models.RobustCSClassifier` will impute outliers for this symbol's value.
        This is ignored when not using a :class:`~empulse.models.RobustCSClassifier` model.

        Parameters
        ----------
        symbol: str | sympy.Symbol
            The symbol to mark as outlier-sensitive.

        Returns
        -------
        Metric

        Examples
        --------
        .. code-block:: python

            import numpy as np
            import sympy as sp
            from empulse.metrics import Metric, Cost
            from empulse.models import CSLogitClassifier, RobustCSClassifier
            from sklearn.datasets import make_classification

            X, y = make_classification()
            a, b = sp.symbols('a b')
            cost_loss = Metric(Cost()).add_fp_cost(a).add_fn_cost(b).mark_outlier_sensitive(a).build()
            fn_cost = np.random.rand(y.size)

            model = RobustCSClassifier(CSLogitClassifier(loss=cost_loss))
            model.fit(X, y, a=np.random.rand(y.size), b=5)
        """
        if isinstance(symbol, str):
            symbol = sympy.sympify(symbol)
        if not isinstance(symbol, sympy.Symbol):
            raise ValueError('The symbol must be a sympy.Symbol or a string that can be converted to a sympy.Symbol')
        self._outlier_sensitive_symbols.add(symbol)
        return self

    def build(self) -> Self:
        """
        Build the metric function.

        This function should be called last after adding all the cost-benefit terms.
        After calling this function, the metric function can be called with the true labels and predicted probabilities.

        This function is automatically called when using
        the :class:`~empulse.metrics.Metric` class as a context manager.

        Returns
        -------
        Metric
        """
        self._built = True
        self.strategy.build(
            tp_benefit=self.tp_benefit,
            tn_benefit=self.tn_benefit,
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )
        return self

    def _prepare_parameters(self, **kwargs: FloatArrayLike | float) -> dict[str, FloatNDArray | float]:
        """Swap aliases with the appropriate symbols and convert the values to numpy arrays."""
        # Use default values if not provided in kwargs
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        for key, value in kwargs.items():
            if not isinstance(value, Real):
                kwargs[key] = np.asarray(value).reshape(-1)

        # Map aliases to the appropriate symbols
        for alias, symbol in self._aliases.items():
            if alias in kwargs:
                kwargs[symbol] = kwargs.pop(alias)
        kwargs: dict[str, FloatNDArray | float]  # redefine kwargs as mypy doesn't understand the above

        return kwargs

    def __call__(self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float) -> float:
        """
        Compute the metric score or loss.

        The :meth:`empulse.metrics.Metric.build` method should be called before calling this method.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

            - If :class:`~empulse.metrics.MaxProfit`, the predicted labels are the decision scores.
            - If :class:`~empulse.metrics.Cost`, the predicted labels are the (calibrated) probabilities.
            - If :class:`~empulse.metrics.Savings`, the predicted labels are the (calibrated) probabilities.

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
        self._check_built()
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        parameters = self._prepare_parameters(**parameters)
        return self.strategy.score(y_true, y_score, **parameters)

    def optimal_threshold(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float
    ) -> FloatNDArray | float:
        """
        Compute the optimal classification threshold(s).

        i.e., the score threshold at which an observation should be classified as positive to optimize the metric.
        For instance-dependent costs and benefits, this will return an array of thresholds, one for each sample.
        For class-dependent costs and benefits, this will return a single threshold value.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

            - If :class:`~empulse.metrics.MaxProfit`, the predicted labels are the decision scores.
            - If :class:`~empulse.metrics.Cost`, the predicted labels are the (calibrated) probabilities.
            - If :class:`~empulse.metrics.Savings`, the predicted labels are the (calibrated) probabilities.

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
        self._check_built()
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        parameters = self._prepare_parameters(**parameters)
        return self.strategy.optimal_threshold(y_true, y_score, **parameters)

    def optimal_rate(
        self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float
    ) -> float:
        """
        Compute the optimal predicted positive rate.

        i.e., the fraction of observations that should be classified as positive to optimize the metric.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,)
            The ground truth labels.

        y_score: array-like of shape (n_samples,)
            The predicted labels, probabilities, or decision scores (based on the chosen metric).

            - If :class:`~empulse.metrics.MaxProfit`, the predicted labels are the decision scores.
            - If :class:`~empulse.metrics.Cost`, the predicted labels are the (calibrated) probabilities.
            - If :class:`~empulse.metrics.Savings`, the predicted labels are the (calibrated) probabilities.

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
        self._check_built()
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        parameters = self._prepare_parameters(**parameters)
        return self.strategy.optimal_rate(y_true, y_score, **parameters)

    def _logit_objective(
        self, features: FloatNDArray, weights: FloatNDArray, y_true: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[float, FloatNDArray]:
        """
        Compute the metric loss and its gradient with respect to the logistic regression weights.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
        weights : NDArray of shape (n_features,)
            The weights of the logistic regression model.
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
        value : float
            The metric loss to be minimized.
        gradient : NDArray of shape (n_features,)
            The gradient of the metric loss with respect to the logistic regression weights.
        """
        parameters = self._prepare_parameters(**parameters)

        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=1)
        for key, value in parameters.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                parameters[key] = np.expand_dims(value, axis=1)

        return self.strategy.logit_objective(features, weights, y_true, **parameters)

    def _gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient and hessian of the metric loss with respect to the gradient boosting weights.

        Parameters
        ----------
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        y_score : NDArray of shape (n_samples,)
            The predicted probabilities or decision scores.
        parameters : float or NDArray of shape (n_samples,)
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
        parameters = self._prepare_parameters(**parameters)
        y_proba = scipy.special.expit(y_score)
        gradient, hessian = self.strategy.gradient_boost_objective(y_true, y_proba, **parameters)
        return gradient, hessian

    def _get_cost_matrix(self, n_samples: int, **parameters: FloatNDArray | float) -> FloatNDArray:
        """
        Compute the cost matrix based on the metric's costs and benefits.

        Parameters
        ----------
        n_samples : int
            The number of samples for which to compute the cost matrix.
            This is used to ensure that the cost matrix has the correct shape.
        parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

        Returns
        -------
        cost_matrix : NDArray of shape (N, 4)
            The cost matrix consisting of fp_cost, fn_cost, tp_cost, and tn_cost (in that order).
        """
        parameters = self._prepare_parameters(**parameters)

        fp_cost = _evaluate_expression(self.fp_cost, **parameters)
        fn_cost = _evaluate_expression(self.fn_cost, **parameters)
        tp_cost = _evaluate_expression(self.tp_cost, **parameters)
        tn_cost = _evaluate_expression(self.tn_cost, **parameters)

        # Ensure all costs and benefits are numpy arrays with the same length
        if isinstance(fp_cost, float | int):
            fp_cost = np.full(n_samples, fp_cost)
        if isinstance(fn_cost, float | int):
            fn_cost = np.full(n_samples, fn_cost)
        if isinstance(tp_cost, float | int):
            tp_cost = np.full(n_samples, tp_cost)
        if isinstance(tn_cost, float | int):
            tn_cost = np.full(n_samples, tn_cost)

        return np.column_stack((fp_cost, fn_cost, tp_cost, tn_cost))

    def _check_built(self) -> None:
        if not self._built:
            raise ValueError('The metric function has not been built. Call the build method before calling the metric')

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(strategy={self.strategy}, '
            f'tp_benefit={self.tp_benefit}, tn_benefit={self.tn_benefit}, '
            f'fp_cost={self.fp_cost}, fn_cost={self.fn_cost})'
        )

    def _repr_latex_(self) -> str:
        return self.strategy.to_latex(
            tp_benefit=self.tp_benefit, tn_benefit=self.tn_benefit, fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.build()
