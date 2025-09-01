from numbers import Real

import numpy as np
import sympy

from ..._types import FloatArrayLike, FloatNDArray
from .common import Direction, _evaluate_expression
from .cost_matrix import CostMatrix
from .metric_strategies import MetricStrategy


class Metric:
    """
    Class to create a custom value/cost-sensitive metric.

    The metric is defined by a cost matrix and a strategy for computing the metric.
    The cost matrix defines the costs and benefits associated with each type of prediction outcome
    (true positive, true negative, false positive, false negative).
    The strategy defines how to compute the metric based on the cost matrix.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    Parameters
    ----------
    cost_matrix : CostMatrix
        The cost matrix defining the costs and benefits associated with each type of prediction outcome.

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
        from empulse.metrics import Metric, MaxProfit, CostMatrix

        clv, d, f, alpha, beta = sp.symbols(
            'clv d f alpha beta'
        )  # define deterministic variables
        gamma = sp.stats.Beta('gamma', alpha, beta)  # define gamma to follow a Beta distribution

        cost_matrix = (
            CostMatrix()
            .add_tp_benefit(gamma * (clv - d - f))  # when churner accepts offer
            .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
            .add_fp_cost(d + f)  # when you send an offer to a non-churner
            .alias({'incentive_cost': 'd', 'contact_cost': 'f'})
        )
        empc_score = Metric(cost_matrix, MaxProfit())

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]

        empc_score(y_true, y_proba, clv=100, incentive_cost=10, contact_cost=1, alpha=6, beta=14)

    Reimplementing :func:`~empulse.metrics.expected_cost_loss_churn` using the :class:`Metric` class.

    .. code-block:: python

        import sympy as sp
        from empulse.metrics import Metric, Cost, CostMatrix

        clv, delta, f, gamma = sp.symbols('clv delta f gamma')

        cost_matrix = (
            CostMatrix()
            .add_tp_benefit(gamma * (clv - delta * clv - f))  # when churner accepts offer
            .add_tp_benefit((1 - gamma) * -f)  # when churner does not accept offer
            .add_fp_cost(delta * clv + f)  # when you send an offer to a non-churner
            .alias({'incentive_fraction': 'delta', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        )
        cost_loss = Metric(cost_matrix, Cost())

        y_true = [1, 0, 1, 0, 1]
        y_proba = [0.9, 0.1, 0.8, 0.2, 0.7]

        cost_loss(
            y_true, y_proba, clv=100, incentive_fraction=0.05, contact_cost=1, accept_rate=0.3
        )
    """

    def __init__(self, cost_matrix: CostMatrix, strategy: MetricStrategy) -> None:
        self.cost_matrix = cost_matrix
        self.strategy = strategy
        self.strategy.build(
            tp_benefit=self.tp_benefit,
            tn_benefit=self.tn_benefit,
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )

    @property
    def tp_benefit(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.tp_benefit

    @property
    def tn_benefit(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.tn_benefit

    @property
    def fp_benefit(self) -> sympy.Expr:  # noqa: D102
        return -self.cost_matrix.fp_cost

    @property
    def fn_benefit(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.fn_benefit

    @property
    def tp_cost(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.tp_cost

    @property
    def tn_cost(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.tn_cost

    @property
    def fp_cost(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.fp_cost

    @property
    def fn_cost(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.fn_cost

    @property
    def direction(self) -> Direction:  # noqa: D102
        return self.strategy.direction

    def _prepare_parameters(self, **kwargs: FloatArrayLike | float) -> dict[str, FloatNDArray | float]:
        """Swap aliases with the appropriate symbols and convert the values to numpy arrays."""
        # Map aliases to the appropriate symbols
        for alias, symbol in self.cost_matrix._aliases.items():
            if alias in kwargs:
                kwargs[symbol] = kwargs.pop(alias)

        # Use default values if not provided in kwargs
        for key, value in self.cost_matrix._defaults.items():
            kwargs.setdefault(key, value)

        for key, value in kwargs.items():
            if not isinstance(value, Real):
                kwargs[key] = np.asarray(value).reshape(-1)

        # convert any ints to floats
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) and not np.issubdtype(value.dtype, np.floating):
                kwargs[key] = value.astype(np.float64)
            elif isinstance(value, int):
                kwargs[key] = float(value)

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
        optimal_threshold: float or NDArray of shape (n_samples,)
            The optimal classification threshold(s).
        """
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

    def _prepare_logit_objective(
        self, features: FloatNDArray, y_true: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
        """
        Compute the constant term of the loss and gradient of the metric wrt logistic regression coefficients.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
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
        gradient_const : NDArray of shape (n_samples, n_features)
            The constant term of the gradient.
        loss_const1 : NDArray of shape (n_features,)
            The first constant term of the loss function.
        loss_const2 : NDArray of shape (n_features,)
            The second constant term of the loss function.
        """
        parameters = self._prepare_parameters(**parameters)
        for key, value in parameters.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                parameters[key] = np.expand_dims(value, axis=1)
        return self.strategy.prepare_logit_objective(features, y_true, **parameters)

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
        # y_proba = scipy.special.expit(y_score)
        y_proba = y_score
        gradient, hessian = self.strategy.gradient_boost_objective(y_true, y_proba, **parameters)
        return gradient, hessian

    def _prepare_boost_objective(self, y_true: FloatNDArray, **parameters: FloatNDArray | float) -> FloatNDArray:
        """
        Compute the gradient's constant term of the metric wrt gradient boost.

        Parameters
        ----------
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
        gradient_const : NDArray of shape (n_samples, n_features)
            The constant term of the gradient.
        """
        parameters = self._prepare_parameters(**parameters)
        for key, value in parameters.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                parameters[key] = np.expand_dims(value, axis=1)
        return self.strategy.prepare_boost_objective(y_true, **parameters)

    def _evaluate_costs(
        self, **parameters: FloatNDArray | float
    ) -> tuple[
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
        FloatNDArray | float,
    ]:
        """
        Evaluate the costs expressions.

        Parameters
        ----------
        parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

        Returns
        -------
        fp_cost : float or NDArray of shape (n_samples,)
            The false positive cost(s).
        fn_cost : float or NDArray of shape (n_samples,)
            The false negative cost(s).
        tp_cost : float or NDArray of shape (n_samples,)
            The true positive cost(s).
        tn_cost : float or NDArray of shape (n_samples,)
            The true negative cost(s).
        """
        parameters = self._prepare_parameters(**parameters)
        fp_cost = _evaluate_expression(self.fp_cost, **parameters)
        fn_cost = _evaluate_expression(self.fn_cost, **parameters)
        tp_cost = _evaluate_expression(self.tp_cost, **parameters)
        tn_cost = _evaluate_expression(self.tn_cost, **parameters)
        return fp_cost, fn_cost, tp_cost, tn_cost

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cost_matrix={self.cost_matrix}, strategy={self.strategy})'

    def _repr_latex_(self) -> str:
        return self.strategy.to_latex(
            tp_benefit=self.tp_benefit, tn_benefit=self.tn_benefit, fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )
