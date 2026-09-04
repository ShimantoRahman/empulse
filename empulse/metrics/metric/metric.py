import copy
import warnings
from numbers import Real

import numpy as np
import sympy

from ..._types import FloatArrayLike, FloatNDArray
from .._validation import _check_y_pred, _check_y_true
from .base_metric import BaseMetric
from .common import (
    Direction,
    _check_known_alias_and_default_targets,
    _check_reserved_symbol_names,
    _evaluate_expression,
    replace_random_var_with_mean,
)
from .cost_matrix import CostMatrix
from .strategies import LogitObjective, MetricStrategy


class Metric(BaseMetric):
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
          Any stochastic variable is reduced to its mean before use; the metric itself is always
          evaluated deterministically.
        - If :class:`~empulse.metrics.Savings`,
          the metric computes the savings that can be achieved by a classifier
          over a naive classifier which always predicts 0 or 1 (whichever is better).
          This metric supports passing instance-dependent costs in the form of array-likes.
          Any stochastic variable is reduced to its mean before use; the metric itself is always
          evaluated deterministically.

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
        self.cost_matrix = copy.deepcopy(cost_matrix)
        _check_reserved_symbol_names(
            self.tp_benefit,
            self.tn_benefit,
            self.fp_cost,
            self.fn_cost,
            alias_names=self.cost_matrix._aliases.keys(),
        )
        _check_known_alias_and_default_targets(
            self.tp_benefit,
            self.tn_benefit,
            self.fp_cost,
            self.fn_cost,
            aliases=self.cost_matrix._aliases,
            default_names=self.cost_matrix._defaults.keys(),
        )
        if self.tp_benefit == 0 and self.tn_benefit == 0 and self.fp_cost == 0 and self.fn_cost == 0:
            warnings.warn(
                'The cost matrix has no cost or benefit terms (or they cancel out to exactly '
                'zero); this metric will always evaluate to 0.0 regardless of y_true, y_score, or '
                'the parameters passed in. Did you forget to call add_tp_benefit()/'
                'add_tn_benefit()/add_fp_cost()/add_fn_cost() on the CostMatrix?',
                UserWarning,
                stacklevel=2,
            )
        self._strategy = copy.deepcopy(strategy)
        self._strategy.build(
            tp_benefit=self.tp_benefit,
            tn_benefit=self.tn_benefit,
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )

    @property
    def strategy(self) -> MetricStrategy:
        """The strategy used to compute the metric."""
        return self._strategy

    @property
    def __name__(self) -> str:
        return self.strategy.name

    @__name__.setter  # noqa: A003
    def __name__(self, value: str) -> None:
        self.strategy.name = value

    @property
    def tp_benefit(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.tp_benefit

    @property
    def tn_benefit(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.tn_benefit

    @property
    def fp_benefit(self) -> sympy.Expr:  # noqa: D102
        return self.cost_matrix.fp_benefit

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

    @property
    def _all_symbols(self) -> set[str]:
        """The set of all symbols used in the cost matrix."""
        all_symbols = (
            self.tp_cost.free_symbols
            | self.tn_cost.free_symbols
            | self.fp_cost.free_symbols
            | self.fn_cost.free_symbols
        )

        # Extract parameters from stochastic variables
        stochastic_params = set()
        for expr in [
            self.cost_matrix.tp_cost,
            self.cost_matrix.tn_cost,
            self.cost_matrix.fp_cost,
            self.cost_matrix.fn_cost,
        ]:
            for atom in expr.atoms(sympy.stats.rv.RandomSymbol):
                pspace = atom.pspace
                if hasattr(pspace, 'distribution') and hasattr(pspace.distribution, 'args'):
                    for arg in pspace.distribution.args:
                        stochastic_params.update(arg.free_symbols)

        return {str(symbol) for symbol in all_symbols | stochastic_params} | set(self.cost_matrix._aliases.keys())

    @property
    def _all_parameters(self) -> set[str]:
        """The set of cost matrix parameters which can be used."""
        all_symbols = (
            self.tp_cost.free_symbols
            | self.tn_cost.free_symbols
            | self.fp_cost.free_symbols
            | self.fn_cost.free_symbols
        )

        # Extract parameters from stochastic variables
        stochastic_symbols = set()
        stochastic_params = set()
        for expr in [
            self.cost_matrix.tp_cost,
            self.cost_matrix.tn_cost,
            self.cost_matrix.fp_cost,
            self.cost_matrix.fn_cost,
        ]:
            for atom in expr.atoms(sympy.stats.rv.RandomSymbol):
                stochastic_symbols.add(atom)
                pspace = atom.pspace
                if hasattr(pspace, 'distribution') and hasattr(pspace.distribution, 'args'):
                    for arg in pspace.distribution.args:
                        stochastic_params.update(arg.free_symbols)

        return {str(symbol) for symbol in (all_symbols | stochastic_params) - stochastic_symbols} | set(
            self.cost_matrix._aliases.keys()
        )

    @property
    def _default_parameter_names(self) -> set[str]:
        """The set of parameter names that have a default value and need not be supplied."""
        return set(self.cost_matrix._defaults.keys())

    @property
    def _is_stochastic(self) -> bool:
        all_symbols = (
            self.tp_cost.free_symbols
            | self.tn_cost.free_symbols
            | self.fp_cost.free_symbols
            | self.fn_cost.free_symbols
        )
        return any(sympy.stats.rv.is_random(symbol) for symbol in all_symbols)

    @property
    def _is_deterministic(self) -> bool:
        return not self._is_stochastic

    def _prepare_parameters(
        self, *, n_samples: int | None = None, **kwargs: FloatArrayLike | float
    ) -> dict[str, FloatNDArray | float]:
        """
        Swap aliases with the appropriate symbols and convert the values to numpy arrays.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples being scored (typically ``y_true.size``). When given, every
            array-like parameter must have this length or be a scalar-like array of length 1;
            anything else raises a ``ValueError`` naming the offending parameter. When ``None``,
            no length check is performed - callers that don't have an obvious sample count to
            check against (e.g. ``_evaluate_costs``) simply omit it.
        """
        # Use a separate output dict to avoid dual-purpose mutation of kwargs
        params: dict[str, FloatArrayLike | float] = {}

        # Map aliases to the appropriate symbol names. Track which caller-supplied key each
        # resolved symbol name came from, so that passing both a symbol and one of its aliases
        # (or two different aliases for the same symbol) raises instead of silently letting
        # whichever kwarg happens to be seen last win - which would make the result depend on
        # the caller's kwarg order.
        resolved_from: dict[str, str] = {}
        for key, value in kwargs.items():
            alias_target = self.cost_matrix._aliases.get(key)
            symbol_name = str(alias_target) if alias_target is not None else key
            if symbol_name in resolved_from:
                raise ValueError(
                    f"Got conflicting values for symbol '{symbol_name}': passed both as "
                    f"'{resolved_from[symbol_name]}' and '{key}'. Pass only one of these."
                )
            resolved_from[symbol_name] = key
            params[symbol_name] = value

        # Use default values if not provided
        for key, value in self.cost_matrix._defaults.items():
            params.setdefault(key, value)

        # Warn about unknown parameters (likely typos), excluding strategy-specific extras
        # and sklearn metadata-routing kwargs (e.g. sample_weight passed via fit()).
        sklearn_internal_params = frozenset({'sample_weight'})
        known = self._all_parameters | self.strategy._extra_kwargs | sklearn_internal_params
        extra_keys = set(params) - known
        if extra_keys:
            warnings.warn(
                f'Unknown parameters passed to metric: {sorted(extra_keys)}. These will be ignored.',
                UserWarning,
                stacklevel=3,
            )

        out: dict[str, FloatNDArray | float] = {}
        for key, value in params.items():
            if not isinstance(value, Real | str):
                arr = np.asarray(value).reshape(-1)
                arr = arr.astype(np.float64) if not np.issubdtype(arr.dtype, np.floating) else arr
                if n_samples is not None and arr.size not in {1, n_samples}:
                    caller_key = resolved_from.get(key, key)
                    raise ValueError(
                        f"Parameter '{caller_key}' has length {arr.size}, but expected length "
                        f'{n_samples} (one value per sample, matching y_true/y_score) or a '
                        f'single value (length 1, applied to every sample).'
                    )
                out[key] = arr
            elif isinstance(value, int):
                out[key] = float(value)
            else:
                out[key] = value  # type: ignore[assignment]

        return out

    def __call__(self, y_true: FloatArrayLike, y_score: FloatArrayLike, **parameters: FloatArrayLike | float) -> float:
        """
        Compute the metric score or loss.

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
        y_true = _check_y_true(np.asarray(y_true).reshape(-1), check_variance=False)
        y_score = _check_y_pred(np.asarray(y_score).reshape(-1))
        if y_true.size != y_score.size:
            raise ValueError(f'y_true and y_score must have the same length, got {y_true.size} and {y_score.size}.')
        parameters = self._prepare_parameters(n_samples=y_true.size, **parameters)
        return self.strategy.score(y_true.astype(np.intp), y_score, **parameters)

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
        y_true = np.asarray(y_true).reshape(-1)
        # y_true may be empty when optimal_threshold/optimal_rate don't need labels
        if y_true.size > 0:
            y_true = _check_y_true(y_true, check_variance=False)
        y_score = _check_y_pred(np.asarray(y_score).reshape(-1))
        if y_true.size > 0 and y_true.size != y_score.size:
            raise ValueError(f'y_true and y_score must have the same length, got {y_true.size} and {y_score.size}.')
        n_samples = y_true.size or y_score.size or None
        parameters = self._prepare_parameters(n_samples=n_samples, **parameters)
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
        y_true = np.asarray(y_true).reshape(-1)
        # y_true may be empty when optimal_rate doesn't need labels (threshold-only path)
        if y_true.size > 0:
            y_true = _check_y_true(y_true, check_variance=False)
        y_score = _check_y_pred(np.asarray(y_score).reshape(-1))
        if y_true.size > 0 and y_true.size != y_score.size:
            raise ValueError(f'y_true and y_score must have the same length, got {y_true.size} and {y_score.size}.')
        n_samples = y_true.size or y_score.size or None
        parameters = self._prepare_parameters(n_samples=n_samples, **parameters)
        return self.strategy.optimal_rate(y_true, y_score, **parameters)

    def _logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        soft_threshold: bool,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """
        Compute the logit loss and its gradient with respect to the logistic regression weights.

        Parameters
        ----------
        features : NDArray of shape (n_samples, n_features)
            The features of the samples.
        y_true : NDArray of shape (n_samples,)
            The ground truth labels.
        C : float
            The inverse of regularization strength for logistic regression.
        l1_ratio : float
            The mixing parameter for elastic net regularization in logistic regression.
        soft_threshold : bool
            If ``True``, apply soft-thresholding to the regression coefficients.
        fit_intercept : bool
            Whether the logistic regression model includes an intercept term.
        parameters : float or NDArray of shape (n_samples,)
            The parameter values for the costs and benefits defined in the metric.
            If any parameter is a stochastic variable, you should pass values for their distribution parameters.
            You can set the parameter values for either the symbol names or their aliases.

            - If ``float``, the same value is used for all samples (class-dependent).
            - If ``array-like``, the values are used for each sample (instance-dependent).

        Returns
        -------
        logistic_objective : LogitObjective
            A class that implements the logit loss and its gradient.
        """
        parameters = self._prepare_parameters(**parameters)  # type: ignore[arg-type]

        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=1)
        for key, value in parameters.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                parameters[key] = np.expand_dims(value, axis=1)

        return self.strategy.logit_objective(
            features=features,
            y_true=y_true,
            C=C,
            l1_ratio=l1_ratio,
            soft_threshold=soft_threshold,
            fit_intercept=fit_intercept,
            **parameters,
        )

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
        parameters = self._prepare_parameters(**parameters)  # type: ignore[arg-type]
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
        parameters = self._prepare_parameters(**parameters)  # type: ignore[arg-type]
        for key, value in parameters.items():
            if isinstance(value, np.ndarray) and value.ndim == 1:
                parameters[key] = np.expand_dims(value, axis=1)
        return self.strategy.prepare_boost_objective(y_true, **parameters)

    def _evaluate_costs(
        self, *, replace_stochastic: bool = False, **parameters: FloatNDArray | float
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
        replace_stochastic : bool, default=False
            If ``True`` and the metric contains stochastic (random) variables, each random
            variable is first replaced by its mean, allowing the costs to be evaluated as if
            they were deterministic. If ``False``, stochastic variables are left as-is and
            evaluating them directly will raise an error.

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
        parameters = self._prepare_parameters(**parameters)  # type: ignore[arg-type]
        fp_expr, fn_expr, tp_expr, tn_expr = self.fp_cost, self.fn_cost, self.tp_cost, self.tn_cost
        if replace_stochastic and self._is_stochastic:
            fp_expr, fn_expr, tp_expr, tn_expr = replace_random_var_with_mean(fp_expr, fn_expr, tp_expr, tn_expr)
        fp_cost = _evaluate_expression(fp_expr, **parameters)
        fn_cost = _evaluate_expression(fn_expr, **parameters)
        tp_cost = _evaluate_expression(tp_expr, **parameters)
        tn_cost = _evaluate_expression(tn_expr, **parameters)
        return fp_cost, fn_cost, tp_cost, tn_cost

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cost_matrix={self.cost_matrix}, strategy={self.strategy})'

    def _repr_latex_(self) -> str:
        return self.strategy.to_latex(
            tp_benefit=self.tp_benefit, tn_benefit=self.tn_benefit, fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )
