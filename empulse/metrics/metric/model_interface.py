from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, final

import numpy as np

from ..._common._objective import objective_scale_from_costs
from ..._types import FloatArrayLike, FloatNDArray
from .._validation import _check_shape, _check_y_pred
from ._compile import CountScoreFn
from .capabilities import Capability
from .strategies import LogitObjective
from .strategies.auepc_strategy import _ranked_profit_curve

if TYPE_CHECKING:
    from .base_metric import BaseMetric

_Costs = tuple[FloatNDArray | float, FloatNDArray | float, FloatNDArray | float, FloatNDArray | float]


@final
class ModelInterface:
    """
    What a model needs of a metric to train on it, decide with it, and route its parameters.

    Reach it through a metric's :attr:`~empulse.metrics.BaseMetric.model` attribute, rather than
    constructing it. It holds nothing but the metric, so it costs nothing to create; a model that
    uses it in a loop can bind it once (``interface = loss.model``).

    Only someone writing their own estimator needs it. To evaluate a model or pick a threshold, call
    the metric itself, :meth:`~empulse.metrics.Metric.optimal_threshold` or
    :meth:`~empulse.metrics.Metric.optimal_rate`.

    Parameters
    ----------
    metric : BaseMetric
        The metric to expose.

    Notes
    -----
    A model never has to branch on the metric's :attr:`~empulse.metrics.BaseMetric.direction`:

    - :meth:`loss`, :meth:`logit_objective`, :meth:`logit_value_objective`,
      :meth:`gradient_boost_objective` and :meth:`prepare_boost_objective` are always **minimized**,
      whether the metric is a cost or a profit. :class:`~empulse.metrics.Cost` and
      :class:`~empulse.metrics.Profit` give a model identical values.
    - :meth:`evaluate_costs` and :meth:`decision_costs` do not depend on the direction, and neither do
      the metric's own :meth:`~empulse.metrics.Metric.optimal_threshold` and
      :meth:`~empulse.metrics.Metric.optimal_rate`.

    The training objectives have the same minimizer as :meth:`loss`, but need not equal it.
    :class:`~empulse.metrics.Savings` trains on the expected cost, because savings is a decreasing
    affine map of cost; :meth:`cost_weight` is the slope of that map.

    Arguments a metric is evaluated on many times during a fit are validated once, with
    :meth:`validate_parameters`. Pass ``validate=False`` to the methods that take it inside a
    training loop, since validation reads the array data.

    Examples
    --------
    The costs a model trains on, and the loss of its predictions:

    .. code-block:: python

        import numpy as np
        from empulse.metrics import Cost, CostMatrix, Metric

        metric = Metric(CostMatrix().add_fp_cost('fp').add_fn_cost('fn'), Cost())
        interface = metric.model

        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.2, 0.7, 0.4, 0.1])
        fn = np.array([5.0, 8.0, 2.0, 5.0])

        interface.validate_parameters(fp=1.0, fn=fn)
        fp_cost, fn_cost, tp_cost, tn_cost = interface.decision_costs(y_true, fp=1.0, fn=fn)
        loss = interface.loss(y_true, y_proba, validate=False, fp=1.0, fn=fn)
    """

    __slots__ = ('_metric',)

    def __init__(self, metric: 'BaseMetric') -> None:
        self._metric = metric

    def __repr__(self) -> str:
        return f'ModelInterface({self._metric!r})'

    @property
    def accepted_parameters(self) -> set[str]:
        """
        Every keyword argument the metric accepts.

        Its :attr:`~empulse.metrics.BaseMetric.parameter_names` and the strategy's own options, such as
        :class:`~empulse.metrics.Savings`' ``baseline``. A model that forwards its ``fit`` keyword
        arguments to the metric accepts these.
        """
        return self._metric._all_parameters | self._metric._extra_parameters

    def missing_parameters(self, supplied: Iterable[str]) -> set[str]:
        """
        Return the required parameters that *supplied* does not cover.

        A parameter is covered when *supplied* names it under any of its spellings, or when it has a
        default value. Names in *supplied* that the metric does not use are ignored.

        Parameters
        ----------
        supplied : iterable of str
            The keyword arguments a caller has.

        Returns
        -------
        missing : set of str
            The symbol names of the parameters still needed.
        """
        return self._metric._missing_parameters(supplied)

    def validate_parameters(self, **parameters: Any) -> None:
        """
        Check parameter values against the domain the metric declares.

        Call it once, where the values arrive, such as at the start of ``fit``; then pass
        ``validate=False`` wherever training re-evaluates the metric on the same values.

        Parameters
        ----------
        **parameters
            The metric's parameters.

        Raises
        ------
        ValueError
            If a distribution rejects its shape parameters, or a value falls outside bounds or
            conditions declared with :meth:`~empulse.metrics.CostMatrix.constrain`.
        """
        self._metric._validate_parameters(**parameters)

    def parameter_spellings(self, name: str) -> set[str]:
        """
        Return every keyword the parameter *name* can be passed as.

        Parameters
        ----------
        name : str
            A symbol name.

        Returns
        -------
        spellings : set of str
            The symbol name and its aliases.
        """
        return self._metric._parameter_spellings(name)

    def outlier_sensitive_parameters(self) -> dict[str, Literal['positive', 'negative', 'both']]:
        """
        Map each outlier-sensitive parameter to the class whose rows it describes.

        A parameter is outlier-sensitive when it was declared with
        :meth:`~empulse.metrics.CostMatrix.mark_outlier_sensitive`. ``'positive'`` means it appears
        only in the costs of positives, ``'negative'`` only in those of negatives, and ``'both'``
        that it appears in both or that the metric cannot tell.

        Returns
        -------
        parameters : dict of str to {'positive', 'negative', 'both'}
            Keyed by symbol name; see :meth:`parameter_spellings` for the other keywords. Empty when
            the metric declares no outlier-sensitive parameters.
        """
        return self._metric._outlier_sensitive_parameters()

    def loss(
        self,
        y_true: FloatArrayLike,
        y_score: FloatArrayLike,
        *,
        validate: bool = True,
        **parameters: FloatArrayLike | float,
    ) -> float:
        """
        Compute the metric as a value to minimize, whatever its direction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Binary labels.
        y_score : array-like of shape (n_samples,)
            Predicted probabilities, or for a metric with ``Capability.RANKING`` any scores that rank
            the samples.
        validate : bool, default=True
            Whether to check the parameter values; see :meth:`validate_parameters`.
        **parameters
            The metric's parameters.

        Returns
        -------
        loss : float
            The metric's value, negated when the metric is maximized.
        """
        return self._metric._loss(y_true, y_score, validate=validate, **parameters)

    def prepare_count_loss(
        self, *, n_samples: int | None = None, validate: bool = True, **parameters: FloatArrayLike | float
    ) -> CountScoreFn | None:
        """
        Prepare :meth:`loss` for samples grouped by score, such as the leaves of a decision tree.

        The parameters are resolved, and validated if ``validate`` is set, once here rather than on
        every call. The returned function does not check its arguments.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples the groups stand for, which array-valued parameters must match.
        validate : bool, default=True
            Whether to check the parameter values; see :meth:`validate_parameters`.
        **parameters
            The metric's parameters.

        Returns
        -------
        count_loss : callable or None
            ``count_loss(y_score, n_positive, n_negative)`` returns the loss of the samples those groups
            stand for, with one entry per group in each array. ``None`` when the metric needs every
            sample, in which case call :meth:`loss`.
        """
        return self._metric._prepare_count_loss(n_samples=n_samples, validate=validate, **parameters)

    def evaluate_costs(self, *, replace_stochastic: bool = False, **parameters: FloatNDArray | float) -> _Costs:
        """
        Evaluate the cost matrix's four costs.

        Parameters
        ----------
        replace_stochastic : bool, default=False
            Whether to replace stochastic variables by their mean. Without it, a metric with stochastic
            variables raises.
        **parameters
            The metric's parameters.

        Returns
        -------
        fp_cost, fn_cost, tp_cost, tn_cost : float or ndarray of shape (n_samples,)
            The cost of each outcome; a benefit is a negative cost.
        """
        return self._metric._evaluate_costs(replace_stochastic=replace_stochastic, **parameters)

    def decision_costs(self, y_true: FloatNDArray, **parameters: Any) -> _Costs:
        """
        Return the costs whose expected value the metric's decision and training minimize.

        These are :meth:`evaluate_costs` with stochastic variables replaced by their mean, except for a
        :class:`~empulse.metrics.MixtureMetric` whose components' costs the labels weigh. They are what
        a model that trains on per-sample costs, such as a cost-sensitive tree or sampler, should use.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        **parameters
            The metric's parameters.

        Returns
        -------
        fp_cost, fn_cost, tp_cost, tn_cost : float or ndarray of shape (n_samples,)
            The cost of each outcome; a benefit is a negative cost.
        """
        return self._metric._decision_costs(y_true, **parameters)

    def cost_weight(self, y_true: FloatNDArray, **parameters: Any) -> float:
        """
        Return how much :meth:`loss` changes per unit of the expected cost of :meth:`decision_costs`.

        One, except for a metric that divides the cost by something that depends on the labels, such as
        :class:`~empulse.metrics.Savings`.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        **parameters
            The metric's parameters.

        Returns
        -------
        weight : float
            The slope of the loss in the expected cost.
        """
        return self._metric._cost_weight(y_true, **parameters)

    @property
    def decides_from_costs(self) -> bool:
        """
        Whether the optimal decision is the one that minimizes the expected cost of :meth:`decision_costs`.

        A model can then decide each sample from its predicted probability and its costs alone, without
        a threshold learned from the training scores.
        """
        return self._metric._decides_from_costs

    def decision_cost_scale(self, y_true: FloatNDArray, **parameters: Any) -> float:
        """
        Return the average cost of deciding a training sample wrongly instead of rightly.

        For a positive that is ``|fn_cost - tp_cost|``, for a negative ``|fp_cost - tn_cost|``, from
        :meth:`decision_costs`. Measure a hyperparameter with absolute units, such as a minimum
        impurity decrease, a parsimony penalty or a tolerance, against it, and it means the same
        whatever currency the costs are in; adding a constant to the costs of one class leaves it
        unchanged too.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        **parameters
            The metric's parameters.

        Returns
        -------
        scale : float
            The average cost of a wrong decision. One when no decision costs anything, and for a metric
            whose value is a pure number rather than an amount of money (see :attr:`is_unitless`).
        """
        if self.is_unitless:
            return 1.0
        fp_cost, fn_cost, tp_cost, tn_cost = self.decision_costs(y_true, **parameters)
        return objective_scale_from_costs(y_true, -np.asarray(tp_cost), -np.asarray(tn_cost), fp_cost, fn_cost)

    def loss_scale(self, y_true: FloatNDArray, **parameters: Any) -> float:
        """
        Return :meth:`decision_cost_scale` in the units of :meth:`loss`.

        The two differ for a metric whose value is not an amount of money, such as the savings, which
        divide the cost by the baseline's cost.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        **parameters
            The metric's parameters.

        Returns
        -------
        scale : float
            The average cost of a wrong decision, times :meth:`cost_weight`.
        """
        return self.decision_cost_scale(y_true, **parameters) * self.cost_weight(y_true, **parameters)

    def decision_threshold(self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: Any) -> float:
        """
        Return the score above which predicting the positive class costs the samples least.

        Each sample is charged the cost of its own outcome under :meth:`decision_costs`, so
        instance-dependent costs count sample by sample. Samples with equal scores get the same
        decision. The threshold lies halfway between the lowest score predicted positive and the
        highest score predicted negative.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        y_score : ndarray of shape (n_samples,)
            Scores that rank the samples; higher means more likely positive.
        **parameters
            The metric's parameters.

        Returns
        -------
        threshold : float
            Predict positive when ``y_score > threshold``. ``inf`` when predicting every sample
            negative costs least, and ``-inf`` when predicting every sample positive does.

        Raises
        ------
        ValueError
            If ``y_true`` and ``y_score`` differ in length, or a score is NaN or infinite.
        """
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
        _check_shape(y_true, y_score)
        if not np.isfinite(y_score).all():
            _check_y_pred(y_score)  # raises, naming what is not finite
        fp_cost, fn_cost, tp_cost, tn_cost = self.decision_costs(y_true, **parameters)
        positive_cost = np.where(y_true == 1, tp_cost, fp_cost)
        negative_cost = np.where(y_true == 1, fn_cost, tn_cost)
        n_positive, profits = _ranked_profit_curve(negative_cost - positive_cost, y_score)
        k = int(n_positive[np.argmax(profits)])
        if k == 0:
            return float('inf')
        if k == y_score.size:
            return float('-inf')
        ordered = np.sort(y_score)[::-1]
        lowest_positive, highest_negative = ordered[k - 1], ordered[k]
        midpoint = 0.5 * (lowest_positive + highest_negative)
        # Between two adjacent floats the midpoint rounds onto one of them.
        return float(midpoint if highest_negative <= midpoint < lowest_positive else highest_negative)

    def logit_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """
        Build the objective of a logistic regression trained on the metric, with its gradient.

        Requires ``Capability.LOGIT_OBJECTIVE``.

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)
            The design matrix, with a leading column of ones when ``fit_intercept`` is set.
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        C : float
            Inverse regularization strength.
        l1_ratio : float
            The elastic-net mixing parameter, from 0 (L2) to 1 (L1).
        fit_intercept : bool
            Whether the first column of ``features`` is the intercept, which is not regularized.
        **parameters
            The metric's parameters.

        Returns
        -------
        objective : LogitObjective
            The objective to minimize over the weights.
        """
        return self._metric._logit_objective(
            features=features, y_true=y_true, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **parameters
        )

    def logit_value_objective(
        self,
        features: FloatNDArray,
        y_true: FloatNDArray,
        C: float,
        l1_ratio: float,
        fit_intercept: bool,
        **parameters: FloatNDArray | float,
    ) -> LogitObjective:
        """
        Build the objective of a logistic regression, for an optimizer that needs only its value.

        Requires ``Capability.LOGIT_VALUE_OBJECTIVE``. Takes the same arguments as
        :meth:`logit_objective`, and may skip computing the gradient.

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)
            The design matrix, with a leading column of ones when ``fit_intercept`` is set.
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        C : float
            Inverse regularization strength.
        l1_ratio : float
            The elastic-net mixing parameter, from 0 (L2) to 1 (L1).
        fit_intercept : bool
            Whether the first column of ``features`` is the intercept, which is not regularized.
        **parameters
            The metric's parameters.

        Returns
        -------
        objective : LogitObjective
            The objective to minimize over the weights.
        """
        return self._metric._logit_value_objective(
            features=features, y_true=y_true, C=C, l1_ratio=l1_ratio, fit_intercept=fit_intercept, **parameters
        )

    def gradient_boost_objective(
        self, y_true: FloatNDArray, y_score: FloatNDArray, **parameters: FloatNDArray | float
    ) -> tuple[FloatNDArray, FloatNDArray]:
        """
        Compute the gradient and hessian of :meth:`loss` for one boosting round.

        Requires ``Capability.BOOST_OBJECTIVE``.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        y_score : ndarray of shape (n_samples,)
            The booster's current raw scores (logits).
        **parameters
            The metric's parameters.

        Returns
        -------
        gradient, hessian : ndarray of shape (n_samples,)
            With respect to the raw scores.
        """
        return self._metric._gradient_boost_objective(y_true, y_score, **parameters)

    def prepare_boost_objective(self, y_true: FloatNDArray, **parameters: FloatNDArray | float) -> FloatNDArray:
        """
        Compute the part of the boosting gradient that is the same in every round.

        Requires ``Capability.PRECOMPUTED_BOOST_OBJECTIVE``.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Binary labels.
        **parameters
            The metric's parameters.

        Returns
        -------
        gradient_constant : ndarray of shape (n_samples,)
            The constant factor of each sample's gradient.
        """
        return self._metric._prepare_boost_objective(y_true, **parameters)

    def require(self, capability: Capability, *, requester: str, hint: str | None = None) -> None:
        """
        Raise if the metric lacks *capability*.

        Parameters
        ----------
        capability : Capability
            What the caller needs, such as ``Capability.LOGIT_OBJECTIVE``.
        requester : str
            Who needs it, named in the message, usually the model's class name.
        hint : str, optional
            Ends the message, for example with what to use instead.

        Raises
        ------
        ValueError
            If ``capability`` is not in the metric's :attr:`~empulse.metrics.BaseMetric.capabilities`.
        """
        self._metric._require(capability, requester=requester, hint=hint)

    @property
    def is_deterministic(self) -> bool:
        """Whether the metric has no stochastic variables."""
        return self._metric._is_deterministic

    @property
    def has_single_cost_matrix(self) -> bool:
        """
        Whether the metric is computed from one cost matrix, rather than combining several.

        A maximum profit over the class costs is then the maximum profit of the metric itself, which a
        weighted sum of maximum profits is not.
        """
        return self._metric._has_single_cost_matrix

    @property
    def is_unitless(self) -> bool:
        """Whether the metric's value is a pure number, like a ratio of profits, rather than an amount of money."""
        return self._metric._is_unitless
