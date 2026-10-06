from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, ClassVar, Self

import numpy as np
from joblib import effective_n_jobs
from scipy.special import expit
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..._common._sklearn_compat import validate_data
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, Capability, MaxProfit
from .._base import CostSensitiveClassifier, MetricStrategyFactory
from .._base.cost_scale import decision_cost_scale
from .._base.ensemble_weighting import subset_loss_params
from ._symbolic import ParetoPoint, Program, ProgramSpace, SearchSettings, evolve, get_function


def _score_location_and_spread(y_score: FloatNDArray) -> tuple[float, float]:
    """Return the median of the finite scores and a robust estimate of their standard deviation."""
    y_score = y_score[np.isfinite(y_score)]
    if y_score.size == 0:
        return 0.0, 1.0
    q1, median, q3 = np.percentile(y_score, [25, 50, 75])
    # The interquartile range of a normal distribution is 1.349 standard deviations.
    for spread in ((q3 - q1) / 1.349, np.std(y_score)):
        if spread > 0 and np.isfinite(spread):
            return float(median), float(spread)
    return float(median), 1.0


class _LossFitness:
    """Loss of a program's output on rows of the training data, with ``inf`` for outputs that cannot be scored."""

    def __init__(self, loss: BaseMetric, y: IntNDArray, loss_params: dict[str, Any], scores_rank_only: bool) -> None:
        self.loss = loss
        self.y = y
        self.loss_params = loss_params
        self.scores_rank_only = scores_rank_only

    def __call__(self, y_pred: FloatNDArray, rows: IntNDArray | None) -> float:
        if not np.isfinite(y_pred).all():
            return np.inf
        y_score = y_pred if self.scores_rank_only else expit(y_pred)
        if rows is None:
            value = self.loss._loss(self.y, y_score, validate=False, **self.loss_params)
        else:
            loss_params = subset_loss_params(self.loss_params, rows, self.y.shape[0])
            value = self.loss._loss(self.y[rows], y_score, validate=False, **loss_params)
        return float(value) if np.isfinite(value) else np.inf


class ProfSRClassifier(CostSensitiveClassifier):
    """
    Profit-driven symbolic regression classifier.

    Maximizes an empirical cost-sensitive/value-driven metric by evolving a population of
    mathematical expressions through genetic programming (symbolic regression) [1]_.
    The predicted score of an expression is squashed through the logistic function to obtain
    a probability estimate, which is used to evaluate the loss function. A loss that depends only on
    how the samples are ranked (the ``RANKING`` member of :class:`~empulse.metrics.Capability`), such as the default
    maximum profit, is evaluated on the expressions' outputs directly. The scale of those outputs is then arbitrary,
    so :meth:`decision_function` centers them on their median on the training data and divides them by their spread
    there before :meth:`predict_proba` squashes them, which keeps large outputs from rounding to the same probability.

    The size of an expression is limited by ``max_length`` and penalized by ``parsimony_coefficient``,
    which keeps the fitted formula readable. The search keeps the best expression found at every length,
    available as ``pareto_front_``, and the constants of the best expressions are tuned periodically
    with the Nelder-Mead simplex method.

    Read more in the :ref:`User Guide <profsr>`.

    Parameters
    ----------
    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true negatives. If ``float``, then all true negatives have the same cost.
        If array-like, then it is the cost of each true negative classification.
        Is overwritten if another `tn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fn_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false negatives. If ``float``, then all false negatives have the same cost.
        If array-like, then it is the cost of each false negative classification.
        Is overwritten if another `fn_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent costs to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    loss : :class:`~empulse.metrics.BaseMetric` or None, default=None
        Fitness function for the genetic programming algorithm to optimize.

        If :class:`~empulse.metrics.BaseMetric`, metric parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.ProfSRClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    max_iter : int, default=50
        Maximum number of generations to evolve the population of expressions.

    patience : int or None, default=None
        Number of consecutive generations whose best loss improved by less than ``tolerance``
        after which the search stops.
        If ``None``, the search runs for ``max_iter`` generations.

    tolerance : float, default=1e-4
        Improvement of the best loss below which a generation counts towards ``patience``, as a
        fraction of the average cost of a wrong decision on the training data.

    max_time : float or None, default=None
        Number of seconds after which the search stops once the current generation has finished.
        If ``None``, there is no time limit.

        .. note::
            With a time limit, the fitted expression depends on the speed of the machine.

    population_size : int, default=1000
        Number of expressions in each generation.

    tournament_size : int, default=20
        Number of expressions that compete to become a parent. The one with the lowest
        regularized loss wins; larger values raise the selection pressure.

    function_set : tuple of str, default=('add', 'sub', 'mul', 'div', 'exp', 'log', 'sig')
        Operators an expression may use. The available operators are:

        - ``'add'``, ``'sub'``, ``'mul'``, ``'max'``, ``'min'``: binary arithmetic operators.
        - ``'div'``: protected division, which is 1 where the denominator is smaller than 0.001 in magnitude.
        - ``'log'``: protected logarithm of the absolute value, which is 0 where the input is smaller than 0.001
          in magnitude.
        - ``'inv'``: protected inverse, which is 0 where the input is smaller than 0.001 in magnitude.
        - ``'sqrt'``: square root of the absolute value.
        - ``'abs'``, ``'neg'``, ``'sin'``, ``'cos'``, ``'tan'``: unary functions.
        - ``'exp'``: exponential, with the input capped at 100.
        - ``'sig'``: logistic function.

    max_length : int or None, default=20
        Maximum number of nodes (operators, features and constants) in an expression.
        Expressions that would become longer are never created.
        If ``None``, the length is only discouraged by ``parsimony_coefficient``.

    parsimony_coefficient : float, default=0.0003
        Constant that penalizes the length of an expression when selecting parents:
        the regularized loss is the loss plus ``parsimony_coefficient`` times the number of nodes,
        times the average cost of a wrong decision on the training data, so that the penalty does not
        depend on the units of the costs.
        Larger values favour shorter expressions. The same penalty selects the fitted expression from the
        Pareto front.

    init_depth : tuple of two ints, default=(2, 6)
        Range of the maximum depth of the initial expressions. Each expression draws its own maximum depth
        from this range.

    init_method : {'grow', 'full', 'half_and_half'}, default='half_and_half'
        How the initial expressions are grown.

        - ``'grow'``: nodes are chosen at random from both operators and terminals,
          which allows for shallower and asymmetrical expressions.
        - ``'full'``: operators are chosen until the maximum depth is reached, which makes for bushy expressions.
        - ``'half_and_half'``: half of the expressions are grown with each method (ramped half-and-half).

    const_range : tuple of two floats or None, default=(-1.0, 1.0)
        Range random constants are drawn from. If ``None``, expressions contain no constants.

    constant_rate : float or None, default=0.5
        Probability that a leaf of a new expression is a constant rather than a feature.
        Constants are what the tuning refines, so they need to be common enough for it to have something to
        work on. If ``None``, a constant is one more choice next to the features, with probability
        ``1 / (n_features + 1)``. Ignored without ``const_range``.

    crossover_rate : float, default=0.9
        Probability of creating an expression by exchanging a random subtree between two parents.

        The four variation rates (``crossover_rate``, ``mutate_subtree_rate``, ``hoist_rate``,
        ``mutate_point_rate``) are mutually exclusive and must sum to at most 1.
        The remaining probability is used to copy a parent unchanged.

    mutate_subtree_rate : float, default=0.01
        Probability of creating an expression by replacing a random subtree of a parent by a new random subtree.

    hoist_rate : float, default=0.01
        Probability of creating an expression by replacing a random subtree of a parent
        by one of its own subtrees, which shrinks the expression.

    mutate_point_rate : float, default=0.01
        Probability of creating an expression by replacing random nodes of a parent by
        other nodes of the same arity, which keeps the shape of the expression.
        Each node is replaced with probability ``point_replace_rate``.

    point_replace_rate : float, default=0.05
        Probability that a node is replaced in a point mutation.

    max_samples : float, default=1.0
        Fraction of the training samples each expression is scored on. Each expression gets its own random
        batch, which makes every generation cheaper on large datasets at the cost of a noisier loss.
        With a value below 1, the expressions on the Pareto front are scored on all samples after the search,
        and ``run_details_['best_oob_fitness']`` holds the loss of the best expression on the samples left
        out of its batch.

    n_tuned_programs : int, default=10
        Number of distinct expressions whose constants are tuned each time tuning runs. The expressions with
        the lowest loss that contain constants are tuned. If 0, constants are only changed by the genetic
        operators.

    tuning_interval : int, default=5
        Number of generations between tuning runs.

    tuning_max_iter : int, default=50
        Maximum number of Nelder-Mead iterations for each tuned expression. The structure of an expression
        is held fixed during tuning, and tuning never makes an expression worse.

    n_jobs : int or None, default=1
        Number of jobs scoring the population in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        The fitted expression does not depend on ``n_jobs``.

    random_state : int, :class:`numpy:numpy.random.RandomState` or None, default=None
        Controls the randomness of the estimator.
        To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Sklearn Glossary <sklearn:random_state>` for details.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    n_features_in_ : int
        Number of features seen during :term:`fit <sklearn:fit>`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit <sklearn:fit>`. Defined only when `X`
        has feature names that are all strings.

    program_ : Program
        The fitted expression: the expression on ``pareto_front_`` with the lowest regularized loss.
        ``str(program_)`` prints it as nested function calls, using the feature names if there are any.

    pareto_front_ : list of ParetoPoint
        The shortest expressions that reached a lower loss than every shorter expression found during the search,
        sorted by length. Each point has a ``length``, a ``loss`` and a ``program``.

    run_details_ : dict of lists
        For every generation: the average length and loss of the population, the length and loss of the best
        expression, its out-of-batch loss, the time the generation took, and the number of expressions that
        had to be scored (an expression seen before in the run is not scored again).

    n_iter_ : int
        Number of generations evolved.

    score_center_ : float
        Output of ``program_`` that :meth:`decision_function` maps to 0 and :meth:`predict_proba` to one half:
        the median output on the training data for a loss that only ranks the samples, 0 otherwise.

    score_scale_ : float
        Spread by which :meth:`decision_function` divides the centered outputs of ``program_``:
        their interquartile range on the training data, normalized to the standard deviation of a normal
        distribution (their standard deviation if that is 0), for a loss that only ranks the samples, 1 otherwise.

    References
    ----------
    .. [1] Aliaga, Samuel and Vairetti, Carla and Maldonado, Sebastián,
       Profit-driven Symbolic Regression for Customer Churn Prediction (August 09, 2026).
       Available at SSRN: https://ssrn.com/abstract=7256098 or http://dx.doi.org/10.2139/ssrn.7256098

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfSRClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfSRClassifier(max_iter=10, population_size=100, random_state=42)
        model.fit(X, y, tp_cost=-200, fp_cost=10)
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'max_iter': [Interval(Integral, 1, None, closed='left')],
        'patience': [Interval(Integral, 1, None, closed='left'), None],
        'tolerance': [Interval(Real, 0, None, closed='left')],
        'max_time': [Interval(Real, 0, None, closed='neither'), None],
        'population_size': [Interval(Integral, 1, None, closed='left')],
        'tournament_size': [Interval(Integral, 1, None, closed='left')],
        'function_set': [list, tuple],
        'max_length': [Interval(Integral, 2, None, closed='left'), None],
        'parsimony_coefficient': [Interval(Real, 0, None, closed='left')],
        'init_depth': [tuple],
        'init_method': [StrOptions({'grow', 'full', 'half_and_half'})],
        'const_range': [tuple, None],
        'constant_rate': [Interval(Real, 0, 1, closed='both'), None],
        'crossover_rate': [Interval(Real, 0, 1, closed='both')],
        'mutate_subtree_rate': [Interval(Real, 0, 1, closed='both')],
        'hoist_rate': [Interval(Real, 0, 1, closed='both')],
        'mutate_point_rate': [Interval(Real, 0, 1, closed='both')],
        'point_replace_rate': [Interval(Real, 0, 1, closed='both')],
        'max_samples': [Interval(Real, 0, 1, closed='right')],
        'n_tuned_programs': [Interval(Integral, 0, None, closed='left')],
        'tuning_interval': [Interval(Integral, 1, None, closed='left')],
        'tuning_max_iter': [Interval(Integral, 1, None, closed='left')],
        'n_jobs': [Interval(Integral, 1, None, closed='left'), Interval(Integral, None, -1, closed='right'), None],
        'random_state': ['random_state'],
    }
    _default_metric_strategy: ClassVar[MetricStrategyFactory] = MaxProfit

    def __init__(
        self,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: BaseMetric | None = None,
        max_iter: int = 50,
        patience: int | None = None,
        tolerance: float = 1e-4,
        max_time: float | None = None,
        population_size: int = 1000,
        tournament_size: int = 20,
        function_set: Sequence[str] = ('add', 'sub', 'mul', 'div', 'exp', 'log', 'sig'),
        max_length: int | None = 20,
        parsimony_coefficient: float = 0.0003,
        init_depth: tuple[int, int] = (2, 6),
        init_method: str = 'half_and_half',
        const_range: tuple[float, float] | None = (-1.0, 1.0),
        constant_rate: float | None = 0.5,
        crossover_rate: float = 0.9,
        mutate_subtree_rate: float = 0.01,
        hoist_rate: float = 0.01,
        mutate_point_rate: float = 0.01,
        point_replace_rate: float = 0.05,
        max_samples: float = 1.0,
        n_tuned_programs: int = 10,
        tuning_interval: int = 5,
        tuning_max_iter: int = 50,
        n_jobs: int | None = 1,
        random_state: np.random.RandomState | int | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.patience = patience
        self.tolerance = tolerance
        self.max_time = max_time
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.function_set = function_set
        self.max_length = max_length
        self.parsimony_coefficient = parsimony_coefficient
        self.init_depth = init_depth
        self.init_method = init_method
        self.const_range = const_range
        self.constant_rate = constant_rate
        self.crossover_rate = crossover_rate
        self.mutate_subtree_rate = mutate_subtree_rate
        self.hoist_rate = hoist_rate
        self.mutate_point_rate = mutate_point_rate
        self.point_replace_rate = point_replace_rate
        self.max_samples = max_samples
        self.n_tuned_programs = n_tuned_programs
        self.tuning_interval = tuning_interval
        self.tuning_max_iter = tuning_max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def _check_search_space(self, n_features: int) -> ProgramSpace:
        if not self.function_set:
            raise ValueError('function_set must contain at least one function.')
        function_set = [get_function(name) for name in self.function_set]

        rates = self.crossover_rate + self.mutate_subtree_rate + self.hoist_rate + self.mutate_point_rate
        if rates > 1 + 1e-12:
            raise ValueError(
                'The sum of crossover_rate, mutate_subtree_rate, hoist_rate and mutate_point_rate '
                f'should total to 1.0 or less, got {rates}.'
            )
        if len(self.init_depth) != 2 or not 1 <= self.init_depth[0] <= self.init_depth[1]:
            raise ValueError(
                f'init_depth should be a tuple (min_depth, max_depth) with 1 <= min_depth <= max_depth, '
                f'got {self.init_depth}.'
            )
        if self.const_range is not None and (len(self.const_range) != 2 or self.const_range[0] > self.const_range[1]):
            raise ValueError(
                f'const_range should be a tuple (low, high) with low <= high or None, got {self.const_range}.'
            )

        space = ProgramSpace(
            function_set=function_set,
            n_features=n_features,
            const_range=self.const_range,
            init_depth=self.init_depth,
            init_method=self.init_method,
            point_replace_rate=self.point_replace_rate,
            max_length=self.max_length,
            constant_rate=self.constant_rate,
        )
        if self.max_length is not None and self.max_length < space.min_length:
            raise ValueError(
                f'max_length={self.max_length} is smaller than the shortest expression the function_set can '
                f'form ({space.min_length} nodes).'
            )
        return space

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        X = np.asfortranarray(X, dtype=np.float64)
        space = self._check_search_space(X.shape[1])

        # A loss that only ranks the samples scores the expressions' outputs directly: squashing them into
        # probabilities would not change their ranking, only cost time and tie the largest outputs,
        # which rounding makes equal probabilities.
        fitness = _LossFitness(loss, y, loss_params, scores_rank_only=Capability.RANKING in loss.capabilities)
        try:  # catch an issue with the loss function before it is evaluated thousands of times
            fitness(np.linspace(-1.0, 1.0, y.size), None)
        except (TypeError, ValueError) as e:
            raise ValueError(f'The loss function {loss} threw an error when evaluating the function.') from e

        settings = SearchSettings(
            max_iter=self.max_iter,
            population_size=self.population_size,
            tournament_size=self.tournament_size,
            crossover_rate=self.crossover_rate,
            mutate_subtree_rate=self.mutate_subtree_rate,
            hoist_rate=self.hoist_rate,
            mutate_point_rate=self.mutate_point_rate,
            parsimony_coefficient=self.parsimony_coefficient,
            max_samples=self.max_samples,
            n_tuned_programs=self.n_tuned_programs,
            tuning_interval=self.tuning_interval,
            tuning_max_iter=self.tuning_max_iter,
            patience=self.patience,
            tolerance=self.tolerance,
            max_time=self.max_time,
            n_jobs=effective_n_jobs(self.n_jobs),
            loss_scale=decision_cost_scale(loss, y, **loss_params),
        )
        result = evolve(X, fitness, space, settings, self.random_state)

        feature_names = getattr(self, 'feature_names_in_', None)
        if feature_names is not None:
            for program in [result.program, *(point.program for point in result.pareto_front)]:
                program.feature_names = list(feature_names)
        self.program_: Program = result.program
        self.pareto_front_: list[ParetoPoint] = result.pareto_front
        self.run_details_ = result.run_details
        self.n_iter_ = len(result.run_details['generation'])
        self.score_center_, self.score_scale_ = 0.0, 1.0
        if fitness.scores_rank_only:
            self.score_center_, self.score_scale_ = _score_location_and_spread(self.program_.execute(X))

        return self

    def decision_function(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Compute the centered and scaled output of the fitted expression.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Features.

        Returns
        -------
        y_score : 1D numpy.ndarray, shape=(n_samples,)
            Output of ``program_`` for each sample, minus ``score_center_``, divided by ``score_scale_``.
            Positive values predict the positive class.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_score: FloatNDArray = (
            self.program_.execute(np.asarray(X, dtype=np.float64)) - self.score_center_
        ) / self.score_scale_
        return y_score

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, 2)
            Predicted probabilities.
        """
        y_score = expit(self.decision_function(X))
        return np.vstack((1 - y_score, y_score)).T
