from functools import partial
from numbers import Integral, Real
from typing import Any, ClassVar, Self

import numpy as np
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import BaseMetric, MaxProfit
from ...metrics.metric.common import Direction
from ..csclassifier import CostSensitiveClassifier, MetricStrategyFactory
from .evolutionary_tree import EvolutionaryTree

MAX_INT = 2147483647


class ProfTreeClassifier(CostSensitiveClassifier):
    """
    Profit-driven evolutionary decision tree classifier.

    The ProfTree classifier is a decision tree classifier that is trained using a genetic algorithm.
    The genetic algorithm is used to evolve a population of trees over multiple generations.
    The fitness of each tree is evaluated using a fitness function,
    which is used to select the best trees for crossover and mutation.

    Parameters
    ----------
    tp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of true positives. If ``float``, then all true positives have the same cost.
        If array-like, then it is the cost of each true positive classification.
        Is overwritten if another `tp_cost` is passed to the ``fit`` method.

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

    fp_cost : float or array-like, shape=(n_samples,), default=0.0
        Cost of false positives. If ``float``, then all false positives have the same cost.
        If array-like, then it is the cost of each false positive classification.
        Is overwritten if another `fp_cost` is passed to the ``fit`` method.

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

    loss : BaseMetric or None
        Fitness function for the genetic algorithm to maximize.
        If ``None``, the :func:`~empulse.metrics.max_profit_score` is used.

    alpha : float, default=0.0
        Complexity penalty for the fitness function. A way to control overfitting.

        When ``alpha`` is 0.0, the fitness function is not penalized for the amount of nodes in the tree.
        When ``alpha`` is greater than 0.0, the fitness function is penalized for the amount of nodes in the tree.

    patience : int, default=100
        Number of iterations to wait for improvement before stopping early.

    tolerance : float, default=1e-4
        Minimum relative improvement in fitness required to consider a solution better.

    max_iter : int, default=1000
        Maximum number of iterations / number of generations the GA is run.

    max_depth : int or None, default=10
        Maximum depth of the tree.
        Computation time scales exponentially with depth, be careful with higher values.

    min_samples_split : int or float, default=20
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=7
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    population_size : int or None, default=None
        Number of decision trees in the population.
        If ``None``, population_size is set to ``10 * n_features``.
        Will be at least 2 trees.

    crossover_rate : float, default=0.2
        Probability of crossover. Must be in [0, 1].

        Variation operator is mutually exclusive with variation operators
        (``crossover_rate``, ``grow_rate``, ``prune_rate``, ``mutate_split_rate``, ``mutate_value_rate``).
        All probabilities must sum to 1.

    grow_rate : float, default=0.2
        Probability to add a randomly generated split rule to a leaf node. Must be in [0, 1].

        Variation operator is mutually exclusive with variation operators
        (``crossover_rate``, ``grow_rate``, ``prune_rate``, ``mutate_split_rate``, ``mutate_value_rate``).
        All probabilities must sum to 1.

    prune_rate : float, default=0.2
        Probability to remove a randomly selected split rule from an internal node with two leaf nodes as children.
        Must be in [0, 1].

        Variation operator is mutually exclusive with variation operators
        (``crossover_rate``, ``grow_rate``, ``prune_rate``, ``mutate_split_rate``, ``mutate_value_rate``).
        All probabilities must sum to 1.

    mutate_split_rate : float, default=0.2
        Probability to change the feature and feature value of a random split in the tree. Must be in [0, 1].

        Variation operator is mutually exclusive with variation operators
        (``crossover_rate``, ``grow_rate``, ``prune_rate``, ``mutate_split_rate``, ``mutate_value_rate``).
        All probabilities must sum to 1.

    mutate_value_rate : float, default=0.2
        Probability to change only the feature value of a random split in the tree. Must be in [0, 1].

        Variation operator is mutually exclusive with variation operators
        (``crossover_rate``, ``grow_rate``, ``prune_rate``, ``mutate_split_rate``, ``mutate_value_rate``).
        All probabilities must sum to 1.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    random_state : np.random.RandomState, int or None, default=None
        Controls the randomness of the estimator.
        To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Sklearn Glossary <sklearn:random_state>` for details.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'alpha': [Interval(Real, 0, None, closed='both')],
        'patience': [Interval(Integral, 1, None, closed='left')],
        'tolerance': [Interval(Real, 0, None, closed='right')],
        'max_depth': [Interval(Integral, 1, None, closed='left'), None],
        'min_samples_split': [
            Interval(Integral, 2, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='right'),
        ],
        'min_samples_leaf': [
            Interval(Integral, 1, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='neither'),
        ],
        'max_iter': [Interval(Integral, 1, None, closed='left')],
        'population_size': [Interval(Integral, 2, None, closed='left'), None],
        'crossover_rate': [Interval(Real, 0, 1, closed='both')],
        'grow_rate': [Interval(Real, 0, 1, closed='both')],
        'prune_rate': [Interval(Real, 0, 1, closed='both')],
        'mutate_split_rate': [Interval(Real, 0, 1, closed='both')],
        'mutate_value_rate': [Interval(Real, 0, 1, closed='both')],
        'n_jobs': [Interval(Integral, 1, None, closed='left')],
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
        alpha: float = 0.0,
        patience: int = 100,
        tolerance: float = 1e-4,
        max_depth: int | None = 10,
        min_samples_split: float = 20,
        min_samples_leaf: float = 7,
        max_iter: int = 1000,
        population_size: int | None = None,
        crossover_rate: float = 0.2,
        grow_rate: float = 0.2,
        prune_rate: float = 0.2,
        mutate_split_rate: float = 0.2,
        mutate_value_rate: float = 0.2,
        n_jobs: int = 1,
        random_state: np.random.RandomState | int | None = None,
    ):
        self.alpha = alpha
        self.patience = patience
        self.tolerance = tolerance
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.population_size = population_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.grow_rate = grow_rate
        self.prune_rate = prune_rate
        self.mutate_split_rate = mutate_split_rate
        self.mutate_value_rate = mutate_value_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: BaseMetric, **loss_params: Any) -> Self:
        """
        Fit a tree to a training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        loss : BaseMetric
            Loss to be optimized.

        loss_params : Any
            Additional parameter to be passed to the loss function.

        Returns
        -------
            Node: The fitted tree.
        """
        n_samples = X.shape[0]
        population_size = 10 * X.shape[1] if self.population_size is None else self.population_size
        if self.min_samples_split < 1:
            min_samples_split = np.ceil(self.min_samples_split * n_samples)
        else:
            min_samples_split = self.min_samples_split
        min_samples_split = max(min_samples_split, 2)

        if self.min_samples_leaf < 1:
            min_samples_leaf = np.ceil(self.min_samples_leaf * n_samples)
        else:
            min_samples_leaf = self.min_samples_leaf
        min_samples_leaf = max(min_samples_leaf, 1)

        total_probability = (
            self.crossover_rate + self.grow_rate + self.prune_rate + self.mutate_split_rate + self.mutate_value_rate
        )
        crossover_rate = self.crossover_rate / total_probability
        grow_rate = self.grow_rate / total_probability
        prune_rate = self.prune_rate / total_probability
        mutate_split_rate = self.mutate_split_rate / total_probability
        mutate_value_rate = self.mutate_value_rate / total_probability

        if isinstance(self.random_state, np.random.RandomState):
            random_state = int(self.random_state.randint(low=0, high=2**31 - 1))
        else:
            random_state = int(self.random_state) if self.random_state is not None else -1

        self.tree_ = EvolutionaryTree()
        loss_ = self._get_metric_loss()
        if loss_ is not None and not isinstance(loss_, BaseMetric):
            raise ValueError(f'Unknown loss function: {loss_}.')

        common_kwargs: dict[str, Any] = {
            'X': X.astype(np.float32),
            'y': y.astype(np.int32),
            'pop_size': int(population_size),
            'crossover_rate': float(crossover_rate),
            'grow_rate': float(grow_rate),
            'prune_rate': float(prune_rate),
            'mutate_split_rate': float(mutate_split_rate),
            'mutate_value_rate': float(mutate_value_rate),
            'max_depth': int(self.max_depth) if self.max_depth is not None else MAX_INT,
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            'alpha': float(self.alpha),
            'max_generations': int(self.max_iter),
            'patience': int(self.patience),
            'tol': float(self.tolerance),
            'random_state': random_state,
        }

        # `_prepare_class_costs` supports exactly the two cases handled by `fit_max_profit`
        # (no custom loss, or a deterministic MaxProfit metric); a stochastic MaxProfit metric or
        # any other strategy needs the full custom fitness-function path instead.
        use_fit_max_profit = loss_ is None or (isinstance(loss_.strategy, MaxProfit) and loss_._is_deterministic)

        if use_fit_max_profit:
            tp_benefit, tn_benefit, fp_cost, fn_cost = self._prepare_class_costs(loss_params)
            self.tree_.fit_max_profit(
                **common_kwargs, tp_benefit=tp_benefit, tn_benefit=tn_benefit, fp_cost=fp_cost, fn_cost=fn_cost
            )
        else:
            # `use_fit_max_profit` is only False when `loss_ is None` is False, i.e. `loss_` is a
            # BaseMetric (either a non-MaxProfit strategy, or a stochastic MaxProfit metric).
            assert loss_ is not None
            if loss_.direction is Direction.MAXIMIZE:
                fitness_fn = lambda *args, **kwargs: -loss_(*args, **kwargs)
            else:
                fitness_fn = loss_
            fitness_fn = partial(fitness_fn, **loss_params)

            y_proba = check_random_state(self.random_state).random(y.size).astype(np.float32)
            try:  # catch issue with the loss function before it goes into C world
                fitness_fn(y.astype(np.int32), y_proba)
            except (TypeError, ValueError) as e:
                raise ValueError(f'The loss function {loss_} threw an error when evaluating the function.') from e
            self.tree_.fit_custom(**common_kwargs, fitness_function=fitness_fn)

        self.n_iter_ = self.tree_.n_generations

        return self

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same class in a leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_ <sklearn:classes_>`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32)
        y_proba = self.tree_.predict_proba(X)
        y_proba = np.vstack((1 - y_proba, y_proba)).T
        return y_proba
