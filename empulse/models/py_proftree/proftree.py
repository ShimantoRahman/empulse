import sys
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntArrayLike, IntNDArray, ParameterConstraint
from ...metrics import Metric
from ...metrics.metric.common import Direction
from ...utils._sklearn_compat import Tags, type_of_target, validate_data
from .._cs_mixin import CostSensitiveMixin
from .node import Node
from .operators import crossover, mutation, selection

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


def _predict_and_evaluate(
    tree: Node,
    X: FloatNDArray,
    y: IntNDArray,
    fitness_function: Callable[..., float],
    complexity_penalty: float,
    is_training: bool = False,
    **loss_params: Any,
) -> Node:
    """
    Evaluate a tree on a training set (in parallel).

    Parameters
    ----------
        tree : Node
            Tree to evaluate.
        X : ndarray of shape (samples, features)
            Training data.
        y : ndarray of shape (samples,)
            Target values.
        fitness_function : Callable
            Fitness function for the genetic algorithm.
        is_training : bool
            If the instances are used for training or predicting.

    Returns
    -------
        Node: The evaluated tree.
    """
    for j in range(X.shape[0]):
        # Predict class for current instance
        tree.predict_one(X[j], y[j], is_training)
    tree.fitness = -fitness_function(tree.y_true, tree.y_pred, **loss_params) + complexity_penalty * tree.size()
    return tree


class ProfTreeClassifier(CostSensitiveMixin, ClassifierMixin, BaseEstimator):
    """
    Profit-driven evolutionary decision tree classifier.

    The ProfTree classifier is a decision tree classifier that is trained using a genetic algorithm.
    The genetic algorithm is used to evolve a population of trees over multiple generations.
    The fitness of each tree is evaluated using a fitness function,
    which is used to select the best trees for crossover and mutation.

    Parameters
    ----------
    tp_benefit : float or array-like, shape=(n_samples,), default=0.0
        Benefit of true positives. If ``float``, then all true positives have the same benefit.
        If array-like, then it is the benefit of each true positive classification.
        Is overwritten if another `tp_benefit` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent benefits to the ``__init__`` method.
            Instead, pass them to the ``fit`` method.

    tn_benefit : float or array-like, shape=(n_samples,), default=0.0
        Benefit of true negatives. If ``float``, then all true negatives have the same benefit.
        If array-like, then it is the benefit of each true negative classification.
        Is overwritten if another `tn_benefit` is passed to the ``fit`` method.

        .. note::
            It is not recommended to pass instance-dependent benefits to the ``__init__`` method.
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

    loss : Metric or None
        Fitness function for the genetic algorithm to maximize.
        If ``None``, the :func:`~empulse.metrics.max_profit_score` is used.

    alpha : float, default=0.0
        Complexity penalty for the fitness function. A way to control overfitting.

        When ``alpha`` is 0.0, the fitness function is not penalized for the amount of nodes in the tree.
        When ``alpha`` is greater than 0.0, the fitness function is penalized for the amount of nodes in the tree.

    patience : int, default=10
        Number of iterations to wait for improvement before stopping early.

    tolerance : float, default=1e-4
        Minimum relative improvement in fitness required to consider a solution better.

    max_iter : int, default=1000
        Maximum number of iterations / number of generations the GA is run.

    max_depth : int or None, default=None
        Maximum depth of the tree.
        If ``None``, then nodes can be expanded until all leaves are pure.

    population_size : int or None, default=None
        Number of decision trees in the population.
        If ``None``, population size is set to ``10 * n_features``.

    crossover_rate : float, default=1.0
        Probability of crossover. Must be in [0, 1].

    mutation_rate : float, default=0.1
        Probability of mutation. Must be in [0, 1].

    elitism_rate : float, default=0.05
        Fraction of the population that transferred to the next generation without change.
        Must be in [0, 1].

    tournament_rate : float, default=0.01
        Fraction of the population included during tournament selection. Must be in [0, 1].
        Will be at least 2 trees.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    random_state : np.random.RandomState, int or None, default=None
        Controls the randomness of the estimator.
        To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Sklearn Glossary <random_state>` for details.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'loss': [Metric, None],
        'alpha': [Interval(Real, 0, None, closed='both')],
        'patience': [Interval(Integral, 1, None, closed='left')],
        'tolerance': [Interval(Real, 0, None, closed='right')],
        'max_depth': [Interval(Integral, 1, None, closed='left'), None],
        'max_iter': [Interval(Integral, 1, None, closed='left')],
        'population_size': [Interval(Integral, 2, None, closed='left'), None],
        'crossover_rate': [Interval(Real, 0, 1, closed='both')],
        'mutation_rate': [Interval(Real, 0, 1, closed='both')],
        'elitism_rate': [Interval(Real, 0, 1, closed='both')],
        'tournament_rate': [Interval(Real, 0, 1, closed='both')],
        'n_jobs': [Interval(Integral, 1, None, closed='left')],
        'random_state': ['random_state'],
    }

    def __init__(
        self,
        *,
        tp_benefit: FloatArrayLike | float = 0.0,
        tn_benefit: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
        alpha: float = 0.0,
        patience: int = 10,
        tolerance: float = 1e-4,
        max_depth: int | None = None,
        max_iter: int = 100,
        population_size: int | None = None,
        crossover_rate: float = 1.0,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.05,
        tournament_rate: int = 0.01,
        n_jobs: int = 1,
        random_state: np.random.RandomState | int | None = None,
    ):
        self.tp_benefit = tp_benefit
        self.tn_benefit = tn_benefit
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
        self.alpha = alpha
        self.patience = patience
        self.tolerance = tolerance
        self.max_depth = max_depth
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_rate = tournament_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._tree = None
        self._best_fitness = []
        self._avg_fitness = []

    def _more_tags(self) -> dict[str, bool]:
        return {
            'binary_only': True,
            'poor_score': True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def _get_loss(self):
        if self.loss is not None:
            if self.loss.direction is Direction.MAXIMIZE:
                loss = self.loss
            else:
                loss = lambda *args, **kwargs: -self.loss(*args, **kwargs)
        else:
            # loss = max_profit_score
            loss = accuracy_score
        return loss

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: IntArrayLike,
        *,
        tp_benefit: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_benefit: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> Self:
        """
        Fit a tree to a training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        tp_benefit : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Benefit of true positives. If ``float``, then all true positives have the same benefit.
            If array-like, then it is the benefit of each true positive classification.

        tn_benefit : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Benefit of true negatives. If ``float``, then all true negatives have the same benefit.
            If array-like, then it is the benefit of each true negative classification.

        fp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        fn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        loss_params : Any
            Additional parameter to be passed to the loss function.

        Returns
        -------
            Node: The fitted tree.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
            raise ValueError(
                f'Unknown label type: Only binary classification is supported. The type of the target is {y_type}.'
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        y = np.where(y == self.classes_[1], 1, 0)

        rng = check_random_state(self.random_state)
        population_size = 10 * X.shape[1] if self.population_size is None else self.population_size
        elite_size = int(population_size * self.elitism_rate)
        tournament_size = max(int(population_size * self.tournament_rate), 2)

        if self.loss is None:
            tp_benefit, tn_benefit, fn_cost, fp_cost = self._check_cost_benefits(
                tp_benefit=tp_benefit,
                tn_benefit=tn_benefit,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            # loss_params = {
            #     'tp_benefit': tp_benefit,
            #     'tn_benefit': tn_benefit,
            #     'fn_cost': fn_cost,
            #     'fp_cost': fp_cost
            # }
            loss_params = {}

        self.att_indexes_ = np.arange(X.shape[1])
        self.att_values_ = {
            i: [
                (min_val + max_val) / 2
                for min_val, max_val in zip(
                    sorted(np.unique(X[:, i]))[:-1], sorted(np.unique(X[:, i]))[1:], strict=True
                )
            ]
            for i in range(X.shape[1])
        }
        self.att_values_[-1] = sorted(np.unique(y))
        self.n_classes_ = len(self.att_values_[-1])

        # Generation of initial population
        node = Node()
        population = []
        for _ in range(population_size):
            population.append(
                node.make_node(
                    max_depth=self.max_depth,
                    random=rng,
                    att_indexes=self.att_indexes_,
                    att_values=self.att_values_,
                    class_count=self.n_classes_,
                )
            )

        best_fitness = float('inf')
        no_improvement_count = 0

        for i in range(self.max_iter + 1):
            self.n_iter_ = i

            # Clear previous evaluation
            for tree in population:
                tree.clear_evaluation()

            # Evaluation of population
            population = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_and_evaluate)(tree, X, y, self._get_loss(), self.alpha, True, **loss_params)
                for tree in population
            )

            # Sort population by fitness
            population.sort(key=lambda x: x.fitness, reverse=False)

            # Log best and average fitness
            current_best_fitness = population[0].fitness
            self._best_fitness.append(current_best_fitness)
            self._avg_fitness.append(sum(tree.fitness for tree in population) / len(population))

            # Check for improvement
            if best_fitness == float('inf'):
                best_fitness = current_best_fitness
            else:
                relative_improvement = (best_fitness - current_best_fitness) / abs(best_fitness)
                if relative_improvement > self.tolerance:
                    best_fitness = current_best_fitness
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping if no improvement
                if no_improvement_count >= self.patience:
                    break

            if i != self.max_iter:
                elites = population[:elite_size]

                # Descendant generation
                descendants = []
                for _ in range(0, len(population), 2):
                    parent1, parent2 = selection(
                        population=population, selection_tournament_size=tournament_size, random=rng
                    )

                    child1 = crossover(tree1=parent1, tree2=parent2, random=rng)
                    child2 = crossover(tree1=parent2, tree2=parent1, random=rng)

                    if rng.random() < self.mutation_rate:
                        child1 = mutation(
                            root=child1,
                            att_indexes=self.att_indexes_,
                            att_values=self.att_values_,
                            class_count=self.n_classes_,
                            random=rng,
                        )
                    if rng.random() < self.mutation_rate:
                        child2 = mutation(
                            root=child2,
                            att_indexes=self.att_indexes_,
                            att_values=self.att_values_,
                            class_count=self.n_classes_,
                            random=rng,
                        )

                    # Add new trees to descendant population
                    descendants.extend([child1, child2])

                # Elites + descendants
                descendants.sort(key=lambda x: x.fitness, reverse=False)
                descendants = elites + descendants[: population_size - elite_size]

                # Replace old population with new population
                population = descendants

        self._tree = population[0]

        return self

    def predict(self, X: FloatArrayLike) -> IntNDArray:
        """
        Predict classes for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
            list: Predicted classes.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = []
        for i in range(X.shape[0]):
            index = self._tree.predict_one(X[i])
            y_pred.append(self.att_values_[-1][index])
        return np.array(y_pred)

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        y_proba: FloatNDArray = ...
        return y_proba
