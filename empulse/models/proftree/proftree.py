import sys
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.metrics import accuracy_score
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntArrayLike, IntNDArray, ParameterConstraint
from ...metrics import Metric
from ...metrics.metric.common import Direction
from ...utils._sklearn_compat import Tags, type_of_target, validate_data
from .._cs_mixin import CostSensitiveMixin
from .evolutionary_tree import EvolutionaryTree

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


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

    max_depth : int, default=10
        Maximum depth of the tree.
        Computation time scales exponentially with depth, be careful with higher values.

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
        max_depth: int | None = 10,
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

        population_size = 10 * X.shape[1] if self.population_size is None else self.population_size
        elite_size = int(population_size * self.elitism_rate)
        tournament_size = max(int(population_size * self.tournament_rate), 2)
        random_state = self.random_state if self.random_state is not None else -1

        if self.loss is None:
            tp_benefit, tn_benefit, fn_cost, fp_cost = self._check_cost_benefits(
                tp_benefit=tp_benefit,
                tn_benefit=tn_benefit,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )
            loss_params = {'tp_benefit': tp_benefit, 'tn_benefit': tn_benefit, 'fn_cost': fn_cost, 'fp_cost': fp_cost}  # noqa: F841

        self.tree_ = EvolutionaryTree()
        self.tree_.fit(
            X=X.astype(np.float32),
            y=y.astype(np.int32),
            pop_size=population_size,
            tournament_size=tournament_size,
            n_elites=elite_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            max_depth=self.max_depth,
            max_generations=self.max_iter,
            patience=self.patience,
            tol=self.tolerance,
            random_state=random_state,
        )

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
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32)
        y_proba = self.tree_.predict_proba(X)
        y_proba = np.vstack((1 - y_proba, y_proba)).T
        return y_proba

    def predict(self, X: FloatArrayLike) -> IntNDArray:
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to ``dtype=np.float32``.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        y_proba = self.predict_proba(X)
        y_pred: FloatNDArray = self.classes_[np.argmax(y_proba, axis=1)]
        return y_pred
