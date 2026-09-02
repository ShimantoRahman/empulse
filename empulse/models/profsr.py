from numbers import Integral, Real
from typing import Any, ClassVar, Self

import numpy as np
from scipy.special import expit
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from .._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ..metrics import MaxProfit, Metric
from ..metrics.metric.common import Direction
from .csclassifier import CostSensitiveClassifier, MetricStrategyFactory


class ProfSRClassifier(CostSensitiveClassifier):
    """
    Profit-driven symbolic regression classifier.

    Maximizes an empirical cost-sensitive/value-driven metric by evolving a population of
    mathematical expressions through genetic programming (symbolic regression).
    The predicted score of a program is squashed through the logistic function to obtain
    a probability estimate, which is used to evaluate the loss function.

    Read more in the :ref:`User Guide <user_defined_value_metric>`.

    .. note::
        This classifier requires the optional `gplearn <https://gplearn.readthedocs.io/>`_ dependency.
        Install it with ``pip install empulse[symbolic]`` or ``pip install gplearn``.

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

    loss : :class:`empulse.metrics.Metric` or None, default=None
        Fitness function for the genetic programming algorithm to optimize.

        If :class:`~empulse.metrics.Metric`, metric parameters are passed as ``loss_params``
        to the :meth:`~empulse.models.ProfSRClassifier.fit` method.

        If ``None``, the loss is set to the Maximum Profit score.

    generations : int, default=50
        Number of generations to evolve the population of programs.

    population_size : int, default=1000
        Number of programs in each generation.

    parsimony_coefficient : float, default=0.01
        Constant that penalizes large programs by adjusting their fitness to be less favorable
        for selection. Larger values penalize larger programs more severely.

    random_state : int, :class:`numpy:numpy.random.RandomState` or None, default=None
        Controls the randomness of the estimator.
        To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Sklearn Glossary <random_state>` for details.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    model_ : :class:`gplearn:gplearn.genetic.SymbolicRegressor`
        Fitted symbolic regressor.

    n_iter_ : int
        Number of generations evolved.

    Examples
    --------

    .. code-block:: python

        from empulse.models import ProfSRClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_features=4)

        model = ProfSRClassifier(generations=10, population_size=100, random_state=42)
        model.fit(X, y, tp_cost=-200, fp_cost=10)

    References
    ----------
    .. [1] Koza, J. R. (1992). Genetic Programming: On the Programming of Computers by Means
        of Natural Selection. MIT Press.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        **CostSensitiveClassifier._parameter_constraints,
        'generations': [Interval(Integral, 1, None, closed='left')],
        'population_size': [Interval(Integral, 1, None, closed='left')],
        'parsimony_coefficient': [Interval(Real, 0, None, closed='left')],
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
        loss: Metric | None = None,
        generations: int = 50,
        population_size: int = 1000,
        parsimony_coefficient: float = 0.01,
        random_state: np.random.RandomState | int | None = None,
    ) -> None:
        self.generations = generations
        self.population_size = population_size
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state
        super().__init__(tp_cost=tp_cost, tn_cost=tn_cost, fp_cost=fp_cost, fn_cost=fn_cost, loss=loss)

    def _fit(self, X: FloatNDArray, y: IntNDArray, loss: Metric, **loss_params: Any) -> Self:
        try:
            from gplearn.fitness import make_fitness
            from gplearn.genetic import SymbolicRegressor
        except ImportError as e:
            raise ImportError(
                "ProfSRClassifier requires the 'gplearn' package. "
                'Install it with `pip install empulse[symbolic]` or `pip install gplearn`.'
            ) from e

        greater_is_better = loss.direction is Direction.MAXIMIZE
        worst_value = -np.inf if greater_is_better else np.inf

        def _fitness(y_true: FloatNDArray, y_pred: FloatNDArray, sample_weight: FloatNDArray) -> float:
            y_score = expit(y_pred)
            try:
                score = loss(y_true, y_score, **loss_params)
            except (ValueError, TypeError):
                # Raised for example when gplearn validates the fitness function
                # with dummy arrays that do not match the shape of the cost parameters.
                return worst_value
            if not np.isfinite(score):
                return worst_value
            return float(score)

        fitness = make_fitness(function=_fitness, greater_is_better=greater_is_better)

        self.model_ = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            metric=fitness,
            parsimony_coefficient=self.parsimony_coefficient,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        self.n_iter_ = self.generations

        return self

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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_score = expit(self.model_.predict(X))
        return np.vstack((1 - y_score, y_score)).T
