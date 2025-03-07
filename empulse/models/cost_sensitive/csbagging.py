import contextlib
import itertools
import numbers
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from joblib import Parallel, cpu_count, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin, _fit_context, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntNDArray, ParameterConstraint
from ...metrics import savings_score
from ...utils._sklearn_compat import Tags, type_of_target, validate_data  # type: ignore[attr-defined]
from ._cs_mixin import CostSensitiveMixin
from .cslogit import CSLogitClassifier
from .cstree import CSTreeClassifier

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

MAX_INT = np.iinfo(np.int32).max


def _partition_estimators(n_estimators: int, n_jobs: int) -> tuple[int, list[int], list[int]]:
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(cpu_count(), n_estimators) if n_jobs == -1 else min(n_jobs, n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=np.int64)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0, *starts.tolist()]


def _parallel_build_estimators(
    n_estimators: int,
    ensemble: 'BaseBagging',
    X: FloatNDArray,
    y: IntNDArray,
    cost_mat: FloatNDArray,
    seeds: Sequence[int],
    verbose: int,
) -> tuple[list[Any], list[NDArray[np.bool_]], list[IntNDArray]]:
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features_

    if max_samples is None:
        max_samples = n_samples
    elif not isinstance(max_samples, numbers.Integral | np.integer) and (0.0 < max_samples <= 1.0):
        max_samples = int(max_samples * n_samples)

    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features

    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(f'building estimator {i + 1} of {n_estimators}')  # noqa: T201

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        with contextlib.suppress(ValueError):
            estimator.set_params(random_state=seed)  # Not all estimator accept a random_state

        # Draw features
        if bootstrap_features:
            features: IntNDArray = random_state.randint(0, n_features, max_features)
        else:
            features = sample_without_replacement(n_features, max_features, random_state=random_state)

        # Draw samples, using a mask, and then fit
        if bootstrap:
            indices = random_state.randint(0, n_samples, max_samples)
        else:
            indices = sample_without_replacement(n_samples, max_samples, random_state=random_state)

        sample_counts = np.bincount(indices, minlength=n_samples)

        fp_cost = cost_mat[:, 0]
        fn_cost = cost_mat[:, 1]
        tp_cost = cost_mat[:, 2]
        tn_cost = cost_mat[:, 3]
        estimator.fit(
            (X[indices])[:, features],
            y[indices],
            fp_cost=fp_cost[indices],
            fn_cost=fn_cost[indices],
            tp_cost=tp_cost[indices],
            tn_cost=tn_cost[indices],
        )
        samples = sample_counts > 0.0

        estimators.append(estimator)
        estimators_samples.append(samples)
        estimators_features.append(features)

    return estimators, estimators_samples, estimators_features


def _parallel_predict_proba(
    estimators: Iterable[Any],
    estimators_features: list[IntNDArray],
    X: FloatNDArray,
    n_classes: int,
    combination: str,
    estimators_weight: list[float],
) -> FloatNDArray:
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features, weight in zip(estimators, estimators_features, estimators_weight, strict=False):
        proba_estimator = estimator.predict_proba(X[:, features])
        if combination == 'weighted_voting':
            proba += proba_estimator * weight
        else:
            proba += proba_estimator

    return proba


def _parallel_predict(
    estimators: Iterable[Any],
    estimators_features: list[IntNDArray],
    X: FloatNDArray,
    n_classes: int,
    combination: str,
    estimators_weight: list[float],
) -> FloatNDArray:
    """Private function used to compute predictions within a job."""
    n_samples = X.shape[0]
    pred = np.zeros((n_samples, n_classes))

    for estimator, features, weight in zip(estimators, estimators_features, estimators_weight, strict=False):
        # Resort to voting
        predictions = estimator.predict(X[:, features])

        for i in range(n_samples):
            if combination == 'weighted_voting':
                pred[i, int(predictions[i])] += 1 * weight
            else:
                pred[i, int(predictions[i])] += 1

    return pred


def _create_stacking_set(
    estimators: Sequence[Any],
    estimators_features: list[IntNDArray],
    estimators_weight: FloatNDArray,
    X: FloatNDArray,
    combination: str,
) -> FloatNDArray:
    """Private function used to create the stacking training set."""
    n_samples = X.shape[0]

    valid_estimators = np.nonzero(estimators_weight)[0]
    n_valid_estimators = valid_estimators.shape[0]
    X_stacking = np.zeros((n_samples, n_valid_estimators))

    for e in range(n_valid_estimators):
        if combination == 'stacking':
            X_stacking[:, e] = estimators[valid_estimators[e]].predict(X[:, estimators_features[valid_estimators[e]]])
        elif combination == 'stacking_proba':
            X_stacking[:, e] = estimators[valid_estimators[e]].predict_proba(
                X[:, estimators_features[valid_estimators[e]]]
            )[:, 1]

    return X_stacking


class BaseBagging(CostSensitiveMixin, BaseEnsemble, metaclass=ABCMeta):  # type: ignore[misc]
    """Base class for Bagging meta-estimator."""

    @abstractmethod
    def __init__(
        self,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        estimator: Any = None,
        final_estimator: Any = None,
        n_estimators: int = 10,
        estimator_params: tuple[Any, ...] = tuple(),  # noqa: C408
        max_samples: int | float | None = None,
        max_features: Literal['auto', 'sqrt', 'log2'] | int | float | None = None,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        combination: str = 'majority_voting',
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
        verbose: int | bool = 0,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.final_estimator = final_estimator
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.combination = combination
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: ArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
    ) -> Self:
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values.

        tp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of true positives. If ``float``, then all true positives have the same cost.
            If array-like, then it is the cost of each true positive classification.

        fp_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        tn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of true negatives. If ``float``, then all true negatives have the same cost.
            If array-like, then it is the cost of each true negative classification.

        fn_cost : float or array-like, shape=(n_samples,), default=$UNCHANGED$
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

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
        tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            force_array=True,
            n_samples=len(y),
        )
        cost_mat = np.column_stack((fp_cost, fn_cost, tp_cost, tn_cost))

        # Remap output
        n_samples, self.n_features_ = X.shape

        # Check parameters
        self._validate_estimator(default=CSTreeClassifier())

        if self.max_samples is None:
            max_samples = n_samples
        elif isinstance(self.max_samples, numbers.Integral | np.integer):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError('max_samples must be in (0, n_samples]')

        if isinstance(self.max_features, str):
            if self.max_features in {'auto', 'sqrt'}:
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == 'log2':
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError('Invalid value for max_features. Allowed string values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral | np.integer):
            max_features = self.max_features
        else:  # float
            max_features = max(1, int(self.max_features * self.n_features_)) if self.max_features > 0.0 else 1
        self.max_features_ = max_features

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators, self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                cost_mat,
                seeds[starts[i] : starts[i + 1]],
                verbose=self.verbose,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(t[2] for t in all_results))

        self._evaluate_oob_savings(X, y, cost_mat)

        if self.combination in {'stacking', 'stacking_proba'}:
            self._fit_stacking_model(X, y, cost_mat)

        return self

    def _fit_stacking_model(self, X: FloatNDArray, y: IntNDArray, cost_mat: FloatNDArray) -> Self:
        """Private function used to fit the stacking model."""
        final_estimator = self.final_estimator if self.final_estimator is not None else CSLogitClassifier()
        self.final_estimator_ = clone(final_estimator)
        if self.estimators_ is None or self.estimators_weight_ is None:
            raise RuntimeError('You must call fit() before calling fit_stacking_model().')
        X_stacking = _create_stacking_set(
            self.estimators_,
            self.estimators_features_,
            self.estimators_weight_,
            X,
            self.combination,
        )
        fp_cost = cost_mat[:, 0]
        fn_cost = cost_mat[:, 1]
        tp_cost = cost_mat[:, 2]
        tn_cost = cost_mat[:, 3]
        self.final_estimator_.fit(
            X_stacking,
            y,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        return self

    # TODO: _evaluate_oob_savings in parallel
    def _evaluate_oob_savings(self, X: FloatNDArray, y: IntNDArray, cost_mat: FloatNDArray) -> Self:
        """Private function used to calculate the OOB Savings of each estimator."""
        estimators_weight = []
        if self.estimators_ is None or self.estimators_samples_ is None or self.estimators_features_ is None:
            raise RuntimeError('You must call fit() before calling _evaluate_oob_savings().')
        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_, strict=False
        ):
            # Test if all examples where used for training
            if not np.any(~samples):
                # Then use training
                oob_pred = estimator.predict(X[:, features])
                fp_cost = cost_mat[:, 0]
                fn_cost = cost_mat[:, 1]
                tp_cost = cost_mat[:, 2]
                tn_cost = cost_mat[:, 3]
                oob_savings = max(
                    0.0,
                    savings_score(
                        y,
                        oob_pred,
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                        tp_cost=tp_cost,
                        tn_cost=tn_cost,
                        check_input=False,
                    ),
                )
            else:
                # Then use OOB
                oob_pred = estimator.predict((X[~samples])[:, features])
                fp_cost = cost_mat[~samples, 0]
                fn_cost = cost_mat[~samples, 1]
                tp_cost = cost_mat[~samples, 2]
                tn_cost = cost_mat[~samples, 3]
                oob_savings = max(
                    0.0,
                    savings_score(
                        y[~samples],
                        oob_pred,
                        fp_cost=fp_cost,
                        fn_cost=fn_cost,
                        tp_cost=tp_cost,
                        tn_cost=tn_cost,
                        check_input=False,
                    ),
                )

            estimators_weight.append(oob_savings)

        # Control in case were all weights are 0
        if sum(estimators_weight) == 0:
            self.estimators_weight_ = np.ones(len(estimators_weight)) / len(estimators_weight)
        else:
            self.estimators_weight_ = (np.array(estimators_weight) / sum(estimators_weight)).tolist()

        return self


class BaggingClassifier(ClassifierMixin, BaseBagging):  # type: ignore[misc]
    """
    A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_.

    Parameters
    ----------
    estimator : object or None, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    final_estimator : object or None, default=None
        The final estimator to fit on the stacking set.
        Only relevant if combination is "stacking" or "stacking_proba".

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, default=True
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, default=False
        Whether features are drawn with replacement.

    combination : string, default="majority_voting"
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination.

    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default=0
        Controls the verbosity of the building process.

    Attributes
    ----------
    `estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `final_estimator_`: object
        The final estimator which is fitted on the stacking set.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    `classes_`: array of shape = [n_classes]
        The classes labels.

    `n_classes_`: int or list
        The number of classes.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'n_estimators': [Interval(numbers.Integral, 1, None, closed='left')],
        'tp_cost': ['array-like', numbers.Real],
        'tn_cost': ['array-like', numbers.Real],
        'fn_cost': ['array-like', numbers.Real],
        'fp_cost': ['array-like', numbers.Real],
        'max_features': [
            StrOptions({'auto', 'sqrt', 'log2'}),
            Interval(numbers.Integral, 1, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='right'),
            None,
        ],
        'max_samples': [
            Interval(numbers.Integral, 1, None, closed='left'),
            Interval(RealNotInt, 0.0, 1.0, closed='right'),
            None,
        ],
        'bootstrap': ['boolean'],
        'bootstrap_features': ['boolean'],
        'n_jobs': [Interval(numbers.Integral, -1, None, closed='left')],
        'verbose': [Interval(numbers.Integral, 0, None, closed='left'), 'boolean'],
        'random_state': ['random_state'],
    }

    def __init__(
        self,
        estimator: Any = None,
        final_estimator: Any = None,
        n_estimators: int = 10,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        estimator_params: tuple[Any, ...] = tuple(),  # noqa: C408
        max_samples: int | float | None = None,
        max_features: Literal['auto', 'sqrt', 'log2'] | int | float | None = None,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        combination: Literal['majority_voting', 'weighted_voting', 'stacking', 'stacking_proba'] = 'majority_voting',
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
        verbose: bool | int = 0,
    ):
        super().__init__(
            estimator=estimator,
            final_estimator=final_estimator,
            n_estimators=n_estimators,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            estimator_params=estimator_params,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            combination=combination,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

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

    def predict(self, X: FloatArrayLike) -> NDArray[Any]:
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        pred : array of shape = [n_samples]
            The predicted classes.
        """
        check_is_fitted(self)
        self.estimators_: list[Any]  # refine since check_is_fitted is true
        X = validate_data(self, X, reset=False)

        if self.n_features_ != X.shape[1]:
            raise ValueError(
                'Number of features of the model must '
                f'match the input. Model n_features is {self.n_features_} and '
                f'input n_features is {X.shape[1]}.'
            )

        if self.combination in {'stacking', 'stacking_proba'}:
            X_stacking = _create_stacking_set(
                self.estimators_,
                self.estimators_features_,
                self.estimators_weight_,
                X,
                self.combination,
            )
            y_pred: NDArray[Any] = self.final_estimator_.predict(X_stacking)

        elif self.combination in {'majority_voting', 'weighted_voting'}:
            # Parallel loop
            n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

            if hasattr(self.estimator_, 'predict_proba'):
                all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                    delayed(_parallel_predict_proba)(
                        self.estimators_[starts[i] : starts[i + 1]],
                        self.estimators_features_[starts[i] : starts[i + 1]],
                        X,
                        len(self.classes_),
                        self.combination,
                        self.estimators_weight_[starts[i] : starts[i + 1]],
                    )
                    for i in range(n_jobs)
                )

                proba = sum(all_proba) / self.n_estimators

                y_pred = self.classes_.take(np.argmax(proba, axis=1), axis=0)
            else:
                all_pred = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                    delayed(_parallel_predict)(
                        self.estimators_[starts[i] : starts[i + 1]],
                        self.estimators_features_[starts[i] : starts[i + 1]],
                        X,
                        len(self.classes_),
                        self.combination,
                        self.estimators_weight_[starts[i] : starts[i + 1]],
                    )
                    for i in range(n_jobs)
                )

                # Reduce
                pred = sum(all_pred) / self.n_estimators

                y_pred = self.classes_.take(np.argmax(pred, axis=1), axis=0)
        else:
            raise ValueError('Invalid combination method.')
        return y_pred

    def predict_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.n_features_ != X.shape[1]:
            raise ValueError(
                'Number of features of the model must '
                f'match the input. Model n_features is {self.n_features_} and '
                f'input n_features is {X.shape[1]}.'
            )

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
                len(self.classes_),
                self.combination,
                self.estimators_weight_[starts[i] : starts[i + 1]],
            )
            for i in range(n_jobs)
        )

        # Reduce
        if self.combination == 'majority_voting':
            proba: FloatNDArray = sum(all_proba) / self.n_estimators
        elif self.combination == 'weighted_voting':
            proba = sum(all_proba)
        elif self.combination in {'stacking', 'stacking_proba'}:
            X_stacking = _create_stacking_set(
                self.estimators_,
                self.estimators_features_,
                self.estimators_weight_,
                X,
                self.combination,
            )
            proba = self.final_estimator_.predict_proba(X_stacking)
        else:
            raise ValueError('Invalid combination method.')

        return proba
