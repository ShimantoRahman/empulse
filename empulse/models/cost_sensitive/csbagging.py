import itertools
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import Parallel, delayed, cpu_count
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import validate_data, check_random_state, check_is_fitted

from .cstree import CSTreeClassifier
from ..._common import Parameter
from ...metrics import savings_score
from ._cs_mixin import CostSensitiveMixin

# from ...models import CostSensitiveLogisticRegression
# from .models import BayesMinimumRiskClassifier

MAX_INT = np.iinfo(np.int32).max


# Is different in 0.15, copy version from 0.16-git
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    if n_jobs == -1:
        n_jobs = min(cpu_count(), n_estimators)

    else:
        n_jobs = min(n_jobs, n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=np.int64)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _parallel_build_estimators(n_estimators, ensemble, X, y, cost_mat,
                               seeds, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features_

    if (not isinstance(max_samples, (numbers.Integral, np.integer)) and
            (0.0 < max_samples <= 1.0)):
        max_samples = int(max_samples * n_samples)

    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features

    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(("building estimator %d of %d" % (i + 1, n_estimators)))

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=seed)
        except ValueError:
            pass

        # Draw features
        if bootstrap_features:
            features = random_state.randint(0, n_features, max_features)
        else:
            features = sample_without_replacement(n_features,
                                                  max_features,
                                                  random_state=random_state)

        # Draw samples, using a mask, and then fit
        if bootstrap:
            indices = random_state.randint(0, n_samples, max_samples)
        else:
            indices = sample_without_replacement(n_samples,
                                                 max_samples,
                                                 random_state=random_state)

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
        samples = sample_counts > 0.

        estimators.append(estimator)
        estimators_samples.append(samples)
        estimators_features.append(features)

    return estimators, estimators_samples, estimators_features


def _parallel_predict_proba(estimators, estimators_features, X, n_classes, combination, estimators_weight):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features, weight in zip(estimators, estimators_features, estimators_weight):
        proba_estimator = estimator.predict_proba(X[:, features])
        if combination in ['weighted_voting', 'weighted_bmr']:
            proba += proba_estimator * weight
        else:
            proba += proba_estimator

    return proba


def _parallel_predict(estimators, estimators_features, X, n_classes, combination, estimators_weight):
    """Private function used to compute predictions within a job."""
    n_samples = X.shape[0]
    pred = np.zeros((n_samples, n_classes))
    n_estimators = len(estimators)

    for estimator, features, weight in zip(estimators, estimators_features, estimators_weight):
        # Resort to voting
        predictions = estimator.predict(X[:, features])

        for i in range(n_samples):
            if combination == 'weighted_voting':
                pred[i, int(predictions[i])] += 1 * weight
            else:
                pred[i, int(predictions[i])] += 1

    return pred


# TODO: Create stacking set in parallel


def _create_stacking_set(estimators, estimators_features, estimators_weight, X, combination):
    """Private function used to create the stacking training set."""
    n_samples = X.shape[0]

    valid_estimators = np.nonzero(estimators_weight)[0]
    n_valid_estimators = valid_estimators.shape[0]
    X_stacking = np.zeros((n_samples, n_valid_estimators))

    for e in range(n_valid_estimators):
        if combination in ['stacking', 'stacking_bmr']:
            X_stacking[:, e] = estimators[valid_estimators[e]].predict(
                X[:, estimators_features[valid_estimators[e]]])
        elif combination in ['stacking_proba', 'stacking_proba_bmr']:
            X_stacking[:, e] = estimators[valid_estimators[e]].predict_proba(
                X[:, estimators_features[valid_estimators[e]]])[:, 1]

    return X_stacking


class BaseBagging(CostSensitiveMixin, BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
            self,
            tp_cost: ArrayLike | float = 0.0,
            tn_cost: ArrayLike | float = 0.0,
            fn_cost: ArrayLike | float = 0.0,
            fp_cost: ArrayLike | float = 0.0,
            estimator=None,
            n_estimators=10,
            estimator_params=tuple(),
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination='majority_voting',
            n_jobs=1,
            random_state=None,
            verbose=0
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
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.combination = combination
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            *,
            tp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            tn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            fn_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
            fp_cost: ArrayLike | float | Parameter = Parameter.UNCHANGED,
    ):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

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
                'Only binary classification is supported. The type of the target '
                f'is {y_type}.'
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
            n_samples=len(y)
        )
        cost_mat = np.column_stack((fp_cost, fn_cost, tp_cost, tn_cost))

        # Remap output
        n_samples, self.n_features_ = X.shape

        # Check parameters
        self._validate_estimator(default=CSTreeClassifier())

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_))
            else:
                max_features = 1  # On sklearn is 0.
        self.max_features_ = max_features

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators,
            self.n_jobs
        )
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                cost_mat,
                seeds[starts[i]:starts[i + 1]],
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(
            t[2] for t in all_results))

        self._evaluate_oob_savings(X, y, cost_mat)

        if self.combination in ['stacking', 'stacking_proba', 'stacking_bmr', 'stacking_proba_bmr']:
            self._fit_stacking_model(X, y, cost_mat)

        if self.combination in ['majority_bmr', 'weighted_bmr', 'stacking_bmr', 'stacking_proba_bmr']:
            self._fit_bmr_model(X, y)

        return self

    def _fit_bmr_model(self, X, y):
        """Private function used to fit the BayesMinimumRisk model."""
        raise NotImplementedError("BayesMinimumRisk model not implemented yet.")
        # self.f_bmr = BayesMinimumRiskClassifier()
        # X_bmr = self.predict_proba(X)
        # self.f_bmr.fit(y, X_bmr)
        # return self

    def _fit_stacking_model(self, X, y, cost_mat, max_iter=100):
        """Private function used to fit the stacking model."""
        raise NotImplementedError("Stacking model not implemented yet.")
        # self.f_staking = CostSensitiveLogisticRegression(
        #     verbose=self.verbose, max_iter=max_iter)
        # X_stacking = _create_stacking_set(self.estimators_, self.estimators_features_,
        #                                   self.estimators_weight_, X, self.combination)
        # self.f_staking.fit(X_stacking, y, cost_mat)
        # return self

    # TODO: _evaluate_oob_savings in parallel
    def _evaluate_oob_savings(self, X, y, cost_mat):
        """Private function used to calculate the OOB Savings of each estimator."""
        estimators_weight = []
        for estimator, samples, features in zip(self.estimators_, self.estimators_samples_,
                                                self.estimators_features_):
            # Test if all examples where used for training
            if not np.any(~samples):
                # Then use training
                oob_pred = estimator.predict(X[:, features])
                fp_cost = cost_mat[:, 0]
                fn_cost = cost_mat[:, 1]
                tp_cost = cost_mat[:, 2]
                tn_cost = cost_mat[:, 3]
                oob_savings = max(0.0, savings_score(
                    y,
                    oob_pred,
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    check_input=False
                ))
            else:
                # Then use OOB
                oob_pred = estimator.predict((X[~samples])[:, features])
                fp_cost = cost_mat[~samples, 0]
                fn_cost = cost_mat[~samples, 1]
                tp_cost = cost_mat[~samples, 2]
                tn_cost = cost_mat[~samples, 3]
                oob_savings = max(0.0, savings_score(
                    y[~samples], oob_pred,
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    check_input=False
                ))

            estimators_weight.append(oob_savings)

        # Control in case were all weights are 0
        if sum(estimators_weight) == 0:
            self.estimators_weight_ = np.ones(
                len(estimators_weight)) / len(estimators_weight)
        else:
            self.estimators_weight_ = (
                    np.array(estimators_weight) / sum(estimators_weight)).tolist()

        return self


class BaggingClassifier(ClassifierMixin, BaseBagging):
    """A Bagging classifier.

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
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

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

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    """

    def __init__(self,
                 estimator=None,
                 n_estimators=10,
                 tp_cost: ArrayLike | float = 0.0,
                 tn_cost: ArrayLike | float = 0.0,
                 fn_cost: ArrayLike | float = 0.0,
                 fp_cost: ArrayLike | float = 0.0,
                 estimator_params=tuple(),
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 combination='majority_voting',
                 n_jobs=1,
                 random_state=None,
                 verbose=0):

        super().__init__(
            estimator=estimator,
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
            verbose=verbose
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    # def _validate_estimator(self, default=None):
    #     """Check the estimator and set the base_estimator_ attribute."""
    #     super()._validate_estimator(default=DecisionTreeClassifier())

    # def _validate_y(self, y):
    #     y = column_or_1d(y, warn=True)
    #     self.classes_, y = np.unique(y, return_inverse=True)
    #     self.n_classes_ = len(self.classes_)
    #
    #     return y

    def predict(self, X, cost_mat=None):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        cost_mat : optional array-like of shape = [n_samples, 4], (default=None)
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        pred : array of shape = [n_samples]
            The predicted classes.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if self.combination in ['stacking', 'stacking_proba']:

            X_stacking = _create_stacking_set(self.estimators_, self.estimators_features_,
                                              self.estimators_weight_, X, self.combination)
            return self.f_staking.predict(X_stacking)

        elif self.combination in ['majority_voting', 'weighted_voting']:
            # Parallel loop
            n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                                 self.n_jobs)

            all_pred = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict)(
                    self.estimators_[starts[i]:starts[i + 1]],
                    self.estimators_features_[starts[i]:starts[i + 1]],
                    X,
                    len(self.classes_),
                    self.combination,
                    self.estimators_weight_[starts[i]:starts[i + 1]])
                for i in range(n_jobs))

            # Reduce
            pred = sum(all_pred) / self.n_estimators

            return self.classes_.take(np.argmax(pred, axis=1), axis=0)

        elif self.combination in ['majority_bmr', 'weighted_bmr', 'stacking_bmr', 'stacking_proba_bmr']:
            # TODO: Add check if cost_mat == None
            X_bmr = self.predict_proba(X)
            return self.f_bmr.predict(X_bmr, cost_mat)
        else:
            raise ValueError("Invalid combination method.")

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of a an input sample represents the proportion of estimators predicting
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
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                len(self.classes_),
                self.combination,
                self.estimators_weight_[starts[i]:starts[i + 1]])
            for i in range(n_jobs))

        # Reduce
        if self.combination in ['majority_voting', 'majority_bmr']:
            proba = sum(all_proba) / self.n_estimators
        elif self.combination in ['weighted_voting', 'weighted_bmr']:
            proba = sum(all_proba)
        elif self.combination in ['stacking', 'stacking_proba', 'stacking_bmr', 'stacking_proba_bmr']:
            X_stacking = _create_stacking_set(self.estimators_, self.estimators_features_,
                                              self.estimators_weight_, X, self.combination)
            proba = self.f_staking.predict_proba(X_stacking)

        return proba
