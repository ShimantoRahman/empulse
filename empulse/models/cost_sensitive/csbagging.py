import threading
from collections.abc import Callable
from numbers import Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from joblib import Parallel, delayed
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._base import _partition_estimators
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._available_if import available_if
from sklearn.utils._mask import indices_to_mask
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import _estimator_has, check_is_fitted

from ..._common import Parameter
from ..._types import FloatArrayLike, FloatNDArray, IntArrayLike, IntNDArray, ParameterConstraint
from ...metrics import Metric, expected_cost_loss
from ...utils._sklearn_compat import Tags, type_of_target, validate_data  # type: ignore[attr-defined]
from .._cs_mixin import CostSensitiveMixin
from ._impurity import CostImpurity
from .cstree import CSTreeClassifier


class CSBaggingClassifier(CostSensitiveMixin, ClassifierMixin, BaseEstimator):
    """
    Cost-sensitive Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregates their individual predictions (either by voting or by averaging)
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

    .. seealso::

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression classifier.

        :class:`~empulse.models.CSBoostClassifier` : Cost-sensitive gradient boosting classifier.

        :class:`~empulse.models.CSTreeClassifier` : Cost-sensitive decision tree classifier.

        :class:`~empulse.models.CSForestClassifier` : Cost-sensitive random forest classifier.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a :class:`~empulse.models.CSTreeClassifier`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

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

    loss : Metric, default=None
        The loss function to use in order to evaluate the costs.
        If ``None``, then the costs provided to the constructor or to the ``fit``
        method are used directly.
        If a :class:``~empulse.metrics.Metric`` is provided, then the costs are computed using the
        metric, and any costs provided to the
        constructor or to the ``fit`` method are ignored.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see ``bootstrap`` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see ``bootstrap_features`` for more
        details).

        - If int, then draw ``max_features`` features.
        - If float, then draw ``max(1, int(max_features * n_features_in_))`` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimator_: estimator
        The base estimator from which the ensemble is grown.

    estimators_: list of estimators
        The collection of fitted base estimators.

    estimators_samples_: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    estimators_features_: list of arrays
        The subset of drawn features for each base estimator.

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

    .. [5] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    """

    _parameter_constraints: ClassVar[ParameterConstraint] = {
        'tp_cost': ['array-like', Real],
        'tn_cost': ['array-like', Real],
        'fn_cost': ['array-like', Real],
        'fp_cost': ['array-like', Real],
        'loss': [Metric, None],
        'combination': [
            StrOptions({'majority_voting', 'weighted_voting'}),
        ],
        **BaggingClassifier._parameter_constraints,
    }

    @property
    def estimators_(self) -> list[DecisionTreeClassifier]:
        """The collection of fitted sub-estimators."""
        check_is_fitted(self)
        estimators: list[DecisionTreeClassifier] = self.estimator_.estimators_
        return estimators

    @property
    def n_classes_(self) -> IntNDArray:
        """The number of classes seen during :term:`fit`."""
        check_is_fitted(self)
        n_classes: int | list[int] = self.estimator_.n_classes_
        n_classes_ = np.array(n_classes)
        return n_classes_

    @property
    def oob_score_(self) -> float:
        """Score of the training dataset obtained using an out-of-bag estimate."""
        check_is_fitted(self)
        oob_score: float = self.estimator_.oob_score_
        return oob_score

    @property
    def oob_decision_function_(self) -> FloatNDArray:
        """Decision function computed with out-of-bag estimate on the training set."""
        check_is_fitted(self)
        oob_decision_function: FloatNDArray = self.estimator_.oob_decision_function_
        return oob_decision_function

    @property
    def estimators_samples_(self) -> list[IntNDArray]:
        """The subset of drawn samples (i.e., the in-bag samples) for each base estimator."""
        check_is_fitted(self)
        estimators_samples: list[IntNDArray] = self.estimator_.estimators_samples_
        return estimators_samples

    @property
    def estimators_features_(self) -> list[IntNDArray]:
        """The subset of drawn features (i.e., the in-bag samples) for each base estimator."""
        check_is_fitted(self)
        estimators_features: list[IntNDArray] = self.estimator_.estimators_features_
        return estimators_features

    def __init__(
        self,
        estimator: Any = None,
        n_estimators: int = 10,
        *,
        tp_cost: FloatArrayLike | float = 0.0,
        tn_cost: FloatArrayLike | float = 0.0,
        fn_cost: FloatArrayLike | float = 0.0,
        fp_cost: FloatArrayLike | float = 0.0,
        loss: Metric | None = None,
        combination: Literal['majority_voting', 'weighted_voting'] = 'majority_voting',
        max_samples: float | int = 1.0,
        max_features: int | float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.tp_cost = tp_cost
        self.tn_cost = tn_cost
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.loss = loss
        self.combination = combination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

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

    def _get_metric_loss(self) -> Metric | None:
        """Get the metric loss function if available."""
        if isinstance(self.loss, Metric):
            return self.loss
        return None

    @_fit_context(prefer_skip_nested_validation=True)  # type: ignore[misc]
    def fit(
        self,
        X: FloatArrayLike,
        y: IntArrayLike,
        *,
        tp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        tn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fn_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        fp_cost: FloatArrayLike | float | Parameter = Parameter.UNCHANGED,
        **loss_params: Any,
    ) -> Self:
        """
        Build a cost-sensitive bagging classifier from the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            Ground truth (correct) labels.

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

        loss_params : dict
            Additional keyword arguments to pass to the loss function if using a custom loss function.

        Returns
        -------
        self : CSBaggingClassifier
            Returns self.
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

        if isinstance(self.loss, Metric):
            fp_cost, fn_cost, tp_cost, tn_cost = self.loss._evaluate_costs(**loss_params)
        else:
            tp_cost, tn_cost, fn_cost, fp_cost = self._check_costs(
                tp_cost=tp_cost,
                tn_cost=tn_cost,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
            )

        n_samples = X.shape[0]
        for name, cost in zip(
            ['tp_cost', 'tn_cost', 'fn_cost', 'fp_cost'], [tp_cost, tn_cost, fn_cost, fp_cost], strict=True
        ):
            if isinstance(cost, np.ndarray) and cost.shape[0] != n_samples:
                raise ValueError(f'{name} has shape {cost.shape}, but should have shape ({n_samples},)')

        if self.estimator is None:
            criterion = CostImpurity(
                n_outputs=1,
                n_classes=np.array([2], dtype=np.intp),
            )
            criterion.set_costs(
                tp_cost=tp_cost if not isinstance(tp_cost, np.ndarray) else 0.0,
                tn_cost=tn_cost if not isinstance(tn_cost, np.ndarray) else 0.0,
                fp_cost=fp_cost if not isinstance(fp_cost, np.ndarray) else 0.0,
                fn_cost=fn_cost if not isinstance(fn_cost, np.ndarray) else 0.0,
            )
            criterion.set_array_costs(
                tp_cost=tp_cost.reshape(-1).astype(np.float64)
                if isinstance(tp_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                tn_cost=tn_cost.reshape(-1).astype(np.float64)
                if isinstance(tn_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                fp_cost=fp_cost.reshape(-1).astype(np.float64)
                if isinstance(fp_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                fn_cost=fn_cost.reshape(-1).astype(np.float64)
                if isinstance(fn_cost, np.ndarray)
                else np.array([], dtype=np.float64),
                n_samples=n_samples,
            )
            self.base_estimator_ = CSTreeClassifier(criterion=criterion)
        else:
            self.base_estimator_ = clone(self.estimator)

        with config_context(enable_metadata_routing=True):
            self.estimator_ = BaggingClassifier(
                estimator=self.base_estimator_.set_fit_request(tp_cost=True, fp_cost=True, tn_cost=True, fn_cost=True),
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                max_samples=self.max_samples,
            )
            self.estimator_.fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)

        if self.combination == 'weighted_voting':
            if not self.bootstrap:
                raise ValueError('Weighted voting is only available when bootstrap=True.')
            if self.loss is None:
                self.estimator_weights_ = self._get_oob_weights(
                    X,
                    y,
                    tp_cost=tp_cost,
                    tn_cost=tn_cost,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                    check_input=False,
                )
            else:
                self.estimator_weights_ = self._get_oob_weights(X, y, **loss_params)
        return self

    def predict(self, X: FloatArrayLike) -> IntNDArray:
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        y_proba = self.predict_proba(X)
        y_pred: IntNDArray = self.classes_.take(np.argmax(y_proba, axis=1), axis=0)
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
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X: FloatNDArray = validate_data(self, X, reset=False)

        if self.combination == 'weighted_voting':
            y_proba: FloatNDArray = self._predict_weighted_proba(X)
        else:
            y_proba = self.estimator_.predict_proba(X)

        return y_proba

    def predict_log_proba(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        y_proba = self.predict_proba(X)
        return np.log(y_proba)

    @available_if(_estimator_has('decision_function', delegates=('base_estimator_', 'estimator')))
    def decision_function(self, X: FloatArrayLike) -> FloatNDArray:
        """
        Average of the decision functions of the base classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        score : ndarray of shape (n_samples, 1)
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``.
        """
        check_is_fitted(self)
        decisions: FloatNDArray = self.estimator_.decision_function(X)
        return decisions

    def _get_oob_weights(self, X: FloatNDArray, y: IntNDArray, **loss_params: Any) -> FloatNDArray:
        n_samples = y.shape[0]

        estimator_weights = np.zeros(self.n_estimators, dtype=np.float64)
        weight_fn = self.loss if self.loss is not None else expected_cost_loss

        for i, estimator, samples, features in zip(
            range(self.n_estimators), self.estimators_, self.estimators_samples_, self.estimators_features_, strict=True
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba((X[mask, :])[:, features])[:, 1]
            else:
                y_pred = estimator.predict((X[mask, :])[:, features])
            estimator_weights[i] = weight_fn(y[mask], y_pred, **loss_params)

        estimator_weights /= estimator_weights.sum()

        return estimator_weights

    def _predict_weighted_proba(self, X: FloatNDArray) -> FloatNDArray:
        X = validate_data(self, X, reset=False)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        n_classes = int(self.n_classes_) if np.ndim(self.n_classes_) == 0 else int(self.n_classes_[0])
        all_proba = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require='sharedmem')(
            delayed(_accumulate_weighted_prediction_non_tree)(e.predict_proba, X, all_proba, weight, lock)
            for e, weight in zip(self.estimators_, self.estimator_weights_, strict=True)
        )

        return all_proba


def _accumulate_weighted_prediction_non_tree(
    predict: Callable[[FloatNDArray], FloatNDArray],
    X: FloatNDArray,
    out: FloatNDArray,
    weight: float,
    lock: threading.Lock,
) -> None:
    """Calculate the weighted prediction."""
    prediction = predict(X)
    with lock:
        out += prediction * weight  # type: ignore[misc]
