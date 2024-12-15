import warnings
from functools import partial
from itertools import islice
from typing import Callable, Optional, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted

from empulse.optimizers import Generation
from ..metrics import empc_score


class ProfLogitClassifier(ClassifierMixin, BaseEstimator):
    """
    Logistic classifier to optimize profit-driven loss functions

    Maximizing empirical EMP for churn by optimizing
    the regression coefficients of the logistic model through
    a Real-coded Genetic Algorithm (RGA).

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive ``float``.
        Like in support vector machines, smaller values specify stronger regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    soft_threshold : bool, default=True
        If ``True``, apply soft-thresholding to the regression coefficients.

    l1_ratio : float, default=1.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is a L2 penalty.
        For ``l1_ratio = 1`` it is a L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    loss_fn : Callable, default= :func:`empulse.metrics.empc_score`
        Loss function. Should be a Callable with signature ``loss(y_true, y_pred)`` or ``loss(y_true, y_score)``.
        See :func:`empulse.metrics.empc_score` for an example.
        By default, expects a loss function to maximize, customize behaviour in `optimize_fn`.

    optimize_fn : Callable, optional
        Optimization algorithm. Should be a Callable with signature ``optimize(objective, bounds)``.
        See :ref:`proflogit` for more information.

    optimizer_params : dict[str, Any], optional
        Additional keyword arguments passed to `optimize_fn`.

        By default, the optimizer is a Real-coded Genetic Algorithm (RGA) with the following parameters:

        - ``max_iter`` : int, default=1000
            Maximum number of iterations.
        - ``patience`` : int, default=250
            Number of iterations with no improvement to wait before stopping the optimization.
        - ``tolerance`` : float, default=1e-4
            Relative tolerance to declare convergence.
        - all other parameters are passed to the :class:`~empulse.optimizers.Generation` initializer.

    default_bounds : tuple, default=(-3, 3)
        Bounds for every regression parameter. Use the `bounds` parameter
        through `optimize_fn` for individual specifications.

    n_jobs : int, optional
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.



    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    Notes
    -----
    Original implementation of ProfLogit [3]_ in Python.

    References
    ----------
    .. [1] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2017). Profit Maximizing Logistic Model for
        Customer Churn Prediction Using Genetic Algorithms.
        Swarm and Evolutionary Computation.
    .. [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2015). Profit Maximizing Logistic Regression Modeling for
        Customer Churn Prediction. IEEE International Conference on
        Data Science and Advanced Analytics (DSAA) (pp. 1–10). Paris, France.
    .. [3] https://github.com/estripling/proflogit/tree/master
    """

    def __init__(
            self,
            C: float = 1.0,
            fit_intercept: bool = True,
            soft_threshold: bool = True,
            l1_ratio: float = 1.0,
            loss_fn: Callable = empc_score,
            optimize_fn: Optional[Callable] = None,
            optimizer_params: Optional[dict[str, Any]] = None,
            default_bounds: tuple[float, float] = (-3, 3),
            n_jobs: Optional[int] = None,
            # random_state: Optional[Union[int, RandomState]] = None
    ):
        super().__init__()
        self.C = C
        self.fit_intercept = fit_intercept
        self.soft_threshold = soft_threshold
        self.l1_ratio = l1_ratio
        self.loss_fn = loss_fn
        self.n_jobs = n_jobs
        self.default_bounds = default_bounds
        self.optimizer_params = optimizer_params
        self.optimize_fn = optimize_fn
        # self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike, **loss_params) -> 'ProfLogitClassifier':
        """
        Fit ProfLogit model.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_features)
        y : 1D array-like, shape=(n_samples,)
        loss_params : dict
            Additional keyword arguments passed to `loss_fn`.

        Returns
        -------
        self : ProfLogitClassifier
            Fitted ProfLogit model.
        """
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

        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n_dim = X.shape[1]

        # random_state = check_random_state(self.random_state)

        optimizer_params = {} if self.optimizer_params is None else self.optimizer_params.copy()
        optimize_fn = _optimize if self.optimize_fn is None else self.optimize_fn
        # if ('random_state' not in self.optimizer_params and
        #         'random_state' in inspect.signature(optimize_fn).parameters):
        #     optimizer_params['random_state'] = random_state
        optimize_fn = partial(optimize_fn, **optimizer_params)
        if 'bounds' not in optimize_fn.keywords:
            optimize_fn = partial(optimize_fn, bounds=[self.default_bounds] * n_dim)
        elif len(optimize_fn.keywords['bounds']) != n_dim:
            raise ValueError(
                f"Number of bounds ({len(optimize_fn.keywords['bounds'])}) "
                f"must match number of features ({n_dim})."
            )

        objective = partial(
            _objective,
            X=X,
            y=y,
            loss_fn=partial(self.loss_fn, **loss_params),
            C=self.C,
            l1_ratio=self.l1_ratio,
            soft_threshold=self.soft_threshold,
            fit_intercept=self.fit_intercept
        )
        self.result_ = optimize_fn(objective)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, 2)
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = self.result_.x
        logits = np.dot(X, theta)
        with warnings.catch_warnings():  # TODO: look into this
            warnings.simplefilter("ignore")
            y_pred = 1 / (1 + np.exp(-logits))  # Invert logit transformation

        # create 2D array with complementary probabilities
        y_pred = np.vstack((1 - y_pred, y_pred)).T
        return y_pred

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Compute predicted labels.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)

        Returns
        -------
        y_pred : 1D numpy.ndarray, shape=(n_samples,)
        """
        y_pred = self.predict_proba(X)
        return self.classes_[np.argmax(y_pred, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight=None) -> float:
        """
        Compute model score.

        Parameters
        ----------
        X : 2D array-like, shape=(n_samples, n_dim)
        y : 1D array-like, shape=(n_samples,)
        sample_weight : 1D array-like, shape=(n_samples,), default=None
            Sample weights (ignored).

        Returns
        -------
        score : float
            Model score.
        """
        X, y = validate_data(self, X, y, reset=False)
        y = np.where(y == self.classes_[1], 1, 0)
        return self.loss_fn(y, self.predict_proba(X)[:, 1])


def _objective(weights, X, y, loss_fn, C, l1_ratio, soft_threshold, fit_intercept):
    """ProfLogit's objective function (maximization problem)."""

    # b is the vector holding the regression coefficients (no intercept)
    b = weights.copy()[1:] if fit_intercept else weights

    if soft_threshold:
        bool_nonzero = (np.abs(b) - C) > 0
        if np.sum(bool_nonzero) > 0:
            b[bool_nonzero] = np.sign(b[bool_nonzero]) * (
                    np.abs(b[bool_nonzero]) - C
            )
        if np.sum(~bool_nonzero) > 0:
            b[~bool_nonzero] = 0

    logits = np.dot(X, weights)
    y_pred = 1 / (1 + np.exp(-logits))  # Invert logit transformation
    loss = loss_fn(y, y_pred)
    regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b ** 2) + l1_ratio * np.sum(np.abs(b))
    penalty = regularization_term / C
    return loss - penalty


def _optimize(objective, bounds, max_iter=1000, tolerance=1e-4, patience=250, **kwargs) -> OptimizeResult:
    rga = Generation(**kwargs)
    previous_score = np.inf
    iter_stagnant = 0

    for _ in islice(rga.optimize(objective, bounds), max_iter):
        score = rga.result.fun
        relative_improvement = (score - previous_score) / previous_score \
            if previous_score != np.inf else np.inf
        previous_score = score
        if relative_improvement < tolerance:
            if (iter_stagnant := iter_stagnant + 1) >= patience:
                rga.result.message = "Converged."
                rga.result.success = True
                break
        else:
            iter_stagnant = 0
    else:
        rga.result.message = "Maximum number of iterations reached."
        rga.result.success = False
    return rga.result
