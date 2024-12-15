from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data, check_is_fitted
from xgboost import XGBClassifier

from ..metrics import make_objective_churn, mpc_cost_score


class B2BoostClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """
    :class:`xgboost:xgboost.XGBClassifier` with instance-specific cost function for customer churn

    Parameters
    ----------
    estimator : `xgboost:xgboost.XGBClassifier`, optional
        XGBoost classifier to be fit with desired hyperparameters.
        If not provided, a XGBoost classifier with default hyperparameters is used.

    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (0 < `accept_rate` < 1).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
        If ``array``: individualized customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).

    contact_cost : float, default=1
        Constant cost of contact (``contact_cost > 0``).

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : `xgboost:xgboost.XGBClassifier`
        Fitted XGBoost classifier.

    Notes
    -----
    The instance-specific cost function for customer churn is defined as [1]_:

    .. math:: C(s_i) = y_i[s_i(f-\\gamma (1-\\delta )CLV_i] + (1-y_i)[s_i(\\delta CLV_i + f)]

    The measure requires that the churn class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    .. seealso::
        :func:`~empulse.metrics.create_objective_churn` : Creates the instance-specific cost function
        for customer churn.

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """

    def __init__(
            self,
            estimator: Optional[XGBClassifier] = None,
            *,
            accept_rate: float = 0.3,
            clv: Union[float, ArrayLike] = 200,
            incentive_fraction: float = 0.05,
            contact_cost: float = 15,
    ) -> None:
        self.estimator = estimator
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost
        self.accept_rate = accept_rate

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X, y, sample_weights=None, accept_rate=None, clv=None, incentive_fraction=None, contact_cost=None):
        """
        Fit the model.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
        y : 1D numpy.ndarray, shape=(n_samples,)

        sample_weights : 1D numpy.ndarray, shape=(n_samples,), default=None
            Sample weights.

        accept_rate : float, default=0.3
            Probability of a customer responding to the retention offer (``0 < accept_rate < 1``).

        clv : float or 1D array-like, shape=(n_samples), default=200
            If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
            If ``array``: individualized customer lifetime value of each customer when retained
            (``mean(clv) > incentive_cost``).

        incentive_fraction : float, default=10
            Cost of incentive offered to a customer, as a fraction of customer lifetime value
            (``0 < incentive_fraction < 1``).

        contact_cost : float, default=1
            Constant cost of contact (``contact_cost > 0``).

        Returns
        -------
        self : B2BoostClassifier
            Fitted B2Boost model.
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

        objective = make_objective_churn(
            clv=self.clv if clv is None else clv,
            incentive_fraction=self.incentive_fraction if incentive_fraction is None else incentive_fraction,
            contact_cost=self.contact_cost if contact_cost is None else contact_cost,
            accept_rate=self.accept_rate if accept_rate is None else accept_rate,
        )
        if self.estimator is None:
            self.estimator_ = XGBClassifier(objective=objective)
        else:
            if not isinstance(self.estimator, XGBClassifier):
                raise ValueError("estimator must be an instance of XGBClassifier")
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        self.estimator_.fit(X, y, sample_weight=sample_weights)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)

        Returns
        -------
        y_pred : 2D numpy.ndarray, shape=(n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self.estimator_.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)

        Returns
        -------
        y_pred : 1D numpy.ndarray, shape=(n_samples,)
            Predicted class labels.
        """
        y_pred = self.predict_proba(X)
        return self.classes_[np.argmax(y_pred, axis=1)]

    def score(
            self,
            X: ArrayLike,
            y: ArrayLike,
            accept_rate: float = None,
            clv=None,
            incentive_fraction=None,
            contact_cost=None
    ) -> float:
        """
        Compute EMPB score.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
            Features.
        y : 1D numpy.ndarray, shape=(n_samples,)
            Labels.

        accept_rate : float, default=self.accept_rate
            Probability of a customer responding to the retention offer (``0 < accept_rate < 1``).

        clv : float or 1D array-like, shape=(n_samples), default=self.clv
            If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
            If ``array``: individualized customer lifetime value of each customer when retained
            (``mean(clv) > incentive_cost```).

        incentive_fraction : float, default=self.incentive_cost
            Cost of incentive offered to a customer, as a fraction of customer lifetime value
            (``0 < incentive_fraction < 1``).

        contact_cost : float, default=self.contact_cost
            Constant cost of contact (``contact_cost > 0``).

        Returns
        -------
        score : float
            Model score.
        """
        X, y = validate_data(self, X, y, reset=False)
        y = np.where(y == self.classes_[1], 1, 0)
        return mpc_cost_score(
            y,
            self.predict_proba(X)[:, 1],
            accept_rate=accept_rate or self.accept_rate,
            clv=clv or self.clv,
            incentive_fraction=incentive_fraction or self.incentive_fraction,
            contact_cost=contact_cost or self.contact_cost,
        )
