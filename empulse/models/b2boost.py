from typing import Any, Union, Optional

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from xgboost import XGBClassifier

from .wrapper import WrapperMixin
from ..metrics import make_objective_churn, mpc_cost_score


class B2BoostClassifier(ClassifierMixin, WrapperMixin, BaseEstimator):
    """
    :class:`xgboost:xgboost.XGBClassifier` with instance-specific cost function for customer churn

    Parameters
    ----------
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

    params : dict[str, Any], default=None
        Other parameters passed to :class:`xgboost:xgboost.XGBClassifier` initializer.

    **kwargs
        Other parameters passed to :class:`xgboost:xgboost.XGBClassifier` initializer.

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
            *,
            accept_rate: float = 0.3,
            clv: Union[float, ArrayLike] = 200,
            incentive_fraction: float = 0.05,
            contact_cost: float = 15,
            params: Optional[dict[str, Any]] = None,
            **kwargs,
    ) -> None:
        self.clv = clv
        self.incentive_fraction = incentive_fraction
        self.contact_cost = contact_cost
        self.accept_rate = accept_rate

        # necessary to have params because sklearn.clone does not clone **kwargs
        if params is None:
            params = {}
        if kwargs:
            params.update(kwargs)
        self.params = params

        self.estimator = XGBClassifier(**params)


    def fit(self, X, y, sample_weights=None, accept_rate=None, clv=None, incentive_fraction=None, contact_cost=None):
        """
        Fit the model.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
            Training data.

        y : 1D numpy.ndarray, shape=(n_samples,)
            Target values.

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
        objective = make_objective_churn(
            clv=self.clv if clv is None else clv,
            incentive_fraction=self.incentive_fraction if incentive_fraction is None else incentive_fraction,
            contact_cost=self.contact_cost if contact_cost is None else contact_cost,
            accept_rate=self.accept_rate if accept_rate is None else accept_rate,
        )
        self.estimator = XGBClassifier(objective=objective, **self.params)
        self.estimator.fit(X, y, sample_weight=sample_weights)
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
        return self.estimator.predict_proba(X)

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
        return self.estimator.predict(X)

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
        X, y = check_X_y(X, y)
        return mpc_cost_score(
            y,
            self.predict_proba(X)[:, 1],
            accept_rate=accept_rate or self.accept_rate,
            clv=clv or self.clv,
            incentive_fraction=incentive_fraction or self.incentive_fraction,
            contact_cost=contact_cost or self.contact_cost,
        )
