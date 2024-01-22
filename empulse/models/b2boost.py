from typing import Any, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from xgboost import XGBClassifier

from ..metrics import create_objective_churn, empb_score


class B2BoostClassifier(BaseEstimator, ClassifierMixin):
    """
    `XGBoostClassifier` [1]_ wrapper with instance-specific cost function for customer churn [2]_.

    Parameters
    ----------
    accept_rate : float, default=0.3
        Probability of a customer responding to the retention offer (0 < `accept_rate` < 1).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (`clv` > `incentive_cost`).
        If ``array``: individualized customer lifetime value of each customer when retained
        (mean(`clv`) > `incentive_cost`).

    incentive_cost : float, default=10
        Constant cost of retention offer (`incentive_cost` > 0).

    contact_cost : float, default=1
        Constant cost of contact (`contact_cost` > 0).

    kwargs : dict
        Other parameters passed to `XGBClassifier` init.

    Notes
    -----
    The instance-specific cost function for customer churn is defined as [2]_:

    .. math:: C(s_i) = y_i[s_i(f-\\gamma (1-\\delta )CLV_i] + (1-y_i)[s_i(\\delta CLV_i + f)]

    The measure requires that the churn class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    .. seealso::
        :func:`~empulse.metrics.create_objective_churn` : Creates the instance-specific cost function
        for customer churn.

    References
    ----------
    .. [1] https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
    .. [2] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    def __init__(
            self,
            *,
            accept_rate: float = 0.3,
            clv: Union[float, ArrayLike] = 200,
            incentive_cost: float = 10,
            contact_cost: float = 1,
            **kwargs: Any,
    ) -> None:
        self.clv = clv
        self.incentive_cost = incentive_cost
        self.contact_cost = contact_cost
        self.accept_rate = accept_rate
        self.model = None
        self.classes_ = None
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fit the model.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
            Training data.
        y : 1D numpy.ndarray, shape=(n_samples,)
            Target values.

        Returns
        -------
        self : B2BoostClassifier
            Fitted B2Boost model.
        """
        objective = create_objective_churn(
            clv=self.clv,
            incentive_cost=self.incentive_cost,
            contact_cost=self.contact_cost,
            accept_rate=self.accept_rate
        )
        self.model = XGBClassifier(objective=objective, **self.kwargs)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
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
        return self.model.predict_proba(X)

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
        return self.model.predict(X)

    def score(
            self,
            X: ArrayLike,
            y: ArrayLike,
            alpha=6,
            beta=14,
            clv=None,
            incentive_cost_fraction=None,
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
        alpha : float, default=6
        Shape parameter of the beta distribution of the probability that a churner accepts the incentive (`alpha` > 1).

        beta : float, default=14
            Shape parameter of the beta distribution of the probability that a churner accepts the incentive (`beta` > 1).

        clv : float or 1D array-like, shape=(n_samples), default=self.clv
            If ``float``: constant customer lifetime value per retained customer (`clv` > `incentive_cost`).
            If ``array``: individualized customer lifetime value of each customer when retained
            (mean(`clv`) > `incentive_cost`).

        incentive_cost_fraction : float, default=self.incentive_cost / np.mean(clv)
            Fraction of the customer lifetime value that is used as the incentive cost (`incentive_cost_fraction` > 0).

        contact_cost : float, default=self.contact_cost
            Constant cost of contact (`contact_cost` > 0).

        Returns
        -------
        score : float
            Model score.
        """
        X, y = check_X_y(X, y)
        return empb_score(
            y,
            self.predict_proba(X)[:, 1],
            alpha=alpha,
            beta=beta,
            clv=clv or self.clv,
            incentive_cost_fraction=incentive_cost_fraction or self.incentive_cost / np.mean(clv),
            contact_cost=contact_cost or self.contact_cost,
        )
