from collections import defaultdict
from typing import Any, Union, Optional

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from xgboost import XGBClassifier

from ..metrics import make_objective_churn, mpc_cost_score


class B2BoostClassifier(BaseEstimator, ClassifierMixin):
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

    incentive_cost : float, default=10
        Constant cost of retention offer (``incentive_cost > 0``).

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
            incentive_cost: float = 10,
            contact_cost: float = 1,
            params: Optional[dict[str, Any]] = None,
            **kwargs,
    ) -> None:
        self.clv = clv
        self.incentive_cost = incentive_cost
        self.contact_cost = contact_cost
        self.accept_rate = accept_rate

        # necessary to have params because sklearn.clone does not clone **kwargs
        if params is None:
            params = {}
        if kwargs:
            params.update(kwargs)
        self.params = params

        self.model = XGBClassifier(**params)

    def __getattr__(self, attr):
        """
        If the attribute is not found in B2BoostClassifier,
        it checks if it is present in `self.model` and if it is, returns that.
        """
        if hasattr(self.model, attr):
            return getattr(self.model, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`sklearn:sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        valid_params_xgb = self.model.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params and key not in valid_params_xgb:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            if key not in valid_params and key in valid_params_xgb:
                self.model.set_params(**{key: value})
                self.params[key] = value
                continue

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def fit(self, X, y, sample_weights=None, accept_rate=None, clv=None, incentive_cost=None, contact_cost=None):
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

        incentive_cost : float, default=10
            Constant cost of retention offer (``incentive_cost > 0``).

        contact_cost : float, default=1
            Constant cost of contact (``contact_cost > 0``).

        Returns
        -------
        self : B2BoostClassifier
            Fitted B2Boost model.
        """
        objective = make_objective_churn(
            clv=self.clv if clv is None else clv,
            incentive_cost=self.incentive_cost if incentive_cost is None else incentive_cost,
            contact_cost=self.contact_cost if contact_cost is None else contact_cost,
            accept_rate=self.accept_rate if accept_rate is None else accept_rate,
        )
        self.model = XGBClassifier(objective=objective, **self.params)
        self.model.fit(X, y, sample_weight=sample_weights)
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
            accept_rate: float = None,
            clv=None,
            incentive_cost=None,
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

        incentive_cost : float, default=self.incentive_cost
            Constant cost of retention offer (``incentive_cost > 0``).

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
            incentive_cost=incentive_cost or self.incentive_cost,
            contact_cost=contact_cost or self.contact_cost,
        )
