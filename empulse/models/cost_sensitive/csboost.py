from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from xgboost import XGBClassifier

from .._base import BaseBoostClassifier
from ...metrics import make_objective_aec


class CSBoostClassifier(BaseBoostClassifier):
    """
    :class:`xgboost:xgboost.XGBClassifier` with instance-specific cost function.

    .. seealso::

        :func:`~empulse.metrics.make_objective_aec` : Creates the instance-specific cost function.

        :class:`~empulse.models.CSLogitClassifier` : Cost-sensitive logistic regression.

    Parameters
    ----------
    estimator : `xgboost:xgboost.XGBClassifier`, optional
        XGBoost classifier to be fit with desired hyperparameters.
        If not provided, a XGBoost classifier with default hyperparameters is used.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape=(n_classes,)
        Unique classes in the target.

    estimator_ : `xgboost:xgboost.XGBClassifier`
        Fitted XGBoost classifier.

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """

    def __init__(
            self,
            estimator: Optional[XGBClassifier] = None,
    ) -> None:
        super().__init__(estimator=estimator)

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            tp_cost: Union[ArrayLike, float] = 0.0,
            tn_cost: Union[ArrayLike, float] = 0.0,
            fn_cost: Union[ArrayLike, float] = 0.0,
            fp_cost: Union[ArrayLike, float] = 0.0,
            **fit_params
    ) -> 'CSBoostClassifier':
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)

        tp_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of true positives. If ``float``, then all true positives have the same cost.
            If array-like, then it is the cost of each true positive classification.

        fp_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of false positives. If ``float``, then all false positives have the same cost.
            If array-like, then it is the cost of each false positive classification.

        tn_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of true negatives. If ``float``, then all true negatives have the same cost.
            If array-like, then it is the cost of each true negative classification.

        fn_cost : float or array-like, shape=(n_samples,), default=0.0
            Cost of false negatives. If ``float``, then all false negatives have the same cost.
            If array-like, then it is the cost of each false negative classification.

        fit_params : dict
            Additional keyword arguments to pass to the estimator's fit method.

        Returns
        -------
        self : B2BoostClassifier
            Fitted B2Boost model.
        """
        return super().fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **fit_params)

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            tp_cost: Union[ArrayLike, float] = 0.0,
            tn_cost: Union[ArrayLike, float] = 0.0,
            fn_cost: Union[ArrayLike, float] = 0.0,
            fp_cost: Union[ArrayLike, float] = 0.0,
            **fit_params
    ) -> 'CSBoostClassifier':
        objective = make_objective_aec(
            'csboost',
            tp_cost=tp_cost,
            tn_cost=tn_cost,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        if self.estimator is None:
            self.estimator_ = XGBClassifier(objective=objective)
        else:
            if not isinstance(self.estimator, XGBClassifier):
                raise ValueError("estimator must be an instance of XGBClassifier")
            self.estimator_ = clone(self.estimator).set_params(objective=objective)
        self.estimator_.fit(X, y, **fit_params)
        return self
