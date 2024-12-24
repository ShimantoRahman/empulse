import numbers
import warnings
from functools import partial
from typing import Callable, Optional, Any, Union, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from .._base import BaseLogitClassifier
from ...metrics import make_objective_aec

Loss = Literal["average expected cost"]


class CSLogitClassifier(BaseLogitClassifier):
    """
    Cost-Sensitive Logistic Regression Classifier.

    .. seealso::

        :func:`~empulse.metrics.make_objective_aec` : Creates the instance-specific cost function.

        :class:`~empulse.models.CSBoostClassifier` : Cost-sensitive XGBoost classifier.

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

    loss : ...

    optimize_fn : Callable, optional
        Optimization algorithm. Should be a Callable with signature ``optimize(objective, bounds)``.
        See :ref:`proflogit` for more information.

    optimizer_params : dict[str, Any], optional
        Additional keyword arguments passed to `optimize_fn`.


    Attributes
    ----------
    classes_ : numpy.ndarray
        Unique classes in the target found during fit.

    result_ : :class:`scipy:scipy.optimize.OptimizeResult`
        Optimization result.

    References
    ----------
    .. [1] Höppner, S., Baesens, B., Verbeke, W., & Verdonck, T. (2022).
           Instance-dependent cost-sensitive learning for detecting transfer fraud.
           European Journal of Operational Research, 297(1), 291-300.
    """

    def __init__(
            self,
            C=1.0,
            fit_intercept=True,
            soft_threshold: bool = True,
            l1_ratio: float = 1.0,
            loss: Union[Loss, Callable] = 'average expected cost',
            optimize_fn: Optional[Callable] = None,
            optimizer_params: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            C=C,
            fit_intercept=fit_intercept,
            soft_threshold=soft_threshold,
            l1_ratio=l1_ratio,
            loss=loss,
            optimize_fn=optimize_fn,
            optimizer_params=optimizer_params,
        )

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            tp_cost: Union[ArrayLike, float] = 0.0,
            tn_cost: Union[ArrayLike, float] = 0.0,
            fn_cost: Union[ArrayLike, float] = 0.0,
            fp_cost: Union[ArrayLike, float] = 0.0,
            **loss_params
    ) -> 'CSLogitClassifier':
        """

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

        Returns
        -------
        self

        """
        return super().fit(X, y, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost, **loss_params)

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            tp_cost: Union[ArrayLike, float] = 0.0,
            tn_cost: Union[ArrayLike, float] = 0.0,
            fn_cost: Union[ArrayLike, float] = 0.0,
            fp_cost: Union[ArrayLike, float] = 0.0,
            **loss_params
    ) -> 'CSLogitClassifier':
        optimizer_params = self.optimizer_params or {}

        # Assume that the loss function takes the following parameters:
        if not isinstance(tp_cost, numbers.Number) and (tp_cost := np.asarray(tp_cost)).ndim == 1:
            tp_cost = np.expand_dims(tp_cost, axis=1)
        if not isinstance(tn_cost, numbers.Number) and (tn_cost := np.asarray(tn_cost)).ndim == 1:
            tn_cost = np.expand_dims(tn_cost, axis=1)
        if not isinstance(fn_cost, numbers.Number) and (fn_cost := np.asarray(fn_cost)).ndim == 1:
            fn_cost = np.expand_dims(fn_cost, axis=1)
        if not isinstance(fp_cost, numbers.Number) and (fp_cost := np.asarray(fp_cost)).ndim == 1:
            fp_cost = np.expand_dims(fp_cost, axis=1)
        loss_params['tp_cost'] = tp_cost
        loss_params['tn_cost'] = tn_cost
        loss_params['fn_cost'] = fn_cost
        loss_params['fp_cost'] = fp_cost

        if self.loss == 'average expected cost':
            loss = make_objective_aec(model='cslogit', **loss_params)
            objective = partial(
                _objective_jacobian,
                X=X,
                y=y,
                loss_fn=loss,
                C=self.C,
                l1_ratio=self.l1_ratio,
                soft_threshold=self.soft_threshold,
                fit_intercept=self.fit_intercept,
            )

            if self.optimize_fn is None:
                optimize_fn = _optimize_jacobian
            else:
                optimize_fn = self.optimize_fn

            self.result_ = optimize_fn(
                objective=objective,
                X=X,
                **optimizer_params
            )
        # elif self.loss == 'cross_entropy':
        #     loss = LinearModelLoss(
        #         base_loss=HalfBinomialLoss(), fit_intercept=self.fit_intercept
        #     )
        #     func = loss.loss_gradient
        #     l2_reg_strength = 1.0 / self.C
        #     w0 = np.zeros(X.shape[1], order="F", dtype=X.dtype)
        #     sample_weight = np.ones(X.shape[0], dtype=X.dtype)
        #     self.result_ = minimize(
        #         func,
        #         w0,
        #         method="L-BFGS-B",
        #         jac=True,
        #         args=(X, y, sample_weight, l2_reg_strength, 1),
        #         options={
        #             "maxiter": 10_000,
        #             "maxls": 50,  # default is 20
        #             "gtol": 1e-4,
        #             "ftol": 64 * np.finfo(float).eps,
        #         },
        #     )
        elif isinstance(self.loss, Callable):

            objective = partial(
                _objective_callable,
                X=X,
                y=y,
                loss_fn=partial(self.loss, **loss_params),
                C=self.C,
                l1_ratio=self.l1_ratio,
                soft_threshold=self.soft_threshold,
                fit_intercept=self.fit_intercept,
            )
            if self.optimize_fn is None:
                optimize_fn = self._optimize
            else:
                optimize_fn = self.optimize_fn

            self.result_ = optimize_fn(objective, X=X, **optimizer_params)
        else:
            raise ValueError(f"Invalid loss function: {self.loss}")

        return self

    @staticmethod
    def _optimize(objective, X, max_iter=10000, tolerance=1e-4, **kwargs) -> OptimizeResult:
        initial_weights = np.zeros(X.shape[1], order="F", dtype=np.float64)

        print(max_iter, tolerance, kwargs)
        result = minimize(
            objective,
            initial_weights,
            method="L-BFGS-B",
            options={
                "maxiter": max_iter,
                "maxls": 50,
                "gtol": tolerance,
                "ftol": 64 * np.finfo(float).eps,
            },
            **kwargs,
        )
        _check_optimize_result(result)

        return result


def _optimize_jacobian(
        objective,
        X,
        max_iter=10000,
        tolerance=1e-4,
        **kwargs
) -> OptimizeResult:
    initial_weights = np.zeros(X.shape[1], order="F", dtype=X.dtype)

    result = minimize(
        objective,
        initial_weights,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "maxls": 50,
            "gtol": tolerance,
            "ftol": 64 * np.finfo(float).eps,
        },
        **kwargs,
    )
    _check_optimize_result(result)

    return result


def _objective_jacobian(weights, X, y, loss_fn, C, l1_ratio, soft_threshold, fit_intercept):
    """compute the objective function and its gradient using elastic net regularization."""

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

    loss, gradient = loss_fn(X, weights, y)
    regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b ** 2) + l1_ratio * np.sum(np.abs(b))
    penalty = regularization_term / C
    gradient_penalty = (1 - l1_ratio) * b + l1_ratio * np.sign(b)
    if fit_intercept:
        gradient_penalty = np.hstack((np.array([0]), gradient_penalty))
    return loss + penalty, gradient + gradient_penalty


def _objective_callable(weights, X, y, loss_fn, C, l1_ratio, soft_threshold, fit_intercept):
    """objective function (minimization problem)."""

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

    logits = np.dot(weights, X.T)
    y_pred = expit(logits)
    loss_output = loss_fn(y, y_pred)
    if isinstance(loss_output, tuple):
        if len(loss_output) == 2:
            loss, gradient = loss_output
            return loss + _compute_penalty(b, C, l1_ratio), gradient
        elif len(loss_output) == 3:
            loss, gradient, hessian = loss_output
            return loss + _compute_penalty(b, C, l1_ratio), gradient, hessian
        else:
            raise ValueError(f"Invalid loss function output length: {len(loss_output)}, expected 1, 2 or 3. "
                             f"(loss, gradient, hessian)")
    else:
        loss = loss_output
        return loss + _compute_penalty(b, C, l1_ratio)


def _compute_penalty(b, C, l1_ratio):
    regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b ** 2) + l1_ratio * np.sum(np.abs(b))
    penalty = regularization_term / C
    return penalty


def _check_optimize_result(result):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    result : OptimizeResult
       Result of the scipy.optimize.minimize function.

    max_iter : int, default=None
       Expected maximum number of iterations.
    """
    # handle both scipy and scikit-learn solver names
    if result.status != 0:
        try:
            # The message is already decoded in scipy>=1.6.0
            result_message = result.message.decode("latin1")
        except AttributeError:
            result_message = result.message
        warning_msg = (
            "L-BFGS failed to converge (status={}):\n{}.\n\n"
            "Increase the number of iterations (max_iter) "
            "or scale the data as shown in:\n"
            "    https://scikit-learn.org/stable/modules/"
            "preprocessing.html"
        ).format(result.status, result_message)
        warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)

# def transform_cost_matrix(x: NDArray, y: NDArray, cost_matrix: NDArray) -> NDArray[np.float64]:
#     # Transform cost matrix
#     # create df: a merge of x, y and amount
#     amounts = cost_matrix[:, 0, 1]
#     df = pd.DataFrame(data=x)
#     df['y'] = y
#     df['Amount'] = amounts
#
#     # step 1: fit model on df, return df_enhanced
#     df_enhanced = enhance_df(df)
#     # step 2: replace conditionally outlying amounts with estimated amounts
#     df = impute_amounts(df_enhanced)
#     # step 3: return cost_matrix_transformed
#     cost_matrix_transformed = get_cost_matrix(df['Amount'])
#     return cost_matrix_transformed
# def get_cost_matrix(cost):
#     cost_matrix = np.zeros((len(cost), 2, 2))  # cost_matrix [[TN, FN], [FP, TP]]
#     cost_matrix[:, 0, 0] = 0
#     cost_matrix[:, 0, 1] = cost
#     cost_matrix[:, 1, 0] = cost
#     cost_matrix[:, 1, 1] = 0
#
#     return cost_matrix
#
#
# def enhance_df(df):
#     model_0 = HuberRegressor(max_iter=1000)
#     model_1 = HuberRegressor(max_iter=1000)
#
#     df_0 = df[df.y == 0]
#     model_0.fit(df_0.loc[:, df_0.columns != 'Amount'], df_0['Amount'])
#     Ahat_0_array = model_0.predict(df_0.loc[:, df_0.columns != 'Amount'])
#     Ahat_0_array[Ahat_0_array < 0] = 0
#
#     df_1 = df[df.y == 1]
#     model_1.fit(df_1.loc[:, df_1.columns != 'Amount'], df_1['Amount'])
#     Ahat_1_array = model_1.predict(df_1.loc[:, df_1.columns != 'Amount'])
#     Ahat_1_array[Ahat_1_array < 0] = 0
#
#     df_0['Amount_pred'] = Ahat_0_array
#     df_1['Amount_pred'] = Ahat_1_array
#     prediction = pd.concat([df_0, df_1])
#     df['Amount_pred'] = prediction['Amount_pred']
#
#     return df
#
#
# # Function to detect and impute conditionally outlying amount
# def impute_amounts(df):
#     std = st.sem(df['Amount'])
#     df['resid'] = df['Amount'] - df['Amount_pred']
#     df['resid_std'] = df['resid'] / std
#     df['resid_std_abs'] = abs(df['resid_std'])
#     df.loc[df.resid_std_abs > 3, 'Amount'] = df['Amount_pred']
#     print("")
#     df = df.drop(columns=['Amount_pred', 'resid', 'resid_std', 'resid_std_abs'])
#     return df