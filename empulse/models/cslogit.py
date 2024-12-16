import inspect
import warnings
from functools import partial
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult
import scipy.stats as st
from scipy.special import expit
from sklearn.utils import check_X_y

from sklearn.linear_model import LogisticRegression, HuberRegressor

from .logit import BaseLogitClassifier
from ..metrics import aec_loss

# TODO: base class from CSLogitClassifier and ProfLogitClassifier
# TODO: implement robust version
# TODO: check whether scipy.special.expit is the same as 1 / (1 + np.exp(-x)) and does not issue warnings
# TODO: write tests for CSLogitClassifier

class CSLogitClassifier(BaseLogitClassifier):

    def __init__(
            self,
            C=1.0,
            fit_intercept=True,
            soft_threshold: bool = True,
            l1_ratio: float = 1.0,
            loss_fn: Callable = aec_loss,
            optimize_fn: Optional[Callable] = None,
            default_bounds: tuple[float, float] = (-3, 3),
            n_jobs: Optional[int] = None,
            optimizer_params: Optional[dict[str, Any]] = None,
            robust: bool = False,
            **kwargs,
    ):
        super().__init__(
            C=C,
            fit_intercept=fit_intercept,
            soft_threshold=soft_threshold,
            l1_ratio=l1_ratio,
            loss_fn=loss_fn,
            optimize_fn=optimize_fn,
            default_bounds=default_bounds,
            n_jobs=n_jobs,
            optimizer_params=optimizer_params,
            **kwargs,
        )
        self.robust = robust


    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            tp_costs: Optional[ArrayLike] = 0.0,
            tn_costs: Optional[ArrayLike] = 0.0,
            fn_costs: Optional[ArrayLike] = 0.0,
            fp_costs: Optional[ArrayLike] = 0.0,
            **loss_params
    ) -> 'CSLogitClassifier':
        """

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        tp_costs : 1D array-like, shape=(n_samples,), optional
        Costs for true positive predictions.
        tn_costs : 1D array-like, shape=(n_samples,), optional
            Costs for true negative predictions.
        fn_costs : 1D array-like, shape=(n_samples,), optional
            Costs for false negative predictions.
        fp_costs : 1D array-like, shape=(n_samples,), optional
            Costs for false positive predictions.

        Returns
        -------
        self

        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = np.unique(y)

        if self.fit_intercept and not np.all(X[:, 0] == 1):
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        n_dim = X.shape[1]

        optimize_fn = partial(self.optimize_fn, **self.optimizer_params)
        if 'bounds' not in optimize_fn.keywords:
            optimize_fn = partial(optimize_fn, bounds=[self.default_bounds] * n_dim)
        elif len(optimize_fn.keywords['bounds']) != n_dim:
            raise ValueError(
                f"Number of bounds ({len(optimize_fn.keywords['bounds'])}) "
                f"must match number of features ({n_dim})."
            )

        if self.robust:
            raise NotImplementedError("Robust version is not implemented yet.")
            # cost_matrix = self.transform_cost_matrix(X, y, cost_matrix)

        # if loss_fn takes cost_matrix as an argument, pass it
        loss_fn_params = inspect.signature(self.loss_fn).parameters
        n_cost_params = 0
        if 'cost_matrix' in loss_fn_params:
            loss_params['cost_matrix'] = build_cost_matrix(tp_costs, tn_costs, fn_costs, fp_costs)
            n_cost_params += 1
        elif 'cost_mat' in loss_fn_params:
            loss_params['cost_mat'] = build_cost_matrix(tp_costs, tn_costs, fn_costs, fp_costs)
            n_cost_params += 1
        else:
            for param, cost in zip(('tp_costs', 'tn_costs', 'fn_costs', 'fp_costs'), (tp_costs, tn_costs, fn_costs, fp_costs)):
                if param in inspect.signature(self.loss_fn).parameters :
                    loss_params[param] = cost
                    n_cost_params += 1
        if n_cost_params == 0:
            warnings.warn(
                "loss function does not take 'cost_matrix', 'cost_mat', "
                "'tp_costs', 'tn_costs', 'fn_costs', or 'fp_costs' as an argument. "
                "Consider adding costs to the loss function signature.",
                UserWarning,
            )

        objective = partial(
            _objective,
            X=X,
            y=y,
            loss_fn=partial(self.loss_fn, **loss_params),
            C=self.C,
            l1_ratio=self.l1_ratio,
            soft_threshold=self.soft_threshold,
            fit_intercept=self.fit_intercept,
        )

        self.result = optimize_fn(objective, rng=self.random_state, **self.optimizer_params)

        return self

def transform_cost_matrix(x: NDArray, y: NDArray, cost_matrix: NDArray) -> NDArray[np.float64]:
    # Transform cost matrix
    # create df: a merge of x, y and amount
    amounts = cost_matrix[:, 0, 1]
    df = pd.DataFrame(data=x)
    df['y'] = y
    df['Amount'] = amounts

    #step 1: fit model on df, return df_enhanced
    df_enhanced = enhance_df(df)
    #step 2: replace conditionally outlying amounts with estimated amounts
    df = impute_amounts(df_enhanced)
    #step 3: return cost_matrix_transformed
    cost_matrix_transformed = get_cost_matrix(df['Amount'])
    return cost_matrix_transformed


def build_cost_matrix(tp_costs, tn_costs, fn_costs, fp_costs):
    cost_matrix = np.zeros((len(tp_costs), 2, 2))  # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = tn_costs
    cost_matrix[:, 0, 1] = fn_costs
    cost_matrix[:, 1, 0] = fp_costs
    cost_matrix[:, 1, 1] = tp_costs
    return cost_matrix


def _objective(weights, X, y, loss_fn, C, l1_ratio, soft_threshold, fit_intercept):
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

    logits = np.dot(X, weights)
    y_pred = expit(logits)
    # y_pred = 1 / (1 + np.exp(-logits))  # Invert logit transformation
    loss = loss_fn(y, y_pred)
    regularization_term = 0.5 * (1 - l1_ratio) * np.sum(b ** 2) + l1_ratio * np.sum(np.abs(b))
    penalty = regularization_term / C
    return loss + penalty


def get_cost_matrix(cost):
    cost_matrix = np.zeros((len(cost), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0
    cost_matrix[:, 0, 1] = cost
    cost_matrix[:, 1, 0] = cost
    cost_matrix[:, 1, 1] = 0

    return cost_matrix

def enhance_df(df):

    model_0 = HuberRegressor(max_iter=1000)
    model_1 = HuberRegressor(max_iter=1000)

    df_0 = df[df.y == 0]
    model_0.fit(df_0.loc[:, df_0.columns != 'Amount'], df_0['Amount'])
    Ahat_0_array = model_0.predict(df_0.loc[:, df_0.columns != 'Amount'])
    Ahat_0_array[Ahat_0_array<0] = 0

    df_1 = df[df.y == 1]
    model_1.fit(df_1.loc[:, df_1.columns != 'Amount'], df_1['Amount'])
    Ahat_1_array = model_1.predict(df_1.loc[:, df_1.columns != 'Amount'])
    Ahat_1_array[Ahat_1_array < 0] = 0

    df_0['Amount_pred'] = Ahat_0_array
    df_1['Amount_pred'] = Ahat_1_array
    prediction = pd.concat([df_0,df_1])
    df['Amount_pred'] = prediction['Amount_pred']

    return df

#Function to detect and impute conditionally outlying amount
def impute_amounts(df):

    std = st.sem(df['Amount'])
    df['resid'] = df['Amount'] - df['Amount_pred']
    df['resid_std'] = df['resid']/std
    df['resid_std_abs'] = abs(df['resid_std'])
    df.loc[df.resid_std_abs > 3, 'Amount'] = df['Amount_pred']
    print("")
    df = df.drop(columns=['Amount_pred','resid','resid_std','resid_std_abs'])
    return df


# def _optimize(objective, X, y, sample_weight, C, bounds, fit_intercept, max_iter=10000, tolerance=1e-4, **kwargs) -> OptimizeResult:
#     n_threads = 1
#     n_samples, n_features = X.shape
#
#     initial_weights = np.zeros(
#         (1, n_features + int(fit_intercept)), order="F", dtype=X.dtype
#     )
#
#     sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
#
#     l2_reg_strength = 1.0 / (C * sw_sum)
#     result = minimize(
#         objective,
#         initial_weights,
#         method="L-BFGS-B",
#         jac=True,
#         args=(X, y, sample_weight, l2_reg_strength, n_threads),
#         options={
#             "maxiter": max_iter,
#             "maxls": 50,  # default is 20
#             "gtol": tolerance,
#             "ftol": 64 * np.finfo(float).eps,
#         },
#         **kwargs,
#     )
#     _check_optimize_result(result,)
#
#     return result


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

