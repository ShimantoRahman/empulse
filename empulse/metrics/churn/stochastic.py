import warnings
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as st

from ._validation import _validate_input_emp, _validate_input_empb
from ..common import _compute_prior_class_probabilities, _compute_tpr_fpr_diffs, _range
from .._convex_hull import _compute_convex_hull


def empc_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        alpha: float = 6,
        beta: float = 14,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
) -> float:
    """
    Convenience function around :func:`~empulse.metrics.empc()` only returning EMPC score

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

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
    empc : float
        Expected Maximum Profit Measure for Customer Churn.

    Notes
    -----
    The EMPC is defined as [1]_:

    .. math:: \int_\gamma CLV (\gamma (1 - \delta) - \phi) \pi_0 F_0(T) - CLV (\delta + \phi) \pi_1 F_1(T) d\gamma

    The EMPC requires that the churn class is encoded as 0, and it is NOT interchangeable (see [3]_ p37).
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    An equivalent R implementation is available in [2]_.

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289
    .. [2] Bravo, C. and Vanden Broucke, S. and Verbraken, T. (2019).
        EMP: Expected Maximum Profit Classification Performance Measure.
        R package version 2.0.5. Available Online:
        http://cran.r-project.org/web/packages/EMP/index.html
    .. [3] Verbraken, T. (2013). Business-Oriented Data Analytics:
        Theory and Case Studies. Ph.D. dissertation, Dept. LIRIS, KU Leuven,
        Leuven, Belgium, 2013.

    Examples
    --------
    >>> from empulse.metrics import empc_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empc_score(y_true, y_pred)
    23.875593418348124

    Using scorer:

    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_val_score, StratifiedKFold
    >>> from sklearn.metrics import make_scorer
    >>> from empulse.metrics import empa_score
    >>>
    >>> X, y = make_classification(random_state=42)
    >>> model = LogisticRegression()
    >>> cv = StratifiedKFold(n_splits=5, random_state=42)
    >>> scorer = make_scorer(
    >>>     empa_score,
    >>>     needs_proba=True,
    >>>     clv=300,
    >>>     incentive_cost=15,
    >>> )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    42.09000050753503
    """
    return empc(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
    )[0]


def empc(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        alpha: float = 6,
        beta: float = 14,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
) -> tuple[float, float]:
    """
    Expected Maximum Profit Measure for Customer Churn (EMPC)

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

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
    empc : float
        Expected Maximum Profit Measure for Customer Churn

    threshold : float
        Threshold at which the expected maximum profit is achieved

    Notes
    -----
    The EMPC is defined as [1]_:

    .. math:: \int_\gamma CLV (\gamma (1 - \delta) - \phi) \pi_0 F_0(T) - CLV (\delta + \phi) \pi_1 F_1(T) d\gamma

    The EMPC requires that the churn class is encoded as 0, and it is NOT interchangeable (see [3]_ p37).
    However, this implementation assumes the standard notation ('churn': 1, 'no churn': 0).

    Code adapted from [4]_.
    An equivalent R implementation is available in [2]_.

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289
    .. [2] Bravo, C. and Vanden Broucke, S. and Verbraken, T. (2019).
        EMP: Expected Maximum Profit Classification Performance Measure.
        R package version 2.0.5. Available Online:
        http://cran.r-project.org/web/packages/EMP/index.html
    .. [3] Verbraken, T. (2013). Business-Oriented Data Analytics:
        Theory and Case Studies. Ph.D. dissertation, Dept. LIRIS, KU Leuven,
        Leuven, Belgium, 2013.
    .. [4] https://github.com/estripling/proflogit/blob/master/proflogit/empc.py

    Examples
    --------
    >>> from empulse.metrics import empc
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empc(y_true, y_pred)
    (23.875593418348124, 0.8743700763487141)
    """
    y_true, y_pred, clv = _validate_input_emp(y_true, y_pred, alpha, beta, clv, incentive_cost, contact_cost)

    if isinstance(clv, np.ndarray):
        clv = clv.mean()

    delta = incentive_cost / clv
    phi = contact_cost / clv
    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_pred)
    tpr_diff, fpr_diff = _compute_tpr_fpr_diffs(true_positive_rates, false_positive_rates)

    tpr_coef = phi * positive_class_prob
    fpr_coef = (delta + phi) * negative_class_prob

    gamma_bounds = _compute_gamma_bounds(tpr_coef, fpr_coef, delta, tpr_diff, fpr_diff, positive_class_prob)
    gamma_cdf_diff = np.diff(st.beta.cdf(gamma_bounds, a=alpha, b=beta))
    gamma_cdf_1_diff = np.diff(st.beta.cdf(gamma_bounds, a=alpha + 1, b=beta))

    cutoff = len(true_positive_rates) - len(gamma_cdf_diff)
    if cutoff > 0:
        true_positive_rates = true_positive_rates[:-cutoff]
        false_positive_rates = false_positive_rates[:-cutoff]
    mean_gamma = st.beta.mean(a=alpha, b=beta)
    N = mean_gamma * (clv * (1 - delta) * positive_class_prob * true_positive_rates)
    M = clv * (tpr_coef * true_positive_rates + fpr_coef * false_positive_rates)
    empc = (N * gamma_cdf_1_diff - M * gamma_cdf_diff).sum()

    customer_threshold = (
            gamma_cdf_diff * (positive_class_prob * true_positive_rates + negative_class_prob * false_positive_rates)
    ).sum()

    return empc, customer_threshold


def _compute_gamma_bounds(
        tpr_coef: float,
        fpr_coef: float,
        delta: float,
        tpr_diff: np.ndarray,
        fpr_diff: np.ndarray,
        positive_class_prob: float
) -> np.ndarray:
    """Compute the gamma bounds of the integral"""
    numerator = fpr_coef * fpr_diff + tpr_coef * tpr_diff
    denominator = positive_class_prob * (1 - delta) * tpr_diff
    # ignore division by zero warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        gamma_bounds = numerator / denominator
    gamma_bounds = np.append([0], gamma_bounds)
    return np.append(gamma_bounds[gamma_bounds < 1], [1])


def empb_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        clv: ArrayLike,
        alpha: float = 6,
        beta: float = 14,
        contact_cost: float = 15,
        incentive_cost_fraction: float = 0.05,
        n_buckets: int = 250
) -> float:
    """
    Convenience function around :func:`~empulse.metrics.empb()` only returning EMPB score

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
        If ``array``: individualized customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

    incentive_cost_fraction : float, default=10
        Fraction of the customer lifetime value that is used as the incentive cost (``incentive_cost_fraction > 0``).

    contact_cost : float, default=1
        Constant cost of contact (``contact_cost > 0``).

    n_buckets : int, default=250
        Number of buckets to use for the calculation of the EMPB.

    Returns
    -------
    empb : float
        Expected Maximum Profit Measure for B2B Customer Churn [1]_

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    return empb(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
        clv=clv,
        contact_cost=contact_cost,
        incentive_cost_fraction=incentive_cost_fraction,
        n_buckets=n_buckets
    )[0]


def empb(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        *,
        clv: ArrayLike,
        alpha: float = 6,
        beta: float = 14,
        contact_cost: float = 15,
        incentive_cost_fraction: float = 0.05,
        n_buckets: int = 250
) -> tuple[float, float]:
    """
    Expected Maximum Profit Measure for B2B Customer Churn (EMPB) [1]_

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_pred : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: constant customer lifetime value per retained customer (``clv > incentive_cost``).
        If ``array``: individualized customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

    incentive_cost_fraction : float, default=10
        Fraction of the customer lifetime value that is used as the incentive cost (``incentive_cost_fraction > 0``).

    contact_cost : float, default=1
        Constant cost of contact (``contact_cost > 0``).

    n_buckets : int, default=250
        Number of buckets to use for the calculation of the EMPB.

    Returns
    -------
    empb : float
        Expected Maximum Profit Measure for B2B Customer Churn

    threshold : float
        Threshold at which the expected maximum profit is achieved

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    y_true, y_pred, clv = _validate_input_empb(y_true, y_pred, clv, alpha, beta, incentive_cost_fraction, contact_cost)
    step_size = 1 / n_buckets

    gammas = np.arange(0, 1, step_size)
    gamma_weights = st.beta.pdf(gammas, alpha, beta) * gammas * step_size

    sorted_indices = y_pred.argsort()[::-1]

    emp = -np.inf
    threshold = 0
    for fraction_targeted in _range(0, 1, 0.005):
        n_targeted = round(fraction_targeted * len(sorted_indices))
        targeted_indices = sorted_indices[0:n_targeted]
        targets = y_true[targeted_indices]
        clv_targets = clv[targeted_indices]

        benefit = np.sum(
            np.sum(
                np.expand_dims(gamma_weights, axis=1) *
                ((1 - incentive_cost_fraction) * np.expand_dims(clv_targets, axis=0) - contact_cost),
                axis=0) * targets
        )
        cost = np.sum((-contact_cost - incentive_cost_fraction * clv_targets) * (1 - targets))
        profit = benefit + cost

        if profit > emp:
            emp = profit
            threshold = fraction_targeted

    return emp, threshold


