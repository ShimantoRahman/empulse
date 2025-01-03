import warnings
from typing import Union
from packaging.version import Version

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as st

from ._validation import _validate_input_emp, _validate_input_empb
from .._convex_hull import _compute_convex_hull
from ..common import _compute_prior_class_probabilities, _compute_tpr_fpr_diffs

if Version(np.version.version) >= Version("2.0.0"):
    np.trapz = np.trapezoid


def empc_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        alpha: float = 6,
        beta: float = 14,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
        check_input: bool = True,
) -> float:
    """
    :func:`~empulse.metrics.empc()` but only returning the EMPC score.

    EMPC presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer,
    this fraction is described by a :math:`Beta(\\alpha, \\beta)` distribution.
    As opposed to :func:`~empulse.metrics.empb`, the incentive cost is a fixed value,
    rather than a fraction of the customer lifetime value. For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.empc` : to also return the fraction of the customer base
        that should be targeted to maximize profit.

        :func:`~empulse.metrics.mpc_score` : for a deterministic version of this metric.

        :func:`~empulse.metrics.empb_score` : for a similar metric, but with a variable incentive cost.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: average customer lifetime value of retained customers (``clv > incentive_cost``).
        If ``array``: customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

        .. note::
            Passing a CLV array is equivalent to passing a float with the average CLV of that array.

    incentive_cost : float, default=10
        Cost of incentive offered to a customer (``incentive_cost > 0``).

    contact_cost : float, default=1
        Cost of contacting a customer (``contact_cost > 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empc : float
        Expected Maximum Profit Measure for Customer Churn.

    Notes
    -----
    The EMPC is defined as [1]_:

    .. math:: \\int_\\gamma CLV (\\gamma (1 - \\delta) - \\phi) \\pi_0 F_0(T) - CLV (\\delta + \\phi) \\pi_1 F_1(T) d\\gamma

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
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empc_score(y_true, y_score)
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
    >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> scorer = make_scorer(
    ...     empc_score,
    ...     response_method='predict_proba',
    ...     clv=300,
    ...     incentive_cost=15,
    ... )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    42.09000050753503
    """
    return empc(
        y_true,
        y_score,
        alpha=alpha,
        beta=beta,
        clv=clv,
        incentive_cost=incentive_cost,
        contact_cost=contact_cost,
        check_input=check_input,
    )[0]


def empc(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        alpha: float = 6,
        beta: float = 14,
        clv: Union[float, ArrayLike] = 200,
        incentive_cost: float = 10,
        contact_cost: float = 1,
        check_input: bool = True,
) -> tuple[float, float]:
    """
    Expected Maximum Profit Measure for Customer Churn (EMPC).

    EMPC presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer,
    this fraction is described by a :math:`Beta(\\alpha, \\beta)` distribution.
    As opposed to :func:`~empulse.metrics.empb`, the incentive cost is a fixed value,
    rather than a fraction of the customer lifetime value. For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.empc_score` : to only return the EMPC score.

        :func:`~empulse.metrics.mpc` : for a deterministic version of this metric.

        :func:`~empulse.metrics.empb` : for a similar metric, but with a variable incentive cost.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples), default=200
        If ``float``: average customer lifetime value of retained customers (``clv > incentive_cost``).
        If ``array``: customer lifetime value of each customer when retained
        (``mean(clv) > incentive_cost``).

        .. note::
            Passing a CLV array is equivalent to passing a float with the average CLV of that array.

    incentive_cost : float, default=10
        Cost of incentive offered to a customer (``incentive_cost > 0``).

    contact_cost : float, default=1
        Cost of contacting a customer (``contact_cost > 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empc : float
        Expected Maximum Profit Measure for Customer Churn

    threshold : float
        Fraction of the customer base that should be targeted to maximize profit

    Notes
    -----
    The EMPC is defined as [1]_:

    .. math:: \\int_\\gamma CLV (\\gamma (1 - \\delta) - \\phi) \\pi_0 F_0(T) - CLV (\\delta + \\phi) \\pi_1 F_1(T) d\\gamma

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
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empc(y_true, y_score)
    (23.875593418348124, 0.8743700763487141)
    """
    if check_input:
        y_true, y_score, clv = _validate_input_emp(y_true, y_score, alpha, beta, clv, incentive_cost, contact_cost)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)

    if isinstance(clv, np.ndarray):
        clv = clv.mean()

    delta = incentive_cost / clv
    phi = contact_cost / clv
    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score)
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
        y_score: ArrayLike,
        *,
        clv: ArrayLike,
        alpha: float = 6,
        beta: float = 14,
        incentive_fraction: float = 0.05,
        contact_cost: float = 15,
        check_input: bool = True,
) -> float:
    """
    :func:`~empulse.metrics.empb()` but only returning the EMPB score.

    EMPB presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer,
    this fraction is described by a :math:`Beta(\\alpha, \\beta)` distribution.
    As opposed to :func:`~empulse.metrics.empc`, the incentive cost is a fraction of the customer lifetime value,
    rather than a fixed value. For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.empb` : to also return the fraction of the customer base
        that should be targeted to maximize profit.

        :func:`~empulse.metrics.empc_score` : for a similar metric, but with a fixed incentive cost.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples)
        If ``float``: average customer lifetime value of retained customers.
        If ``array``: customer lifetime value of each customer when retained.

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).

    contact_cost : float, default=15
        Cost of contacting a customer (``contact_cost > 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empb : float
        Expected Maximum Profit Measure for B2B Customer Churn

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    return empb(
        y_true,
        y_score,
        alpha=alpha,
        beta=beta,
        clv=clv,
        contact_cost=contact_cost,
        incentive_fraction=incentive_fraction,
        check_input=check_input,
    )[0]


def empb(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        clv: ArrayLike,
        alpha: float = 6,
        beta: float = 14,
        incentive_fraction: float = 0.05,
        contact_cost: float = 15,
        check_input: bool = True,
) -> tuple[float, float]:
    """
    Expected Maximum Profit Measure for B2B Customer Churn (EMPB).

    EMPB presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer,
    this fraction is described by a :math:`Beta(\\alpha, \\beta)` distribution.
    As opposed to :func:`~empulse.metrics.empc`, the incentive cost is a fraction of the customer lifetime value,
    rather than a fixed value. For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.empb_score` : to only return the EMPB score.

        :func:`~empulse.metrics.empc` : for a similar metric, but with a fixed incentive cost.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples)
        If ``float``: average customer lifetime value of retained customers.
        If ``array``: customer lifetime value of each customer when retained.

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).

    contact_cost : float, default=15
        Cost of contacting a customer (``contact_cost > 0``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empb : float
        Expected Maximum Profit Measure for B2B Customer Churn

    threshold : float
        Fraction of the customer base that should be targeted to maximize profit

    References
    ----------
    .. [1] Janssens, B., Bogaert, M., Bagué, A., & Van den Poel, D. (2022).
        B2Boost: Instance-dependent profit-driven modelling of B2B churn.
        Annals of Operations Research, 1-27.
    """
    if check_input:
        y_true, y_score, clv = _validate_input_empb(y_true, y_score, clv, alpha, beta, incentive_fraction, contact_cost)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)
    gamma = alpha / (alpha + beta)

    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true[sorted_indices]
    sorted_clv = clv[sorted_indices]

    # Calculate cumulative sums for benefits and costs
    cumulative_benefits = np.cumsum(gamma * ((1 - incentive_fraction) * sorted_clv - contact_cost) * sorted_y_true)
    cumulative_costs = np.cumsum((-contact_cost - incentive_fraction * sorted_clv) * (1 - sorted_y_true))
    cumulative_profits = cumulative_benefits + cumulative_costs

    # Add a zero at the beginning to indicate not contacting anyone
    cumulative_profits = np.insert(cumulative_profits, 0, 0)

    # Find the maximum profit and corresponding threshold
    max_profit_index = np.argmax(cumulative_profits)
    max_profit = cumulative_profits[max_profit_index]
    threshold = max_profit_index / len(y_score)

    return float(max_profit), float(threshold)


def auepc_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        clv: ArrayLike,
        alpha: float = 6,
        beta: float = 14,
        incentive_fraction: float = 0.05,
        contact_cost: float = 15,
        normalize: bool = True,
        check_input: bool = True,
) -> float:
    """
    Area Under the Expected Profit Curve (AUEPC).

    Calculate the area under the ratio of the expected profit of the model and the perfect model.
    The expected profit is based on the EMPB's definition of profit.

    AUEPC presumes a situation where identified churners are contacted and offered an incentive to remain customers.
    Only a fraction of churners accepts the incentive offer,
    this fraction is described by a :math:`Beta(\\alpha, \\beta)` distribution.
    For detailed information, consult the paper [1]_.

    .. seealso::

        :func:`~empulse.metrics.empb` : to return the maximum profit and threshold.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('churn': 1, 'no churn': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=6
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``alpha > 1``).

    beta : float, default=14
        Shape parameter of the beta distribution of the probability
        that a churner accepts the incentive (``beta > 1``).

    clv : float or 1D array-like, shape=(n_samples)
        If ``float``: average customer lifetime value of retained customers.
        If ``array``: customer lifetime value of each customer when retained.

    incentive_fraction : float, default=0.05
        Cost of incentive offered to a customer, as a fraction of customer lifetime value
        (``0 < incentive_fraction < 1``).

    contact_cost : float, default=15
        Cost of contacting a customer (``contact_cost > 0``).

    normalize : bool, default=True
        Whether to normalize the AUEPC score. If True, the score is 1 when the model is perfect.
        This parameter is only useful if a part of the expected profit curve is negative.

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empb : float
        Expected Maximum Profit Measure for B2B Customer Churn

    threshold : float
        Fraction of the customer base that should be targeted to maximize profit

    References
    ----------
    .. [1] Rahman, S., Janssens, B., Bogaert, M. (2025).
        Profit-Driven Pre-Processing in B2B Customer Churn Modeling using Fairness Techniques.
        Journal of Business Research.
    """
    if check_input:
        y_true, y_score, clv = _validate_input_empb(y_true, y_score, clv, alpha, beta, incentive_fraction, contact_cost)
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        clv = np.asarray(clv)
    if clv.ndim > 1:
        clv = clv[:, 0]

    accept_rate = alpha / (alpha + beta)

    # Calculate the expected profit vector for the perfect model
    perfect_pred_indices = np.argsort(np.where(y_true == 1, 1, -1) * clv)[::-1]
    perfect_targets = y_true[perfect_pred_indices]
    perfect_clv_targets = clv[perfect_pred_indices]

    perfect_benefits = np.cumsum(
        accept_rate * ((1 - incentive_fraction) * perfect_clv_targets - contact_cost) * perfect_targets
    )
    perfect_costs = np.cumsum((-contact_cost - incentive_fraction * perfect_clv_targets) * (1 - perfect_targets))
    perfect_profits = perfect_benefits + perfect_costs

    # Calculate the expected profit vector for the perfect model
    sorted_indices = y_score.argsort()[::-1]
    targets = y_true[sorted_indices]
    clv_targets = clv[sorted_indices]

    benefits = np.cumsum(
        accept_rate * ((1 - incentive_fraction) * clv_targets - contact_cost) * targets
    )
    costs = np.cumsum((-contact_cost - incentive_fraction * clv_targets) * (1 - targets))
    profits = benefits + costs

    # Stop at the point where perfect profits become negative
    stop_index = np.argmax(perfect_profits < 0) if np.any(perfect_profits < 0) else len(perfect_profits)

    # Calculate the AUEPC
    score = np.trapz(profits[:stop_index] / perfect_profits[:stop_index], dx=1 / len(profits))
    if normalize:
        score /= (stop_index - 1) / len(profits)
    return score
