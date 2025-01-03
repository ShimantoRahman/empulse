import warnings

import numpy as np
import scipy.stats as st
from numpy.typing import ArrayLike

from ._validation import _validate_input_stochastic
from ..common import _compute_prior_class_probabilities, _compute_tpr_fpr_diffs
from .._convex_hull import _compute_convex_hull


def empa_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        alpha: float = 12,
        beta: float = 0.0015,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
        check_input: bool = True
) -> float:
    """
    :func:`~empulse.metrics.empa()` but only returning the EMPA score.

    EMPA presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The contribution of a successful acquisition is modeled as a :math:`Gamma(\\alpha, \\beta)` distribution.

    .. seealso::

        :func:`~empulse.metrics.empa` : to also return the fraction of the leads
        that should be targeted to maximize profit.

        :func:`~empulse.metrics.mpa_score` : for a deterministic version of this metric.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=10
        Shape parameter of the gamma distribution of the average contribution of a new customer. (``alpha > 0``)

    beta : float, default=10
        Rate parameter of the gamma distribution of the average contribution of a new customer. (``beta > 0``)

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (``sales_cost ≥ 0``).

    contact_cost : float, default=50
        Average contact cost of targeted leads (``contact_cost ≥ 0``).

    direct_selling : float, default=1
        Fraction of leads sold to directly (``0 ≤ direct_selling ≤ 1``).
        `direct_selling` = 0 for indirect channel.
        `direct_selling` = 1 for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermedaries (``0 ≤ commission ≤ 1``).

        .. note::
            The commission is only relevant when there is an indirect channel (``direct_selling < 1``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empa : tuple[float]
        Expected Maximum Profit measure for customer Acquisition.

    Notes
    -----
    The EMPA is defined as:

    .. math::

        \\int_{R} [[ \\rho(R-c-S)+(1-\\rho)(\\gamma R - c)] \\pi_0 F_0(t) - c \\pi_1 F_1(t)] \\cdot g(CLV) \\, dCLV

    The EMPA requires that the acquisition class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('acquisition': 1, 'no acquisition': 0).

    Examples
    --------
    Direct channel (rho = 1):

    >>> from empulse.metrics import empa_score
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empa_score(y_true, y_score, direct_selling=1)
    3706.2500000052773

    Indirect channel using scorer (rho = 0):

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
    ...     empa_score,
    ...     response_method='predict_proba',
    ...     alpha=10,
    ...     beta=0.001,
    ...     sales_cost=2_000,
    ...     contact_cost=100,
    ...     direct_selling=0,
    ... )
    >>> np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer))
    4449.0
    """
    return empa(
        y_true,
        y_score,
        alpha=alpha,
        beta=beta,
        contact_cost=contact_cost,
        sales_cost=sales_cost,
        direct_selling=direct_selling,
        commission=commission,
        check_input=check_input,
    )[0]


def empa(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        alpha: float = 12,
        beta: float = 0.0015,
        contact_cost: float = 50,
        sales_cost: float = 500,
        direct_selling: float = 1,
        commission: float = 0.1,
        check_input: bool = True
) -> tuple[float, float]:
    """
    Expected Maximum Profit measure for customer Acquisition (EMPA).

    EMPA presumes a situation where leads are targeted either directly or indirectly.
    Directly targeted leads are contacted and handled by the internal sales team.
    Indirectly targeted leads are contacted and then referred to intermediaries,
    which receive a commission.
    The contribution of a successful acquisition is modeled as a :math:`Gamma(\\alpha, \\beta)` distribution.

    .. seealso::

        :func:`~empulse.metrics.empa_score` : to only return the EMPA score.

        :func:`~empulse.metrics.mpa` : for a deterministic version of this metric.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('acquisition': 1, 'no acquisition': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    alpha : float, default=10
        Shape parameter of the gamma distribution of the average contribution of a new customer. (``alpha > 0``)

    beta : float, default=10
        Rate parameter of the gamma distribution of the average contribution of a new customer. (``beta > 0``)

    sales_cost : float, default=500
        Average sale conversion cost of targeted leads handled by the company (``sales_cost ≥ 0``).

    contact_cost : float, default=50
        Average contact cost of targeted leads (``contact_cost ≥ 0``).

    direct_selling : float, default=1
        Fraction of leads sold to directly (``0 ≤ direct_selling ≤ 1``).
        `direct_selling` = 0 for indirect channel.
        `direct_selling` = 1 for direct channel.

    commission : float, default=0.1
        Fraction of contribution paid to the intermedaries (``0 ≤ commission ≤ 1``).

        .. note::
            The commission is only relevant when there is an indirect channel (``direct_selling < 1``).

    check_input : bool, default=True
        Perform input validation.
        Turning off improves performance, useful when using this metric as a loss function.

    Returns
    -------
    empa : float
        Expected Maximum Profit measure for customer Acquisition

    threshold : float
        Fraction of the leads that should be targeted to maximize profit

    Notes
    -----
    The EMPA is defined as:

    .. math::

        \\int_{R} [[ \\rho(R-c-S)+(1-\\rho)(\\gamma R - c)] \\pi_0 F_0(t) - c \\pi_1 F_1(t)] \\cdot g(CLV) \\, dCLV

    The EMPA requires that the acquisition class is encoded as 0, and it is NOT interchangeable.
    However, this implementation assumes the standard notation ('acquisition': 1, 'no acquisition': 0).

    Examples
    --------
    Direct channel (rho = 1):

    >>> from empulse.metrics import empa
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empa(y_true, y_score, direct_selling=1)
    (3706.2500000052773, 0.8749999997947746)

    Indirect channel (rho = 0):

    >>> from empulse.metrics import empa
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>> empa(y_true, y_score, direct_selling=0)
    (3556.25, 0.875)
    """

    if check_input:
        y_true, y_score = _validate_input_stochastic(
            y_true,
            y_score,
            alpha,
            beta,
            contact_cost,
            sales_cost,
            direct_selling,
            commission
        )
    else:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    positive_class_prob, negative_class_prob = _compute_prior_class_probabilities(y_true)

    true_positive_rates, false_positive_rates = _compute_convex_hull(y_true, y_score, expand_dims=True)
    tpr_diff, fpr_diff = _compute_tpr_fpr_diffs(true_positive_rates, false_positive_rates)

    fpr_coef = contact_cost * negative_class_prob
    tpr_coef = (-direct_selling * sales_cost - contact_cost) * positive_class_prob
    denominator = (direct_selling + (1 - direct_selling) * (1 - commission)) * positive_class_prob

    bounds = _compute_integration_bounds(tpr_coef, fpr_coef, denominator, tpr_diff, fpr_diff)
    cdf_diff = np.diff(st.gamma.cdf(bounds, a=alpha, loc=0, scale=1 / beta), axis=0)
    cdf_1_diff = np.diff(st.gamma.cdf(bounds, a=alpha + 1, loc=0, scale=1 / beta), axis=0)

    cdf_coef = (tpr_coef * true_positive_rates - fpr_coef * false_positive_rates)
    cdf_1_coef = denominator * true_positive_rates

    expected_profit = ((alpha / beta) * (cdf_1_coef * cdf_1_diff).sum(axis=0) + (cdf_coef * cdf_diff).sum(axis=0))

    threshold = (
            cdf_diff * (positive_class_prob * true_positive_rates + negative_class_prob * false_positive_rates)
    ).sum()

    return expected_profit.sum(), threshold


def _compute_integration_bounds(
        tpr_coef: float,
        fpr_coef: float,
        denominator: float,
        tpr_diff: np.ndarray,
        fpr_diff: np.ndarray,
) -> np.ndarray:
    """Compute the integration bounds for the contribution of a new customer."""
    # ignore division by zero warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        clv_bounds = (fpr_coef * fpr_diff - tpr_coef * tpr_diff) / (denominator * tpr_diff)
    # add zero and infinity to bounds
    if clv_bounds.ndim == 2:
        return np.concatenate([
            np.zeros((1, clv_bounds.shape[1])),
            clv_bounds,
            np.full(shape=(1, clv_bounds.shape[1]), fill_value=np.inf)
        ])
    elif clv_bounds.ndim == 1:
        return np.concatenate([[0], clv_bounds, [np.inf]]).reshape(-1, 1)
    else:
        raise ValueError(f"Invalid number of dimensions: {clv_bounds.ndim}")
