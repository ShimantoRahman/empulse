"""Domain cost-matrix factory functions for the empulse dataset loaders.

Each factory returns a ``(CostMatrix, instance_costs)`` pair that can be
passed directly to :class:`~empulse.datasets.Dataset`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import sympy as sp

from ..metrics.metric.cost_matrix import CostMatrix

if TYPE_CHECKING:
    from .._types import FloatNDArray


def churn_precomputed_cost_matrix() -> CostMatrix:
    """Symbolic cost matrix for datasets that store all four costs directly.

    Symbols used: ``tp_cost``, ``fp_cost``, ``tn_cost``, ``fn_cost``.
    """
    tp_c, fp_c, tn_c, fn_c = sp.symbols('tp_cost fp_cost tn_cost fn_cost')
    return CostMatrix().add_tp_cost(tp_c).add_fp_cost(fp_c).add_tn_cost(tn_c).add_fn_cost(fn_c)


def churn_retention_cost_matrix(
    clv: FloatNDArray,
    *,
    incentive_fraction: float,
    contact_cost: float,
    accept_rate: float,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """Churn retention cost matrix (Bahnsen et al. 2015).

    Parameters
    ----------
    clv : array-like
        Per-customer customer lifetime value.
    incentive_fraction : float
        Fraction of CLV offered as retention incentive (:math:`d`).
    contact_cost : float
        The cost of contacting the customer (:math:`f`).
    accept_rate : float
        Probability that a churner accepts the retention offer (:math:`\\gamma`).

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'clv': array}``
    """
    clv_sym, d_sym, f_sym, gamma_sym = sp.symbols('clv d f gamma')
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma_sym * (clv_sym - d_sym * clv_sym - f_sym))
        .add_tp_benefit(-(1 - gamma_sym) * f_sym)
        .add_fp_cost(d_sym * clv_sym + f_sym)
        .alias({'incentive_fraction': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost,
            accept_rate=accept_rate,
        )
    )
    return cost_matrix, {'clv': clv}


def churn_retention_monthly_cost_matrix(
    monthly_revenue: FloatNDArray,
    *,
    clv_months: float,
    incentive_fraction: float,
    contact_cost: float,
    accept_rate: float,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """Churn retention cost matrix with the CLV approximated from monthly revenue.

    Same framing as :func:`churn_retention_cost_matrix` (Verbraken et al. 2013), for
    datasets that record a customer's monthly revenue but not their lifetime value:
    :math:`CLV_i` is replaced by :math:`m \\cdot MonthlyRevenue_i`.

    Parameters
    ----------
    monthly_revenue : array-like
        Per-customer monthly revenue.
    clv_months : float
        Number of months of revenue that make up the CLV (:math:`m`).
    incentive_fraction : float
        Fraction of CLV offered as retention incentive (:math:`d`).
    contact_cost : float
        The cost of contacting the customer (:math:`f`).
    accept_rate : float
        Probability that a churner accepts the retention offer (:math:`\\gamma`).

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'monthly_revenue': array}``
    """
    revenue_sym, months_sym, d_sym, f_sym, gamma_sym = sp.symbols('monthly_revenue m d f gamma')
    clv = months_sym * revenue_sym
    cost_matrix = (
        CostMatrix()
        .add_tp_benefit(gamma_sym * (clv - d_sym * clv - f_sym))
        .add_tp_benefit(-(1 - gamma_sym) * f_sym)
        .add_fp_cost(d_sym * clv + f_sym)
        .alias({'clv_months': 'm', 'incentive_fraction': 'd', 'contact_cost': 'f', 'accept_rate': 'gamma'})
        .set_default(
            clv_months=clv_months,
            incentive_fraction=incentive_fraction,
            contact_cost=contact_cost,
            accept_rate=accept_rate,
        )
    )
    return cost_matrix, {'monthly_revenue': np.asarray(monthly_revenue, dtype=float)}


def upsell_bank_cost_matrix(
    balance: FloatNDArray,
    *,
    interest_rate: float,
    term_deposit_fraction: float,
    contact_cost: float,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """Upsell (bank telemarketing) cost matrix.

    Parameters
    ----------
    balance : array-like
        Per-client average yearly balance in euros.
    interest_rate : float
        Interest rate of the term deposit (:math:`r`).
    term_deposit_fraction : float
        Fraction of the client's balance deposited (:math:`d`).
    contact_cost : float
        Fixed cost of contacting the client (:math:`c`).

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'balance': array}``
    """
    balance_sym, r_sym, d_sym, c_sym = sp.symbols('balance r d c')
    cost_matrix = (
        CostMatrix()
        .add_tp_cost(c_sym)
        .add_fp_cost(c_sym)
        .add_fn_cost(sp.Max(r_sym * d_sym * balance_sym, c_sym))
        .alias({'interest_rate': 'r', 'term_deposit_fraction': 'd', 'contact_cost': 'c'})
        .set_default(
            interest_rate=interest_rate,
            term_deposit_fraction=term_deposit_fraction,
            contact_cost=contact_cost,
        )
    )
    return cost_matrix, {'balance': balance}


def credit_scoring_cost_matrix(
    monthly_income: FloatNDArray,
    debt_ratio: FloatNDArray,
    target_np: np.ndarray,
    *,
    interest_rate: float,
    fund_cost: float,
    cl_max: float,
    loss_given_default: float,
    term_length_months: int,
    loan_to_income_ratio: float,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """Credit-scoring cost matrix (Bahnsen et al. 2014).

    Parameters
    ----------
    monthly_income : array-like
        Per-instance monthly income (already scaled by the caller if needed).
    debt_ratio : array-like
        Per-instance debt ratio (0–1).
    target_np : array-like
        Binary target array used to compute the class-prior :math:`\\pi_1`.
    interest_rate : float
        Annual loan interest rate.
    fund_cost : float
        Annual cost of funds.
    cl_max : float
        Maximum credit line (already scaled by the caller if needed).
    loss_given_default : float
        Fraction of the credit line lost upon default.
    term_length_months : int
        Loan term in months.
    loan_to_income_ratio : float
        Ratio of loan amount to monthly income.

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'cl': credit_line_array, 'fp_cost': fp_cost_array}``
    """
    params = {
        'int_r': interest_rate / 12,
        'int_cf': fund_cost / 12,
        'cl_max': cl_max,
        'n_term': term_length_months,
        'k': loan_to_income_ratio,
        'lgd': loss_given_default,
    }
    pi_1 = float(target_np.mean())
    cost_mat = _creditscoring_costmat(monthly_income, debt_ratio, pi_1, params)
    cl_vals = _compute_credit_lines(monthly_income, debt_ratio, params)

    cl_sym, lgd_sym, fp_sym = sp.symbols('cl lgd fp_cost')
    cost_matrix = (
        CostMatrix()
        .add_fn_cost(cl_sym * lgd_sym)
        .add_fp_cost(fp_sym)
        .alias({'loss_given_default': 'lgd'})
        .set_default(loss_given_default=loss_given_default)
    )
    return cost_matrix, {'cl': cl_vals, 'fp_cost': cost_mat[:, 0]}


def credit_scoring_known_cl_cost_matrix(
    credit_line: FloatNDArray,
    target_np: np.ndarray,
    *,
    interest_rate: float = 0.0479,
    fund_cost: float = 0.0294,
    loss_given_default: float = 0.75,
    term_length_months: int = 24,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """
    Credit-scoring cost matrix when credit lines are already known.

    Follows Bahnsen et al. (2014) and Vanderschueren et al. (2022).

    Parameters
    ----------
    credit_line : array-like
        Per-instance credit line (e.g. ``LIMIT_BAL``).
    target_np : array-like
        Binary target array used to compute the class-prior :math:`\\pi_1`.
    interest_rate : float, default=0.0479
        Annual loan interest rate.
    fund_cost : float, default=0.0294
        Annual cost of funds.
    loss_given_default : float, default=0.75
        Fraction of the credit line lost upon default (:math:`L_{gd}`).
    term_length_months : int, default=24
        Loan term in months.

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'cl': credit_line, 'fp_cost': fp_cost_array}``
    """
    int_r = interest_rate / 12
    int_cf = fund_cost / 12
    pi_1 = float(target_np.mean())
    cl_avg = float(credit_line.mean())

    cl_arr = np.asarray(credit_line, dtype=float)
    fp_costs = np.vectorize(_cost_fp_calc)(cl_arr, int_r, term_length_months, int_cf, pi_1, loss_given_default, cl_avg)

    cl_sym, lgd_sym, fp_sym = sp.symbols('cl lgd fp_cost')
    cost_matrix = (
        CostMatrix()
        .add_fn_cost(cl_sym * lgd_sym)
        .add_fp_cost(fp_sym)
        .alias({'loss_given_default': 'lgd'})
        .set_default(loss_given_default=loss_given_default)
    )
    return cost_matrix, {'cl': cl_arr, 'fp_cost': fp_costs}


def fraud_detection_cost_matrix(
    amount: FloatNDArray,
    *,
    investigation_cost: float = 10.0,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """
    Fraud detection cost matrix (Vanderschueren et al. 2022, Höppner et al. 2022).

    Parameters
    ----------
    amount : array-like
        Per-transaction amount (:math:`A_i`), representing the cost of a false negative.
    investigation_cost : float, default=10.0
        Fixed cost of investigating an alert (:math:`c_f`), incurred on both
        true positives and false positives.

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'amount': amount}``
    """
    amount_sym, cf_sym = sp.symbols('amount cf')
    cost_matrix = (
        CostMatrix()
        .add_fn_cost(amount_sym)
        .add_fp_cost(cf_sym)
        .add_tp_cost(cf_sym)
        .alias({'investigation_cost': 'cf'})
        .set_default(investigation_cost=investigation_cost)
    )
    return cost_matrix, {'amount': np.asarray(amount, dtype=float)}


def direct_marketing_cost_matrix(
    amount: FloatNDArray,
    *,
    contact_cost: float = 0.68,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """
    Direct marketing cost matrix (Vanderschueren et al. 2022, Zadrozny & Elkan 2001).

    Parameters
    ----------
    amount : array-like
        Per-instance donation / return amount (:math:`A_i`), representing the cost
        of a false negative.
    contact_cost : float, default=0.68
        Fixed mailing / contact cost (:math:`c_f`), incurred on both true positives
        and false positives.

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'amount': amount}``
    """
    amount_sym, c_sym = sp.symbols('amount c')
    cost_matrix = (
        CostMatrix()
        .add_fn_cost(amount_sym)
        .add_fp_cost(c_sym)
        .add_tp_cost(c_sym)
        .alias({'contact_cost': 'c'})
        .set_default(contact_cost=contact_cost)
    )
    return cost_matrix, {'amount': np.asarray(amount, dtype=float)}


def churn_monthly_charges_cost_matrix(
    monthly_charges: FloatNDArray,
    *,
    fn_months: float = 12.0,
    fp_months: float = 2.0,
) -> tuple[CostMatrix, dict[str, FloatNDArray]]:
    """
    Churn prediction cost matrix based on monthly charges (Vanderschueren et al. 2022).

    Parameters
    ----------
    monthly_charges : array-like
        Per-customer monthly charge (:math:`A_i`).
    fn_months : float, default=12.0
        Multiplier for false negative cost (e.g. 12 months of lost revenue).
    fp_months : float, default=2.0
        Multiplier for false positive cost (e.g. 2 months of incentive / contact cost).

    Returns
    -------
    cost_matrix : CostMatrix
    instance_costs : dict
        ``{'monthly_charges': monthly_charges}``
    """
    mc_sym, fn_m_sym, fp_m_sym = sp.symbols('monthly_charges fn_months fp_months')
    cost_matrix = (
        CostMatrix()
        .add_fn_cost(fn_m_sym * mc_sym)
        .add_fp_cost(fp_m_sym * mc_sym)
        .set_default(fn_months=fn_months, fp_months=fp_months)
    )
    return cost_matrix, {'monthly_charges': np.asarray(monthly_charges, dtype=float)}


def _a_calc(cl_i: float, int_r: float, n: int) -> float:
    return cl_i * (int_r * (1 + int_r) ** n) / ((1 + int_r) ** n - 1)  # type: ignore[return-value]


def _pv_calc(a: float, int_r: float, n: int) -> float:
    return a / int_r * (1 - 1 / (1 + int_r) ** n)  # type: ignore[return-value]


def _cost_fn_calc(cl_i: float, lgd: float) -> float:
    return cl_i * lgd  # type: ignore[return-value]


def _cost_fp_calc(
    cl_i: float,
    int_r: float,
    n: int,
    int_cf: float,
    pi_1_: float,
    lgd: float,
    cl_avg: float,
) -> float:
    a = _a_calc(cl_i, int_r, n)
    pv = _pv_calc(a, int_cf, n)
    r = pv - cl_i
    r_avg = _pv_calc(_a_calc(cl_avg, int_r, n), int_cf, n) - cl_avg
    return max(0.0, r - (1 - pi_1_) * r_avg + pi_1_ * _cost_fn_calc(cl_avg, lgd))  # type: ignore[return-value]


def _creditscoring_costmat(
    income: FloatNDArray,
    debt: FloatNDArray,
    pi_1: float,
    params: dict[str, Any],
) -> FloatNDArray:
    """Compute the per-instance cost matrix for credit scoring."""

    def _cl(k: float, inc: float, cl_max: float, debt_i: float, int_r: float, n: int) -> float:
        cl_k = k * inc
        a = _a_calc(cl_k, int_r, n)
        cl_d = _pv_calc(inc * min(a / inc, 1 - debt_i), int_r, n)
        return min(cl_k, cl_max, cl_d)  # type: ignore[return-value]

    k = params['k']
    int_r = params['int_r']
    n_term = params['n_term']
    int_cf = params['int_cf']
    lgd = params['lgd']
    cl_max = params['cl_max']

    cl = np.vectorize(_cl)(k, income, cl_max, debt, int_r, n_term)
    cl_avg = float(cl.mean())

    n = income.shape[0]
    mat = np.zeros((n, 4))
    mat[:, 0] = np.vectorize(_cost_fp_calc)(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
    mat[:, 1] = np.vectorize(_cost_fn_calc)(cl, lgd)
    return mat  # type: ignore[return-value]


def _compute_credit_lines(
    income: FloatNDArray,
    debt: FloatNDArray,
    params: dict[str, Any],
) -> FloatNDArray:
    """Return per-instance estimated credit lines."""
    k = params['k']
    int_r = params['int_r']
    n_term = params['n_term']
    cl_max = params['cl_max']

    def _a(cl_i: float, int_r_: float, n: int) -> float:
        return cl_i * (int_r_ * (1 + int_r_) ** n) / ((1 + int_r_) ** n - 1)  # type: ignore[return-value]

    def _pv(a: float, int_r_: float, n: int) -> float:
        return a / int_r_ * (1 - 1 / (1 + int_r_) ** n)  # type: ignore[return-value]

    def _cl(k_: float, inc: float, cl_max_: float, debt_i: float, int_r_: float, n: int) -> float:
        cl_k = k_ * inc
        a = _a(cl_k, int_r_, n)
        cl_d = _pv(inc * min(a / inc, 1 - debt_i), int_r_, n)
        return min(cl_k, cl_max_, cl_d)  # type: ignore[return-value]

    return np.vectorize(_cl)(k, income, cl_max, debt, int_r, n_term)  # type: ignore[return-value, no-any-return]
