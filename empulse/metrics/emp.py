from itertools import product
from typing import Callable, Generator, Iterable, Union

import numpy as np
from numpy.typing import ArrayLike

from ._convex_hull import _compute_convex_hull
from .common import _range


def emp_score(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        weighted_pdf: Callable[[float, float, float, float, float, float, float, float], float],
        tp_benefit: Union[float, tuple[float, float]] = 0.0,
        tn_benefit: Union[float, tuple[float, float]] = 0.0,
        fn_cost: Union[float, tuple[float, float]] = 0.0,
        fp_cost: Union[float, tuple[float, float]] = 0.0,
        n_buckets: int = 100,
) -> float:
    """
    :func:`~empulse.metrics.emp()` but only returning the EMP score.

    .. seealso::
        :func:`~empulse.metrics.emp` : To also return the threshold at which the EMP score is achieved.

        :func:`~empulse.metrics.max_profit_score` : For a deterministic version of the EMP score.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    weighted_pdf : Callable[[float, float, float, float, float, float, float, float], float]
        Weighted probability density function (PDF) of the joint distribution of the benefits and costs.
        The weighted PDF is defined as the product of the PDF and the step size of the benefits and costs.
        Function takes in the following arguments:

        - `b0` (``float``): benefit for true positive prediction.

        - `b1` (``float``): benefit for true negative prediction.

        - `c0` (``float``): cost for false negative prediction.

        - `c1` (``float``): cost for false positive prediction.

        - `step_size_b0` (``float``): step size of the benefit for true positive prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if `tp_benefit` is a scalar.

        - `step_size_b1` (``float``): step size of the benefit for true negative prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if `tn_benefit` is a scalar.

        - `step_size_c0` (``float``): step size of the cost for false negative prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if `fn_cost` is a scalar.

        - `step_size_c1` (``float``): step size of the cost for false positive prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if fp_cost is a scalar.

    tp_benefit : float or tuple[float, float], default=0.0
        Benefit attributed to true positive predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    tn_benefit : float or tuple[float, float], default=0.0
        Benefit attributed to true negative predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    fn_cost : float or tuple[float, float], default=0.0
        Cost attributed to false negative predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    fp_cost : float or tuple[float, float], default=0.0
        Cost attributed to false positive predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    n_buckets : int, default=100
        Number of buckets to use for the numerical integration per parameter.
        Note that the computational complexity of the algorithm is exponential in the number of buckets.
        For instance if you have 4 parameters with 100 buckets each, the algorithm will have to compute
        the weighted PDF for 100^4 = 100_000_000 combinations.

    Returns
    -------
    emp : float
        Expected Maximum Profit.

    Notes
    -----
    The EMP is defined as [1]_:

    .. math::  \\int_{b_0} \\int_{c_0} \\int_{b_1} \\int_{c_1} P(T;b_0, c_0, b_1, c_1) \\cdot w(b_0, c_0, b_1, c_1) \\, db_0 dc_0 db_1 dc_1

    The EMP requires that the positive class is encoded as 0, and negative class as 1.
    However, this implementation assumes the standard notation ('positive': 1, 'negative': 0).

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289

    Examples
    --------

    Reimplement EMPC:

    >>> from empulse.metrics import emp_score
    >>> from scipy.stats import beta
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>>
    >>> clv = 200
    >>> d = 10
    >>> f = 1
    >>> tp_benefit = (-f, clv * (1 - (d / clv) - (f / clv)))
    >>> fp_cost = d + f
    >>> def weighted_pdf(b0, b1, c0, c1, b0_step, b1_step, c0_step, c1_step):
    ...     gamma = (b0 + f) / (clv - d)
    ...     gamma_step = b0_step / (clv - d)
    ...     return beta.pdf(gamma, a=6, b=14) * gamma_step
    >>>
    >>> emp_score(
    ...     y_true,
    ...     y_score,
    ...     weighted_pdf=weighted_pdf,
    ...     tp_benefit=tp_benefit,
    ...     fp_cost=fp_cost,
    ...     n_buckets=1000
    ... )
    23.875...
    """
    return emp(
        y_true,
        y_score,
        weighted_pdf=weighted_pdf,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        n_buckets=n_buckets
    )[0]


def emp(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        weighted_pdf: Callable[[float, float, float, float, float, float, float, float], float],
        tp_benefit: Union[float, tuple[float, float]] = 0.0,
        tn_benefit: Union[float, tuple[float, float]] = 0.0,
        fn_cost: Union[float, tuple[float, float]] = 0.0,
        fp_cost: Union[float, tuple[float, float]] = 0.0,
        n_buckets: int = 100,
) -> tuple[float, float]:
    """
    Expected Maximum Profit Measure (EMP).

    .. seealso::
        :func:`~empulse.metrics.emp_score` : To only return the EMP score.

        :func:`~empulse.metrics.max_profit` : For a deterministic version of the EMP.

    Parameters
    ----------
    y_true : 1D array-like, shape=(n_samples,)
        Binary target values ('positive': 1, 'negative': 0).

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or non-thresholded decision values.

    weighted_pdf : Callable[[float, float, float, float, float, float, float, float], float]
        Weighted probability density function (PDF) of the joint distribution of the benefits and costs.
        The weighted PDF is defined as the product of the PDF and the step size of the benefits and costs.
        Function takes in the following arguments:

        - `b0` (``float``): benefit for true positive prediction.

        - `b1` (``float``): benefit for true negative prediction.

        - `c0` (``float``): cost for false negative prediction.

        - `c1` (``float``): cost for false positive prediction.

        - `step_size_b0` (``float``): step size of the benefit for true positive prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if `tp_benefit` is a scalar.

        - `step_size_b1` (``float``): step size of the benefit for true negative prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if `tn_benefit` is a scalar.

        - `step_size_c0` (``float``): step size of the cost for false negative prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if `fn_cost` is a scalar.

        - `step_size_c1` (``float``): step size of the cost for false positive prediction for numerical integration.
          Step size is defined as ``(max_val - min_val) / n_buckets``.
          Step size is 0 if fp_cost is a scalar.

    tp_benefit : float or tuple[float, float], default=0.0
        Benefit attributed to true positive predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    tn_benefit : float or tuple[float, float], default=0.0
        Benefit attributed to true negative predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    fn_cost : float or tuple[float, float], default=0.0
        Cost attributed to false negative predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    fp_cost : float or tuple[float, float], default=0.0
        Cost attributed to false positive predictions.
        If ``float``, deterministic parameter defined by a scalar.
        If ``tuple``, stochastic parameter defined by an upper and lower bound.

    n_buckets : int, default=100
        Number of buckets to use for the numerical integration per parameter.
        Note that the computational complexity of the algorithm is exponential in the number of buckets.
        For instance if you have 4 parameters with 100 buckets each, the algorithm will have to compute
        the weighted PDF for 100^4 = 100.000.000 combinations.

    Returns
    -------
    emp : float
        Expected Maximum Profit.

    threshold : float
        Threshold at which the expected maximum profit is achieved.

    Notes
    -----
    The EMP is defined as [1]_:

    .. math::  \\int_{b_0} \\int_{c_0} \\int_{b_1} \\int_{c_1} P(T;b_0, c_0, b_1, c_1) \\cdot w(b_0, c_0, b_1, c_1) \\, db_0 dc_0 db_1 dc_1

    The EMP requires that the positive class is encoded as 0, and negative class as 1.
    However, this implementation assumes the standard notation ('positive': 1, 'negative': 0).

    References
    ----------
    .. [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289

    Examples
    --------

    Reimplement EMPC:

    >>> from empulse.metrics import emp
    >>> from scipy.stats import beta
    >>>
    >>> y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    >>> y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    >>>
    >>> clv = 200
    >>> d = 10
    >>> f = 1
    >>> tp_benefit = (-f, clv * (1 - (d / clv) - (f / clv)))
    >>> fp_cost = d + f
    >>> def weighted_pdf(b0, b1, c0, c1, b0_step, b1_step, c0_step, c1_step):
    ...     gamma = (b0 + f) / (clv - d)
    ...     gamma_step = b0_step / (clv - d)
    ...     return beta.pdf(gamma, a=6, b=14) * gamma_step
    >>>
    >>> emp(
    ...     y_true,
    ...     y_score,
    ...     weighted_pdf=weighted_pdf,
    ...     tp_benefit=tp_benefit,
    ...     fp_cost=fp_cost,
    ...     n_buckets=1000
    ... )
    (23.875..., 0.874...)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pi0 = float(np.mean(y_true))
    pi1 = 1 - pi0

    f0, f1 = _compute_convex_hull(y_true, y_score)

    bounds = [var if isinstance(var, tuple) else (var, var) for var in (tp_benefit, tn_benefit, fn_cost, fp_cost)]
    return _compute_emp(pi0, pi1, f0, f1, bounds, weighted_pdf, n_buckets)


def _compute_emp(
        pi0: float,
        pi1: float,
        f0: np.ndarray,
        f1: np.ndarray,
        bounds: list[tuple[float, float]],
        weighted_pdf: Callable[[float, float, float, float, float, float, float, float], float],
        n_buckets: int
) -> tuple[float, float]:
    """Computes the expected maximum profit and the threshold at which the maximum profit is achieved."""
    emp = 0.0
    eta = 0.0
    step_sizes = [(max_val - min_val) / n_buckets if min_val != max_val else 0 for min_val, max_val in bounds]
    for b0, b1, c0, c1 in _construct_iter(bounds, step_sizes):
        weight = weighted_pdf(b0, b1, c0, c1, *step_sizes)
        expected_profits = ((b0 + c0) * pi0 * f0 - (b1 + c1) * pi1 * f1 + b1 * pi1 - c0 * pi0) * weight
        best_index = np.argmax(expected_profits)
        emp += expected_profits[best_index]
        eta += (pi0 * f0[best_index] + pi1 * f1[best_index]) * weight
    return emp, eta


def _construct_iter(
        bounds: Iterable[tuple[float, float]],
        steps_sizes: Iterable[float],
) -> Generator[tuple[float, ...], None, None]:
    """Construct an iterator over the parameter space."""
    return product(*(
        _range(min_val, max_val, step_size) if min_val != max_val
        else (min_val,)
        for (min_val, max_val), step_size in zip(bounds, steps_sizes)
    ))
