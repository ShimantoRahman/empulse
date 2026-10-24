"""
Reference partial moments, written with SciPy, for the compiled ones the piecewise score uses.

These are the per-distribution Python integrations the piecewise score classes carried before the
compiled :func:`~empulse.metrics._cy_max_profit.partial_moments` became the one implementation.
They are kept here only as the oracle that implementation is tested against.
"""

import numpy as np
import scipy.special as sp
import scipy.stats as st


def _positive(cdf_and_moment, bounds, n_powers):
    rows = []
    for k in range(n_powers):
        shifted_cdf, moment = cdf_and_moment(k)
        rows.append(moment * np.diff(shifted_cdf(bounds)))
    return np.array(rows)


def _uniform(values, bounds, n_powers):
    lower, upper = bounds[:-1], bounds[1:]
    pdf = 1.0 / (bounds[-1] - bounds[0])
    return np.array([pdf / (k + 1.0) * (upper ** (k + 1) - lower ** (k + 1)) for k in range(n_powers)])


def _normal(values, bounds, n_powers):
    mu, sigma = values
    lower, upper = bounds[:-1], bounds[1:]
    z_lower, z_upper = (lower - mu) / sigma, (upper - mu) / sigma
    phi_lower, phi_upper = st.norm.pdf(z_lower), st.norm.pdf(z_upper)

    def pow_phi(x, power, phi):
        with np.errstate(invalid='ignore'):
            result = x**power * phi
        result[phi == 0] = 0.0
        return result

    r = [st.norm.cdf(z_upper) - st.norm.cdf(z_lower)]
    if n_powers > 1:
        r.append(mu * r[0] - sigma * (phi_upper - phi_lower))
    for k in range(2, n_powers):
        r.append(
            mu * r[-1]
            + (k - 1.0) * sigma**2 * r[-2]
            - sigma * (pow_phi(upper, k - 1, phi_upper) - pow_phi(lower, k - 1, phi_lower))
        )
    return np.array(r)


def _triangular(values, bounds, n_powers):
    a, b, m = values
    x = np.clip(bounds, a, b)
    rows = [np.diff(st.triang.cdf(x, c=(m - a) / (b - a), loc=a, scale=b - a))]
    for k in range(1, n_powers):
        rising = (2.0 / ((b - a) * (m - a))) if m > a else 0.0
        at_mode = rising * (
            m ** (k + 2) / (k + 2.0) - a * m ** (k + 1) / (k + 1.0) + a ** (k + 2) / ((k + 1.0) * (k + 2.0))
        )
        below = rising * (
            x ** (k + 2) / (k + 2.0) - a * x ** (k + 1) / (k + 1.0) + a ** (k + 2) / ((k + 1.0) * (k + 2.0))
        )
        above = at_mode + (2.0 / ((b - a) * (b - m))) * (
            b * (x ** (k + 1) - m ** (k + 1)) / (k + 1.0) - (x ** (k + 2) - m ** (k + 2)) / (k + 2.0)
        )
        rows.append(np.diff(np.where(x <= m, below, above)))
    return np.array(rows)


def _gamma(values, bounds, n_powers):
    shape, scale = values
    return _positive(
        lambda k: (
            lambda x: st.gamma.cdf(x, a=shape + k, scale=scale),
            1.0 if k == 0 else scale**k * sp.poch(shape, k),
        ),
        bounds,
        n_powers,
    )


def _pareto(values, bounds, n_powers):
    scale, shape = values
    return _positive(
        lambda k: (
            lambda x: st.pareto.cdf(x, b=shape - k, scale=scale),
            1.0 if k == 0 else shape * scale**k / (shape - k),
        ),
        bounds,
        n_powers,
    )


def _exponential(values, bounds, n_powers):
    scale = 1.0 / values[0]
    return _positive(
        lambda k: (
            (lambda x: st.expon.cdf(x, scale=scale)) if k == 0 else (lambda x: st.gamma.cdf(x, a=1.0 + k, scale=scale)),
            1.0 if k == 0 else sp.gamma(1.0 + k) * scale**k,
        ),
        bounds,
        n_powers,
    )


def _chi_squared(values, bounds, n_powers):
    (df,) = values
    return _positive(
        lambda k: (lambda x: st.chi2.cdf(x, df=df + 2.0 * k), 1.0 if k == 0 else 2.0**k * sp.poch(df / 2.0, k)),
        bounds,
        n_powers,
    )


def _log_normal(values, bounds, n_powers):
    mu, sigma = values
    return _positive(
        lambda k: (
            lambda x: st.lognorm.cdf(x, s=sigma, scale=np.exp(mu + k * sigma**2)),
            1.0 if k == 0 else np.exp(k * mu + k**2 * sigma**2 / 2.0),
        ),
        bounds,
        n_powers,
    )


def _beta(values, bounds, n_powers):
    alpha, beta = values
    return _positive(
        lambda k: (
            lambda x: sp.betainc(alpha + k, beta, x),
            1.0 if k == 0 else sp.poch(alpha, k) / sp.poch(alpha + beta, k),
        ),
        bounds,
        n_powers,
    )


def _weibull(values, bounds, n_powers):
    scale, shape = values
    return _positive(
        lambda k: (
            (lambda x: st.weibull_min.cdf(x, c=shape, scale=scale))
            if k == 0
            else (lambda x: st.gamma.cdf((x / scale) ** shape, a=1.0 + k / shape)),
            1.0 if k == 0 else scale**k * sp.gamma(1.0 + k / shape),
        ),
        bounds,
        n_powers,
    )


ORACLES = {
    'UNIFORM': _uniform,
    'NORMAL': _normal,
    'TRIANGULAR': _triangular,
    'GAMMA': _gamma,
    'PARETO': _pareto,
    'EXPONENTIAL': _exponential,
    'CHI_SQUARED': _chi_squared,
    'LOG_NORMAL': _log_normal,
    'BETA': _beta,
    'WEIBULL': _weibull,
}


def reference_partial_moments(closed_form, values, bounds, n_powers):
    """The integral of ``x**k`` times the density over each region between *bounds*, at ``[k, region]``."""
    return ORACLES[closed_form](values, np.asarray(bounds, dtype=np.float64), n_powers)
