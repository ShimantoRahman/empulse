"""
The sympy.stats distributions with a closed-form piecewise treatment, as scipy equivalents.

``piecewise.py`` used to carry two parallel 11-branch ``isinstance(distribution, sympy.stats.
crv_types.XDistribution)`` chains -- one in ``_build_max_profit_score_piecewise``, one in
``_build_max_profit_rate_piecewise`` -- agreeing on which distributions get a closed-form
treatment and differing only in which ``(scipy_dist, param_adapter)`` pair each branch used for
the *rate* side (the score side additionally picks a distribution-specific score class; that
mapping stays in ``piecewise.py`` itself, since the classes it names are defined there).
:data:`ADAPTERS` is the ``(scipy_dist, param_adapter)`` half of that shared information as one
table, so adding a distribution to the rate side is one row here instead of a matching edit in
each chain.

``quasi_monte_carlo.py`` separately maintains its own, larger ``_sympy_dist_to_scipy``/
``_sympy_dist_to_scipy_params`` tables, covering every quasi-Monte-Carlo-sampleable distribution
(~30, most with no closed-form piecewise treatment at all). Its entries for the ten distributions
this module also covers happen to compute the same adapters (e.g. both give ``GammaDistribution``
``{'a': k, 'scale': theta}``) -- a real, if smaller, duplication left alone here rather than
merged into a table shaped for a different purpose and a different caller.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import scipy.stats as st
import sympy.stats.crv_types as crv
from scipy.stats import rv_continuous
from sympy.stats.rv import RandomSymbol, pspace


def _uniform_params(a: float, b: float) -> dict[str, float]:
    return {'loc': a, 'scale': b - a}


def _beta_params(a: float, b: float) -> dict[str, float]:
    return {'a': a, 'b': b}


def _normal_params(mu: float, sigma: float) -> dict[str, float]:
    return {'loc': mu, 'scale': sigma}


def _lognormal_params(mu: float, sigma: float) -> dict[str, float]:
    return {'s': sigma, 'scale': float(np.exp(mu))}


def _gamma_params(k: float, theta: float) -> dict[str, float]:
    return {'a': k, 'scale': theta}


def _expon_params(rate: float) -> dict[str, float]:
    return {'loc': 0.0, 'scale': 1.0 / rate}


def _chi2_params(df: float) -> dict[str, float]:
    return {'df': df}


def _weibull_params(a: float, b: float) -> dict[str, float]:
    return {'c': b, 'scale': a}


def _pareto_params(a: float, b: float) -> dict[str, float]:
    return {'b': b, 'scale': a}


def _triangular_params(a: float, b: float, c: float) -> dict[str, float]:
    return {'loc': a, 'scale': b - a, 'c': (c - a) / (b - a)}


class DistributionAdapter(NamedTuple):
    """What it takes to treat one ``sympy.stats`` distribution as a scipy one, exactly."""

    #: The scipy continuous distribution with the same shape, used by the rate's exact CDF.
    scipy_dist: rv_continuous
    #: Maps the sympy distribution's own parameter values to *scipy_dist*'s keyword arguments.
    params: Callable[..., dict[str, float]]


#: One row per sympy.stats distribution with a closed-form piecewise treatment, keyed by its
#: sympy.stats distribution class. `adapter_for` looks this up by `type(distribution)` first,
#: falling back to an `isinstance` scan for a subclass sympy.stats doesn't register verbatim here.
ADAPTERS: dict[type, DistributionAdapter] = {
    crv.UniformDistribution: DistributionAdapter(st.uniform, _uniform_params),
    crv.BetaDistribution: DistributionAdapter(st.beta, _beta_params),
    crv.NormalDistribution: DistributionAdapter(st.norm, _normal_params),
    crv.LogNormalDistribution: DistributionAdapter(st.lognorm, _lognormal_params),
    crv.GammaDistribution: DistributionAdapter(st.gamma, _gamma_params),
    crv.ExponentialDistribution: DistributionAdapter(st.expon, _expon_params),
    crv.ChiSquaredDistribution: DistributionAdapter(st.chi2, _chi2_params),
    crv.WeibullDistribution: DistributionAdapter(st.weibull_min, _weibull_params),
    crv.ParetoDistribution: DistributionAdapter(st.pareto, _pareto_params),
    crv.TriangularDistribution: DistributionAdapter(st.triang, _triangular_params),
}


def adapter_for(random_symbol: RandomSymbol) -> DistributionAdapter | None:
    """Return the :class:`DistributionAdapter` for *random_symbol*'s distribution, if any.

    Looks up by exact type first; falls back to an ``isinstance`` scan so a subclass of a
    registered distribution (which :data:`ADAPTERS`'s exact-type lookup would otherwise miss,
    unlike the ``isinstance`` chains this table replaces) still matches. Returns ``None`` for a
    distribution with no closed-form piecewise treatment.
    """
    distribution = pspace(random_symbol).distribution
    adapter = ADAPTERS.get(type(distribution))
    if adapter is not None:
        return adapter
    for distribution_type, candidate in ADAPTERS.items():
        if isinstance(distribution, distribution_type):
            return candidate
    return None
