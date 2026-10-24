"""
What empulse knows about each ``sympy.stats`` distribution, in one table.

:data:`DISTRIBUTIONS` maps a distribution class to its :class:`DistributionSpec`: the equivalent
SciPy distribution and how to map the parameters to it (for sampling, quantiles, CDFs and control
variates), the closed form of its partial moments in the compiled piecewise score, if any, and its
mean where ``sympy.stats.E`` cannot compute it. Supporting a distribution in a new way takes a
field of its row; supporting a new distribution takes a row. :func:`spec_for` is the one lookup.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import scipy.stats as st
import sympy
import sympy.stats.crv_types as crv
from sympy.stats.rv import pspace

#: A frozen SciPy distribution.
FrozenDistribution = Any


class DistributionSpec(NamedTuple):
    """What it takes to treat one ``sympy.stats`` distribution as a SciPy one, and more."""

    #: The SciPy distribution of the same shape.
    scipy: st.rv_continuous | st.rv_discrete
    #: Maps the sympy distribution's own parameter values, in the order its constructor takes them,
    #: to the keyword arguments of :attr:`scipy`; ``None`` passes them on positionally.
    scipy_parameters: Callable[..., dict[str, float]] | None = None
    #: The name of the distribution in :class:`~empulse.metrics._cy_max_profit.Distribution`, whose
    #: partial moments the compiled piecewise score computes in closed form.
    closed_form: str | None = None
    #: The mean, from the distribution's parameters, where ``sympy.stats.E`` cannot compute it.
    mean: Callable[[tuple[sympy.Expr, ...]], sympy.Expr] | None = None

    def freeze(self, values: tuple[float, ...] | list[float]) -> FrozenDistribution:
        """Return :attr:`scipy` frozen at the sympy distribution's parameter *values*."""
        if self.scipy_parameters is None:
            return self.scipy(*values)
        return self.scipy(**self.scipy_parameters(*values))

    def __reduce__(self) -> str | tuple[Any, ...]:
        # A row of the table holds lambdas, which cannot be pickled, so it is pickled as its key.
        for distribution_type, spec in DISTRIBUTIONS.items():
            if spec is self:
                return _spec_of, (distribution_type,)
        return type(self), tuple(self)


DISTRIBUTIONS: dict[type, DistributionSpec] = {
    # Arcsin and PowerFunction are bounded on [a, b], while SciPy's `scale` is the *width* of the
    # support, so passing b straight through would stretch it to [a, a + b].
    crv.ArcsinDistribution: DistributionSpec(
        st.arcsine, lambda a, b: {'loc': a, 'scale': b - a}, mean=lambda params: (params[0] + params[1]) / 2
    ),
    crv.BetaDistribution: DistributionSpec(st.beta, lambda a, b: {'a': a, 'b': b}, closed_form='BETA'),
    crv.BetaPrimeDistribution: DistributionSpec(st.betaprime, mean=lambda params: params[0] / (params[1] - 1)),
    crv.BoundedParetoDistribution: DistributionSpec(
        st.truncpareto, lambda alpha, left, right: {'b': alpha, 'c': right / left, 'scale': left}
    ),
    crv.ChiDistribution: DistributionSpec(st.chi),
    crv.ChiSquaredDistribution: DistributionSpec(st.chi2, lambda df: {'df': df}, closed_form='CHI_SQUARED'),
    crv.DagumDistribution: DistributionSpec(st.mielke, lambda p, a, b: {'k': a * p, 's': a, 'scale': b}),
    # Erlang is built on GammaDistribution, so it needs no entry of its own.
    crv.ExGaussianDistribution: DistributionSpec(
        st.exponnorm, lambda mean, std, rate: {'loc': mean, 'scale': std, 'K': 1 / (std * rate)}
    ),
    crv.ExponentialDistribution: DistributionSpec(
        st.expon, lambda rate: {'loc': 0.0, 'scale': 1.0 / rate}, closed_form='EXPONENTIAL'
    ),
    crv.ExponentialPowerDistribution: DistributionSpec(
        st.gennorm, lambda mu, alpha, beta: {'beta': beta, 'loc': mu, 'scale': alpha}
    ),
    crv.FDistributionDistribution: DistributionSpec(
        st.f, lambda d1, d2: {'dfn': d1, 'dfd': d2}, mean=lambda params: params[1] / (params[1] - 2)
    ),
    crv.FrechetDistribution: DistributionSpec(st.invweibull, lambda a, s, m: {'c': a, 'loc': m, 'scale': s}),
    crv.GammaDistribution: DistributionSpec(st.gamma, lambda k, theta: {'a': k, 'scale': theta}, closed_form='GAMMA'),
    crv.GammaInverseDistribution: DistributionSpec(
        st.invgamma, lambda a, b: {'a': a, 'scale': b}, mean=lambda params: params[1] / (params[0] - 1)
    ),
    crv.GaussianInverseDistribution: DistributionSpec(st.invgauss, lambda mu, lamb: {'mu': mu / lamb, 'scale': lamb}),
    crv.GompertzDistribution: DistributionSpec(st.gompertz, lambda b, eta: {'c': eta, 'scale': 1 / b}),
    crv.LaplaceDistribution: DistributionSpec(st.laplace),
    crv.LogisticDistribution: DistributionSpec(st.logistic),
    crv.LogLogisticDistribution: DistributionSpec(st.fisk, lambda alpha, beta: {'c': beta, 'scale': alpha}),
    crv.LogNormalDistribution: DistributionSpec(
        st.lognorm,
        lambda mu, sigma: {'s': sigma, 'scale': float(np.exp(mu))},
        closed_form='LOG_NORMAL',
        mean=lambda params: sympy.exp(params[0] + params[1] ** 2 / 2),
    ),
    crv.LomaxDistribution: DistributionSpec(
        st.lomax, lambda alpha, lamb: {'c': alpha, 'scale': lamb}, mean=lambda params: params[1] / (params[0] - 1)
    ),
    crv.MaxwellDistribution: DistributionSpec(st.maxwell, lambda a: {'loc': 0, 'scale': a}),
    crv.MoyalDistribution: DistributionSpec(st.moyal),
    crv.NakagamiDistribution: DistributionSpec(st.nakagami, lambda mu, omega: {'nu': mu, 'scale': np.sqrt(omega)}),
    crv.NormalDistribution: DistributionSpec(
        st.norm, lambda mu, sigma: {'loc': mu, 'scale': sigma}, closed_form='NORMAL'
    ),
    crv.ParetoDistribution: DistributionSpec(
        st.pareto,
        lambda a, b: {'b': b, 'scale': a},
        closed_form='PARETO',
        mean=lambda params: (params[1] * params[0]) / (params[1] - 1),
    ),
    crv.PowerFunctionDistribution: DistributionSpec(
        st.powerlaw,
        lambda alpha, a, b: {'a': alpha, 'loc': a, 'scale': b - a},
        mean=lambda params: params[1] + params[0] * (params[2] - params[1]) / (params[0] + 1),
    ),
    crv.RaisedCosineDistribution: DistributionSpec(st.cosine, lambda mu, s: {'loc': mu, 'scale': s / np.pi}),
    crv.RayleighDistribution: DistributionSpec(st.rayleigh, lambda sigma: {'loc': 0, 'scale': sigma}),
    crv.ReciprocalDistribution: DistributionSpec(st.loguniform, lambda a, b: {'a': a, 'b': b}),
    crv.StudentTDistribution: DistributionSpec(st.t, mean=lambda params: sympy.S.Zero),
    crv.TrapezoidalDistribution: DistributionSpec(
        st.trapezoid,
        lambda a, b, c, d: {'loc': a, 'scale': d - a, 'c': (b - a) / (d - a), 'd': (c - a) / (d - a)},
    ),
    crv.TriangularDistribution: DistributionSpec(
        st.triang, lambda a, b, c: {'loc': a, 'scale': b - a, 'c': (c - a) / (b - a)}, closed_form='TRIANGULAR'
    ),
    crv.UniformDistribution: DistributionSpec(
        st.uniform, lambda a, b: {'loc': a, 'scale': b - a}, closed_form='UNIFORM'
    ),
    crv.WeibullDistribution: DistributionSpec(
        st.weibull_min, lambda alpha, beta: {'c': beta, 'scale': alpha}, closed_form='WEIBULL'
    ),
    crv.WignerSemicircleDistribution: DistributionSpec(st.semicircular, lambda radius: {'loc': 0, 'scale': radius}),
}


def _spec_of(distribution_type: type) -> DistributionSpec:
    """Return the row of :data:`DISTRIBUTIONS` for *distribution_type*, which unpickles a row."""
    return DISTRIBUTIONS[distribution_type]


def spec_for(random_symbol: Any) -> DistributionSpec | None:
    """
    Return the :class:`DistributionSpec` of *random_symbol*'s distribution, or ``None`` without one.

    Looks the distribution up by its exact type, then by ``isinstance``, so that a subclass of a
    supported distribution is supported too.
    """
    distribution = pspace(random_symbol).distribution
    spec = DISTRIBUTIONS.get(type(distribution))
    if spec is not None:
        return spec
    for distribution_type, candidate in DISTRIBUTIONS.items():
        if isinstance(distribution, distribution_type):
            return candidate
    return None


def frozen_distribution(random_symbol: Any) -> FrozenDistribution:
    """Return the SciPy distribution of *random_symbol*, whose parameters must all be numbers."""
    spec = spec_for(random_symbol)
    if spec is None:
        raise ValueError(f'{type(pspace(random_symbol).distribution).__name__} has no SciPy counterpart.')
    return spec.freeze([float(argument) for argument in pspace(random_symbol).distribution.args])
