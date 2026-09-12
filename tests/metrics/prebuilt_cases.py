"""
The table of prebuilt metrics under test, and how each maps onto its reference implementation.

``tests/metrics/reference/`` holds the pre-refactor NumPy implementations, kept purely as ground
truth. Every prebuilt metric in ``empulse.metrics`` is checked against its entry here by
``test_prebuilt_metric_contract.py``.

Two behaviour differences between the package and the reference are intentional and are encoded in
the ``to_reference_params`` / ``reference_returns_rate`` fields rather than papered over:

* ``expected_cost_loss_churn`` / ``expected_cost_loss_acquisition`` are built on the ``Cost``
  strategy, which always returns the *mean* cost across samples. The reference functions default to
  the *sum*, so their calls pass ``normalize=True``.
* ``empa_score``'s ``beta`` is the *scale* of the Gamma-distributed contribution
  (mean = ``alpha * beta``), not the *rate* (mean = ``alpha / beta``) the reference ``empa`` uses.
  This follows from how ``MaxProfit``'s stochastic-integration engine parameterizes distributions:
  it requires distribution arguments to be plain symbols, not derived expressions such as
  ``1 / beta``. The conversion is applied when calling the reference.

The tuple-returning reference functions return ``(score, predicted_positive_rate)``. The rate --
not the threshold -- is what ``Metric.optimal_rate`` reproduces; ``optimal_threshold`` is a
different quantity and is checked separately.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from empulse.metrics import (
    auepc_score,
    empa_score,
    empb_score,
    empc_score,
    empcs_score,
    mpa_score,
    mpc_score,
    mpcs_score,
)
from empulse.metrics.acquisition.cost import expected_cost_loss_acquisition
from empulse.metrics.churn.cost import expected_cost_loss_churn

from .reference import acquisition as ref_acquisition
from .reference import churn as ref_churn
from .reference import credit_scoring as ref_credit_scoring

# A per-customer CLV vector for the metrics that require one. Length is checked against the
# fixtures that use it, so keep it long enough for the largest of them.
CLV_VECTOR = np.linspace(100, 500, 64)

# A uniform per-customer CLV. The reference `empb`/`auepc_score` index `clv` per sample, so the
# "constant clv" case is a constant *vector*, not a scalar.
CONSTANT_CLV = np.full(64, 200.0)


@dataclass(frozen=True)
class PrebuiltCase:
    """One prebuilt metric, its reference, and the inputs to compare them on."""

    name: str
    metric: Any
    """The public ``empulse.metrics`` object under test."""
    reference: Callable[..., Any]
    """The ``tests/metrics/reference`` implementation used as ground truth."""
    params: Sequence[Mapping[str, Any]]
    """Parameter sets to sweep. The first entry is conventionally ``{}`` (all defaults)."""
    invalid_params: Sequence[Mapping[str, Any]] = ()
    """Parameter sets the metric must reject: outside the domain its cost matrix declares."""
    unconstrained_params: Sequence[Mapping[str, Any]] = ()
    """Economically odd but mathematically valid values the metric deliberately accepts.

    A negative cost is how the package expresses a benefit -- ``max_profit_score`` relies on it --
    so money quantities carry no default bound. A user who wants one adds it with
    :meth:`~empulse.metrics.CostMatrix.constrain`.
    """
    base_params: Mapping[str, Any] = field(default_factory=dict)
    """Parameters this metric always needs (``empb``/``auepc`` require a per-customer ``clv``)."""
    to_reference_params: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    """Rewrites public parameter names/units into the reference's, where the two differ."""
    reference_returns_rate: bool = True
    """Whether ``reference`` returns ``(score, rate)`` rather than a bare score."""

    def resolve(self, y_true: ArrayLike, params: Mapping[str, Any]) -> dict[str, Any]:
        """
        Merge ``base_params`` with *params*, trimming per-sample vectors to ``len(y_true)``.

        The table is written without knowing how long the test's input will be, so any array-valued
        parameter is declared longer than the longest fixture and sliced down here.
        """
        n = len(y_true)
        return {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in {**self.base_params, **params}.items()}

    def base_for(self, y_true: ArrayLike) -> dict[str, Any]:
        return self.resolve(y_true, {})

    def call_metric(self, y_true, y_score, **params):
        return self.metric(y_true, y_score, **self.resolve(y_true, params))

    def call_reference(self, y_true, y_score, **params):
        merged = self.resolve(y_true, params)
        if self.to_reference_params is not None:
            merged = self.to_reference_params(merged)
        return self.reference(y_true, y_score, **merged)

    def reference_score(self, y_true, y_score, **params):
        result = self.call_reference(y_true, y_score, **params)
        return result[0] if self.reference_returns_rate else result

    def reference_rate(self, y_true, y_score, **params):
        if not self.reference_returns_rate:
            raise AssertionError(f'{self.name} has no reference rate')
        return self.call_reference(y_true, y_score, **params)[1]


def _empa_to_reference(params):
    """``empa_score`` takes the Gamma *scale*; the reference ``empa`` takes the *rate*."""
    converted = dict(params)
    converted['beta'] = 1 / converted.get('beta', 1 / 0.0015)
    return converted


CASES: list[PrebuiltCase] = [
    PrebuiltCase(
        name='empc_score',
        metric=empc_score,
        reference=ref_churn.empc,
        params=[
            {},
            {'alpha': 2},
            {'beta': 5},
            {'clv': 50},
            {'incentive_cost': 100},
            {'contact_cost': 15},
            {'alpha': 10, 'beta': 7, 'clv': 300, 'incentive_cost': 50, 'contact_cost': 25},
        ],
        invalid_params=[
            {'alpha': -1},
            {'beta': -1},
        ],
        unconstrained_params=[
            {'clv': -1},
            {'clv': 1},
            {'incentive_cost': -1},
            {'contact_cost': -1},
        ],
    ),
    PrebuiltCase(
        name='mpc_score',
        metric=mpc_score,
        reference=ref_churn.mpc,
        params=[
            {},
            {'accept_rate': 0.2},
            {'clv': 50},
            {'incentive_cost': 100},
            {'contact_cost': 15},
            {'accept_rate': 0.9, 'clv': 300, 'incentive_cost': 50, 'contact_cost': 25},
        ],
        invalid_params=[
            {'accept_rate': -1},
            {'accept_rate': 2},
        ],
        unconstrained_params=[
            {'clv': -1},
            {'clv': 1},
            {'incentive_cost': -1},
            {'contact_cost': -1},
        ],
    ),
    PrebuiltCase(
        name='empb_score',
        metric=empb_score,
        reference=ref_churn.empb,
        params=[
            {},  # `base_params` supplies an instance-dependent (per-customer) clv vector
            {'clv': CONSTANT_CLV},  # ... and this overrides it with a uniform clv across customers
            {'alpha': 2},
            {'beta': 5},
            {'incentive_fraction': 0.9},
            {'contact_cost': 15},
            {'alpha': 10, 'beta': 7, 'incentive_fraction': 0.02, 'contact_cost': 25},
        ],
        invalid_params=[
            {'alpha': -1},
            {'beta': -1},
            {'incentive_fraction': -1},
            {'incentive_fraction': 2},
        ],
        unconstrained_params=[
            {'contact_cost': -1},
        ],
        base_params={'clv': CLV_VECTOR},
    ),
    PrebuiltCase(
        name='auepc_score',
        metric=auepc_score,
        reference=ref_churn.auepc_score,
        params=[
            {},  # `base_params` supplies an instance-dependent (per-customer) clv vector
            {'clv': CONSTANT_CLV},  # ... and this overrides it with a uniform clv across customers
            {'alpha': 2},
            {'beta': 5},
            {'incentive_fraction': 0.9},
            {'contact_cost': 15},
            {'alpha': 10, 'beta': 7, 'incentive_fraction': 0.02, 'contact_cost': 25},
        ],
        invalid_params=[
            {'alpha': -1},
            {'beta': -1},
            {'incentive_fraction': -1},
            {'incentive_fraction': 2},
        ],
        unconstrained_params=[
            {'contact_cost': -1},
        ],
        base_params={'clv': CLV_VECTOR},
        reference_returns_rate=False,
    ),
    PrebuiltCase(
        name='expected_cost_loss_churn',
        metric=expected_cost_loss_churn,
        reference=partial(ref_churn.expected_cost_loss_churn, normalize=True),
        params=[
            {},
            {'accept_rate': 0.2},
            {'clv': 50},
            {'incentive_fraction': 0.5},
            {'contact_cost': 15},
            {'accept_rate': 0.9, 'clv': 300, 'incentive_fraction': 0.1, 'contact_cost': 25},
        ],
        invalid_params=[
            {'accept_rate': -1},
            {'accept_rate': 2},
            {'incentive_fraction': -1},
            {'incentive_fraction': 2},
        ],
        unconstrained_params=[
            {'contact_cost': -1},
        ],
        reference_returns_rate=False,
    ),
    PrebuiltCase(
        name='empa_score',
        metric=empa_score,
        reference=ref_acquisition.empa,
        params=[
            {},
            {'alpha': 10},
            {'beta': 0.02},  # exercises the scale-vs-rate conversion in `_empa_to_reference`
            {'contact_cost': 100},
            {'sales_cost': 1000},
            {'direct_selling': 0.5},
            {'direct_selling': 0.0, 'commission': 0.5},
            {
                'alpha': 10,
                'beta': 0.0012,
                'contact_cost': 80,
                'sales_cost': 400,
                'direct_selling': 0.3,
                'commission': 0.2,
            },
        ],
        invalid_params=[
            {'alpha': -1},
            {'beta': -1},
            {'direct_selling': 5},
            {'direct_selling': -1},
            {'commission': 5},
            {'commission': -1},
        ],
        unconstrained_params=[
            {'sales_cost': -1},
            {'contact_cost': -1},
        ],
        to_reference_params=_empa_to_reference,
    ),
    PrebuiltCase(
        name='mpa_score',
        metric=mpa_score,
        reference=ref_acquisition.mpa,
        params=[
            {},
            {'contribution': 2000},
            {'contact_cost': 100},
            {'sales_cost': 1000},
            {'direct_selling': 0.5},
            {'direct_selling': 0.0, 'commission': 0.5},
            {'contribution': 12000, 'contact_cost': 80, 'sales_cost': 400, 'direct_selling': 0.3, 'commission': 0.2},
        ],
        invalid_params=[
            {'direct_selling': 5},
            {'direct_selling': -1},
            {'commission': 5},
            {'commission': -1},
        ],
        unconstrained_params=[
            {'contribution': -1},
            {'sales_cost': -1},
            {'contact_cost': -1},
        ],
    ),
    PrebuiltCase(
        name='expected_cost_loss_acquisition',
        metric=expected_cost_loss_acquisition,
        reference=partial(ref_acquisition.expected_cost_loss_acquisition, normalize=True),
        params=[
            {},
            {'contribution': 2000},
            {'contact_cost': 100},
            {'sales_cost': 1000},
            {'direct_selling': 0.5},
            {'direct_selling': 0.0, 'commission': 0.5},
            {'contribution': 12000, 'contact_cost': 80, 'sales_cost': 400, 'direct_selling': 0.3, 'commission': 0.2},
        ],
        invalid_params=[
            {'direct_selling': 5},
            {'direct_selling': -1},
            {'commission': 5},
            {'commission': -1},
        ],
        unconstrained_params=[
            {'contribution': -1},
            {'sales_cost': -1},
            {'contact_cost': -1},
        ],
        reference_returns_rate=False,
    ),
    PrebuiltCase(
        name='empcs_score',
        metric=empcs_score,
        reference=ref_credit_scoring.empcs,
        params=[
            {},
            {'success_rate': 0.7},
            {'default_rate': 0.01},
            {'roi': 0.1},
            {'success_rate': 0.01, 'default_rate': 0.7, 'roi': 0.7},
        ],
        invalid_params=[
            {'success_rate': -1},
            {'success_rate': 2},
            {'default_rate': -1},
            {'default_rate': 2},
        ],
        unconstrained_params=[
            {'roi': -1},
        ],
    ),
    PrebuiltCase(
        name='mpcs_score',
        metric=mpcs_score,
        reference=ref_credit_scoring.mpcs,
        params=[
            {},
            {'loan_lost_rate': 0.7},
            {'roi': 0.1},
            {'loan_lost_rate': 0.01, 'roi': 0.7},
        ],
        invalid_params=[
            {'loan_lost_rate': -1},
            {'loan_lost_rate': 2},
        ],
        unconstrained_params=[
            {'roi': -1},
        ],
    ),
]

CASES_BY_NAME = {case.name: case for case in CASES}


def case_id(case: PrebuiltCase) -> str:
    return case.name


def params_id(params: Mapping[str, Any]) -> str:
    return '+'.join(params) if params else 'defaults'
