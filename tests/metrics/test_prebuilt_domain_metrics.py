"""
Equivalence tests for the thin-wrapper prebuilt metrics in ``metrics.acquisition``,
``metrics.churn``, and ``metrics.credit_scoring``.

These domains used to be implemented as hand-written native math functions. They are now built
from :class:`~empulse.metrics.Metric`/:class:`~empulse.metrics.MixtureMetric` instances (see
``empulse/metrics/metric/prebuilt_metrics.py``). The native math still exists, unchanged, in
``tests/metrics/reference/`` purely as ground truth -- these tests check that the new instances
numerically reproduce it (subject to the two documented, intentional behavior differences noted
below).

Notes
-----
* ``expected_cost_loss_churn``/``expected_cost_loss_acquisition`` are built on the ``Cost``
  strategy, which always returns the *mean* cost across samples. The reference functions default
  to returning the *sum* (``normalize=False``), so comparisons below pass ``normalize=True`` to
  the reference call.
* ``empa_score``'s ``beta`` parameter is the *scale* of the Gamma-distributed contribution
  (mean = alpha * beta), not the *rate* (mean = alpha / beta) used by the reference ``empa``.
  This is a consequence of how :class:`~empulse.metrics.MaxProfit`'s stochastic-integration
  engine parameterizes distributions (it requires distribution arguments to be plain symbols,
  not derived expressions such as ``1 / beta``). Comparisons below convert accordingly.
"""

import numpy as np
import pytest

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


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, n).astype(float)
    y_score = rng.random(n)
    clv = rng.uniform(100, 500, n)
    return y_true, y_score, clv


# --- Customer churn ---------------------------------------------------------------------------


def test_mpc_score_matches_reference_default(dataset):
    y_true, y_score, _clv = dataset
    assert mpc_score(y_true, y_score, clv=200) == pytest.approx(ref_churn.mpc_score(y_true, y_score))


def test_mpc_score_matches_reference_custom_parameters(dataset):
    y_true, y_score, _clv = dataset
    params = {'clv': 300, 'incentive_cost': 15, 'contact_cost': 2, 'accept_rate': 0.4}
    assert mpc_score(y_true, y_score, **params) == pytest.approx(ref_churn.mpc_score(y_true, y_score, **params))


def test_empc_score_matches_reference_default(dataset):
    y_true, y_score, _clv = dataset
    assert empc_score(y_true, y_score, clv=200) == pytest.approx(ref_churn.empc_score(y_true, y_score))


def test_empc_score_matches_reference_custom_parameters(dataset):
    y_true, y_score, _clv = dataset
    params = {'clv': 250, 'incentive_cost': 12, 'contact_cost': 2, 'alpha': 5, 'beta': 10}
    assert empc_score(y_true, y_score, **params) == pytest.approx(ref_churn.empc_score(y_true, y_score, **params))


def test_expected_cost_loss_churn_matches_reference_default(dataset):
    y_true, y_score, _clv = dataset
    assert expected_cost_loss_churn(y_true, y_score, clv=200) == pytest.approx(
        ref_churn.expected_cost_loss_churn(y_true, y_score, normalize=True)
    )


def test_expected_cost_loss_churn_matches_reference_custom_parameters(dataset):
    y_true, y_score, _clv = dataset
    params = {'clv': 220, 'incentive_fraction': 0.1, 'contact_cost': 3, 'accept_rate': 0.5}
    assert expected_cost_loss_churn(y_true, y_score, **params) == pytest.approx(
        ref_churn.expected_cost_loss_churn(y_true, y_score, **params, normalize=True)
    )


def test_empb_score_matches_reference_constant_clv(dataset):
    y_true, y_score, _clv = dataset
    clv = np.full(len(y_true), 200.0)
    assert empb_score(y_true, y_score, clv=clv) == pytest.approx(ref_churn.empb_score(y_true, y_score, clv=clv))


def test_empb_score_matches_reference_instance_dependent_clv(dataset):
    y_true, y_score, clv = dataset
    params = {'alpha': 5, 'beta': 12, 'incentive_fraction': 0.08, 'contact_cost': 10}
    assert empb_score(y_true, y_score, clv=clv, **params) == pytest.approx(
        ref_churn.empb_score(y_true, y_score, clv=clv, **params)
    )


def test_auepc_score_matches_reference_instance_dependent_clv(dataset):
    y_true, y_score, clv = dataset
    assert auepc_score(y_true, y_score, clv=clv) == pytest.approx(ref_churn.auepc_score(y_true, y_score, clv=clv))


def test_auepc_score_matches_reference_custom_parameters(dataset):
    y_true, y_score, clv = dataset
    params = {'alpha': 4, 'beta': 9, 'incentive_fraction': 0.07, 'contact_cost': 20}
    assert auepc_score(y_true, y_score, clv=clv, **params) == pytest.approx(
        ref_churn.auepc_score(y_true, y_score, clv=clv, **params)
    )


# --- Customer acquisition ----------------------------------------------------------------------


def test_mpa_score_matches_reference_direct(dataset):
    y_true, y_score, _clv = dataset
    assert mpa_score(y_true, y_score, contribution=8000) == pytest.approx(ref_acquisition.mpa_score(y_true, y_score))


def test_mpa_score_matches_reference_indirect(dataset):
    y_true, y_score, _clv = dataset
    params = {'contribution': 7000, 'sales_cost': 2000, 'contact_cost': 100, 'direct_selling': 0}
    assert mpa_score(y_true, y_score, **params) == pytest.approx(ref_acquisition.mpa_score(y_true, y_score, **params))


def test_empa_score_matches_reference_direct(dataset):
    y_true, y_score, _clv = dataset
    assert empa_score(y_true, y_score, direct_selling=1) == pytest.approx(
        ref_acquisition.empa_score(y_true, y_score, direct_selling=1)
    )


def test_empa_score_matches_reference_indirect_custom_beta_scale(dataset):
    """`beta` is the Gamma *scale* here, `1 / beta` is the reference's *rate* parameter."""
    y_true, y_score, _clv = dataset
    rate = 0.001
    new_params = {'alpha': 10, 'beta': 1 / rate, 'sales_cost': 2000, 'contact_cost': 100, 'direct_selling': 0}
    ref_params = {'alpha': 10, 'beta': rate, 'sales_cost': 2000, 'contact_cost': 100, 'direct_selling': 0}
    assert empa_score(y_true, y_score, **new_params) == pytest.approx(
        ref_acquisition.empa_score(y_true, y_score, **ref_params)
    )


def test_expected_cost_loss_acquisition_matches_reference_default(dataset):
    y_true, y_score, _clv = dataset
    assert expected_cost_loss_acquisition(y_true, y_score, contribution=7000) == pytest.approx(
        ref_acquisition.expected_cost_loss_acquisition(y_true, y_score, normalize=True)
    )


def test_expected_cost_loss_acquisition_matches_reference_custom_parameters(dataset):
    y_true, y_score, _clv = dataset
    params = {'contribution': 9000, 'sales_cost': 400, 'contact_cost': 60, 'direct_selling': 0.5, 'commission': 0.2}
    assert expected_cost_loss_acquisition(y_true, y_score, **params) == pytest.approx(
        ref_acquisition.expected_cost_loss_acquisition(y_true, y_score, **params, normalize=True)
    )


# --- Credit scoring -----------------------------------------------------------------------------


def test_mpcs_score_matches_reference_default(dataset):
    y_true, y_score, _clv = dataset
    assert mpcs_score(y_true, y_score) == pytest.approx(ref_credit_scoring.mpcs_score(y_true, y_score))


def test_mpcs_score_matches_reference_custom_parameters(dataset):
    y_true, y_score, _clv = dataset
    params = {'roi': 0.2, 'loan_lost_rate': 0.25}
    expected = ref_credit_scoring.mpcs_score(y_true, y_score, **params)
    assert mpcs_score(y_true, y_score, **params) == pytest.approx(expected)


def test_empcs_score_matches_reference_default(dataset):
    y_true, y_score, _clv = dataset
    assert empcs_score(y_true, y_score) == pytest.approx(ref_credit_scoring.empcs_score(y_true, y_score))


def test_empcs_score_matches_reference_custom_parameters(dataset):
    y_true, y_score, _clv = dataset
    params = {'roi': 0.2, 'success_rate': 0.5, 'default_rate': 0.1}
    assert empcs_score(y_true, y_score, **params) == pytest.approx(
        ref_credit_scoring.empcs_score(y_true, y_score, **params)
    )


def test_empcs_score_optimal_rate_matches_reference(dataset):
    y_true, y_score, _clv = dataset
    rate = empcs_score.optimal_rate(y_true, y_score)
    _, expected_rate = ref_credit_scoring.empcs(y_true, y_score)
    assert rate == pytest.approx(expected_rate)
