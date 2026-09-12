"""
The shared contract every prebuilt domain metric in ``empulse.metrics`` must satisfy.

The metric table lives in ``prebuilt_cases.py``, which also documents the two intentional
package-vs-reference behaviour differences.

Two groups of tests are marked ``xfail``. They encode the contract the reference implementations
enforced and the current package does not -- see ``test_bad_parameters_are_rejected`` and
``test_single_class_y_true_is_rejected``. ``xfail_strict`` is on, so if validation is ever
restored these turn red and the marker has to be removed; they are a self-maintaining record of
the gap, not dead tests.
"""

import numpy as np
import pytest

from empulse.metrics import empa_score, empc_score, lift_score, mpa_score, mpc_score

from .prebuilt_cases import CASES, case_id, params_id

# (y_true, y_score) inputs that every metric must reject, and the exception it must raise.
# `None` in an array makes NumPy build an object array, hence TypeError rather than ValueError.
INVALID_INPUTS = [
    pytest.param([1, 0, np.nan], [0.25, 0.75, 0.5], ValueError, id='nan_in_y_true'),
    pytest.param([1, 0, None], [0.25, 0.75, 0.5], TypeError, id='none_in_y_true'),
    pytest.param([1, 0, 1], [0.25, 0.75, np.nan], ValueError, id='nan_in_y_score'),
    pytest.param([1, 0, 1], [0.25, 0.75, None], TypeError, id='none_in_y_score'),
    pytest.param([1, 0, np.inf], [0.25, 0.75, 0.5], ValueError, id='inf_in_y_true'),
    pytest.param([1, 0, 1], [0.25, 0.75, np.inf], ValueError, id='inf_in_y_score'),
    pytest.param(['a', 'b'], [0.25, 0.75], TypeError, id='non_numeric_y_true'),
    pytest.param([0, 1], ['a', 'b'], TypeError, id='non_numeric_y_score'),
    pytest.param([0, 0, 1], [0.25, 0.75], ValueError, id='y_score_too_short'),
    pytest.param([0, 1], [0.25, 0.5, 0.75], ValueError, id='y_score_too_long'),
]


def _case_params():
    """Every (case, parameter set) pair, flattened for parametrization."""
    for case in CASES:
        for params in case.params:
            yield pytest.param(case, params, id=f'{case.name}-{params_id(params)}')


def _case_bad_params():
    for case in CASES:
        for params in case.bad_params:
            yield pytest.param(case, params, id=f'{case.name}-{params_id(params)}')


@pytest.fixture(scope='module')
def ranking_data():
    """A 20-row ranking problem with both classes present."""
    rng = np.random.default_rng(42)
    n = 20
    y_true = rng.integers(0, 2, n).astype(float)
    y_true[0], y_true[1] = 0.0, 1.0  # guarantee both classes regardless of the draw
    return y_true, rng.random(n)


# --- Agreement with the reference implementations ----------------------------------------------


@pytest.mark.parametrize(('case', 'params'), _case_params())
def test_score_matches_reference(case, params, ranking_data):
    y_true, y_score = ranking_data
    assert case.call_metric(y_true, y_score, **params) == pytest.approx(case.reference_score(y_true, y_score, **params))


@pytest.mark.parametrize(
    ('case', 'params'),
    [p for p in _case_params() if p.values[0].reference_returns_rate],
)
def test_optimal_rate_matches_reference(case, params, ranking_data):
    """
    The reference functions return ``(score, predicted_positive_rate)``.

    It is the *rate* that ``Metric.optimal_rate`` reproduces -- ``optimal_threshold`` is a
    different quantity (a cutoff on the score scale) and does not match this value.
    """
    y_true, y_score = ranking_data
    rate = case.metric.optimal_rate(y_true, y_score, **case.resolve(y_true, params))
    assert rate == pytest.approx(case.reference_rate(y_true, y_score, **params))


# --- The shared edge-case battery --------------------------------------------------------------


@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_perfect_prediction_matches_reference(case):
    y_true = [0, 1] * 10
    assert case.call_metric(y_true, y_true) == pytest.approx(case.reference_score(y_true, y_true))


@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_incorrect_prediction_matches_reference(case):
    y_true, y_score = [0, 1] * 10, [1, 0] * 10
    assert case.call_metric(y_true, y_score) == pytest.approx(case.reference_score(y_true, y_score))


@pytest.mark.parametrize('constant_score', [[1, 1], [0, 0], [0.5, 0.5]], ids=['ones', 'zeros', 'halves'])
@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_uninformative_prediction_matches_reference(case, constant_score):
    """A constant score ranks nothing, so the metric must fall back on the class priors alone."""
    y_true, y_score = [0, 1] * 10, constant_score * 10
    assert case.call_metric(y_true, y_score) == pytest.approx(case.reference_score(y_true, y_score))


@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_accepts_array_likes(case):
    """Lists, tuples and pandas Series must all give the same answer as a NumPy array."""
    pd = pytest.importorskip('pandas')
    y_true, y_score = [0, 1] * 10, [0.1, 0.9] * 10
    expected = case.call_metric(np.asarray(y_true), np.asarray(y_score))
    assert case.call_metric(tuple(y_true), tuple(y_score)) == pytest.approx(expected)
    assert case.call_metric(pd.Series(y_true), pd.Series(y_score)) == pytest.approx(expected)


@pytest.mark.parametrize(('y_true', 'y_score', 'expected_exception'), INVALID_INPUTS)
@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_invalid_input_is_rejected(case, y_true, y_score, expected_exception):
    with pytest.raises(expected_exception):
        case.call_metric(y_true, y_score)


@pytest.mark.xfail(
    reason=(
        'The prebuilt metrics no longer reject a single-class y_true. The reference '
        'implementations raised ValueError; the Metric-based replacements return a number '
        '(often nan) instead, because nothing validates that both classes are present.'
    )
)
@pytest.mark.parametrize('label', [0, 1], ids=['all_negative', 'all_positive'])
@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_single_class_y_true_is_rejected(case, label):
    with pytest.raises(ValueError):
        case.call_metric([label, label], [0.25, 0.75])


@pytest.mark.xfail(
    reason=(
        'Domain validation of the business parameters was lost when these metrics moved from '
        'hand-written native math to CostMatrix/Metric. The reference implementations rejected '
        'every one of these with ValueError; the package now returns a number -- sometimes nan, '
        'sometimes a plausible-looking but meaningless value (e.g. accept_rate=2).'
    )
)
@pytest.mark.parametrize(('case', 'params'), _case_bad_params())
def test_bad_parameters_are_rejected(case, params):
    with pytest.raises(ValueError):
        case.call_metric([0, 1], [0.25, 0.75], **params)


# --- Relationships between the stochastic and deterministic members of a family ----------------


def _relation_data(rng):
    """A 1000-row problem with a randomly drawn class prior and Beta-distributed scores."""
    n = 1000
    n_positive = max(1, int(rng.random() * n))
    scale = lambda x: rng.random() * x
    y_score = np.concatenate([
        rng.beta(scale(20) + 1e-9, scale(20) + 1e-9, n_positive),
        rng.beta(scale(20) + 1e-9, scale(20) + 1e-9, n - n_positive),
    ])
    y_true = np.concatenate([np.ones(n_positive, dtype=np.int8), np.zeros(n - n_positive, dtype=np.int8)])
    return y_true, y_score


@pytest.mark.parametrize('trial', range(25))
def test_deterministic_churn_metric_is_a_lower_bound(trial):
    """
    MPC must not exceed EMPC.

    The deterministic metric fixes the accept rate at the mean of EMPC's Beta prior, so it is the
    profit at a single point of a distribution the stochastic metric maximises over.
    """
    rng = np.random.default_rng(trial)
    y_true, y_score = _relation_data(rng)
    contact_cost = rng.uniform(0, 100)
    incentive_cost = rng.uniform(0, 500)
    # Bounded well below the overflow regime documented by
    # `test_empc_score_overflows_for_a_concentrated_beta_prior` below.
    alpha, beta = rng.uniform(0.1, 30), rng.uniform(0.1, 30)
    clv = max(rng.uniform(0, 1000), contact_cost + incentive_cost + 1)
    shared = {'clv': clv, 'incentive_cost': incentive_cost, 'contact_cost': contact_cost}

    stochastic = empc_score(y_true, y_score, alpha=alpha, beta=beta, **shared)
    deterministic = mpc_score(y_true, y_score, accept_rate=alpha / (alpha + beta), **shared)
    assert deterministic <= stochastic + 1e-6


@pytest.mark.parametrize('trial', range(25))
def test_deterministic_acquisition_metric_is_a_lower_bound(trial):
    """MPA must not exceed EMPA, for the same reason as the churn pair."""
    rng = np.random.default_rng(trial)
    y_true, y_score = _relation_data(rng)
    alpha, beta = rng.uniform(1e-5, 100), rng.uniform(1e-5, 100)
    shared = {
        'sales_cost': rng.uniform(0, 500),
        'contact_cost': rng.uniform(0, 100),
        'commission': rng.uniform(0, 1),
        'direct_selling': rng.uniform(0, 1),
    }
    # `empa_score` takes the Gamma scale, so the deterministic contribution -- the distribution's
    # mean -- is `alpha * beta`, not `alpha / beta`.
    stochastic = empa_score(y_true, y_score, alpha=alpha, beta=beta, **shared)
    deterministic = mpa_score(y_true, y_score, contribution=alpha * beta, **shared)
    assert deterministic <= stochastic + 1e-6


@pytest.mark.xfail(
    reason=(
        'empc_score returns nan once alpha + beta is large (roughly >= 160). The Beta moments are '
        'computed as (gamma(alpha + k) * gamma(alpha + beta)) / (gamma(alpha) * gamma(alpha + beta + k)) '
        'in max_profit_strategy/piecewise.py, and the numerator overflows float64 long before any of '
        'the individual gamma calls do. Evaluating the ratio in log space (lgamma/betaln) would fix it. '
        'A concentrated Beta prior is a legitimate way to express a confidently-known accept rate, so '
        'this is reachable from ordinary use.'
    )
)
@pytest.mark.parametrize('concentration', [80, 100, 200], ids=lambda c: f'alpha=beta={c}')
def test_empc_score_overflows_for_a_concentrated_beta_prior(concentration):
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.ones(600, dtype=np.int8), np.zeros(400, dtype=np.int8)])
    y_score = rng.beta(2, 2, 1000)
    score = empc_score(y_true, y_score, alpha=concentration, beta=concentration, clv=200)
    assert np.isfinite(score), f'empc_score returned {score} for alpha = beta = {concentration}'


# --- lift_score ---------------------------------------------------------------------------------


def test_lift_score_perfect_prediction():
    assert lift_score([0, 1] * 10, [0, 1] * 10) == pytest.approx(2.0)


def test_lift_score_uninformative_prediction():
    assert lift_score([0, 1] * 10, [0.5] * 20) == pytest.approx(1.0)


@pytest.mark.parametrize(('y_true', 'y_score', 'expected_exception'), INVALID_INPUTS)
def test_lift_score_rejects_invalid_input(y_true, y_score, expected_exception):
    with pytest.raises(expected_exception):
        lift_score(y_true, y_score)
