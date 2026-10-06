"""
Unit tests for :class:`~empulse.samplers.CostSensitiveSampler`.

This file existed but was empty -- zero bytes, from the commit that added the sampler onward. The
sampler was therefore reached only by the generic scikit-learn/imbalanced-learn conformance checks
in ``test_samplers.py`` plus two class-count assertions in ``sampler_checks.py``; nothing exercised
what makes it cost-sensitive, namely that the costs change which rows survive.
"""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, Metric, MixtureComponent, MixtureMetric
from empulse.samplers import CostSensitiveSampler

METHODS = ['rejection sampling', 'oversampling']


@pytest.fixture(scope='module')
def data():
    X, y = make_classification(n_samples=200, n_features=5, weights=[0.7, 0.3], random_state=42)
    return X, y


@pytest.mark.parametrize('method', METHODS)
def test_fit_resample_returns_consistent_shapes(data, method):
    X, y = data
    sampler = CostSensitiveSampler(method=method, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y, fp_cost=1.0, fn_cost=5.0)
    assert X_res.shape[1] == X.shape[1]
    assert len(X_res) == len(y_res)
    assert set(np.unique(y_res)) <= set(np.unique(y))


@pytest.mark.parametrize('method', METHODS)
def test_costs_change_which_rows_survive(data, method):
    """
    The whole point of the sampler: different costs must produce a different sample.

    Nothing in the suite checked this, so the sampler could have ignored its costs entirely and
    still passed every conformance check.
    """
    X, y = data
    cheap = CostSensitiveSampler(method=method, random_state=42).fit_resample(X, y, fp_cost=1.0, fn_cost=1.0)
    expensive = CostSensitiveSampler(method=method, random_state=42).fit_resample(X, y, fp_cost=1.0, fn_cost=50.0)
    assert not (cheap[0].shape == expensive[0].shape and np.array_equal(cheap[0], expensive[0])), (
        f'{method} produced an identical sample for fn_cost=1 and fn_cost=50'
    )


@pytest.mark.parametrize('method', METHODS)
def test_is_reproducible_for_a_fixed_random_state(data, method):
    X, y = data
    first = CostSensitiveSampler(method=method, random_state=7).fit_resample(X, y, fp_cost=1.0, fn_cost=5.0)
    second = CostSensitiveSampler(method=method, random_state=7).fit_resample(X, y, fp_cost=1.0, fn_cost=5.0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


@pytest.mark.parametrize('method', METHODS)
def test_accepts_instance_dependent_costs(data, method):
    """Per-sample cost vectors are the sampler's headline feature alongside scalars."""
    X, y = data
    rng = np.random.default_rng(42)
    fn_cost = rng.uniform(1, 10, size=len(y))
    sampler = CostSensitiveSampler(method=method, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y, fp_cost=1.0, fn_cost=fn_cost)
    assert len(X_res) == len(y_res)


def test_rejection_sampling_does_not_grow_the_dataset(data):
    X, y = data
    X_res, _ = CostSensitiveSampler(method='rejection sampling', random_state=42).fit_resample(
        X, y, fp_cost=1.0, fn_cost=5.0
    )
    assert len(X_res) <= len(X)


def test_oversampling_does_not_shrink_the_dataset(data):
    X, y = data
    X_res, _ = CostSensitiveSampler(method='oversampling', random_state=42).fit_resample(X, y, fp_cost=1.0, fn_cost=5.0)
    assert len(X_res) >= len(X)


def test_cost_matrix_symbols_are_routed_to_the_loss(data):
    """A ``Metric`` loss turns its free symbols into ``fit_resample`` keyword arguments."""
    X, y = data
    loss = Metric(CostMatrix().add_fp_cost('a').add_fn_cost('b'), Cost())
    sampler = CostSensitiveSampler(method='rejection sampling', loss=loss, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y, a=1.0, b=5.0)
    assert len(X_res) == len(y_res)


@pytest.mark.parametrize('method', METHODS)
def test_all_zero_costs_warns_and_falls_back(data, method):
    """
    With every cost zero the sampler has nothing to weigh, so it falls back to ``fp = fn = 1``.

    The warning text is suppressed globally by ``filterwarnings`` in ``pyproject.toml`` (sklearn's
    own conformance checks trip it constantly), which is why it needs an explicit ``pytest.warns``
    rather than relying on warnings surfacing normally.
    """
    X, y = data
    sampler = CostSensitiveSampler(method=method, random_state=42)
    with pytest.warns(UserWarning, match='All costs are zero'):
        X_res, y_res = sampler.fit_resample(X, y, fp_cost=0.0, fn_cost=0.0)

    expected = CostSensitiveSampler(method=method, random_state=42).fit_resample(X, y, fp_cost=1.0, fn_cost=1.0)
    assert np.array_equal(X_res, expected[0]), 'the fallback did not use fp_cost = fn_cost = 1'
    assert np.array_equal(y_res, expected[1])


def test_constructor_costs_are_used_when_fit_resample_omits_them(data):
    """``Parameter.UNCHANGED`` means "keep whatever the constructor set"."""
    X, y = data
    from_init = CostSensitiveSampler(random_state=42, fp_cost=1.0, fn_cost=5.0).fit_resample(X, y)
    from_fit = CostSensitiveSampler(random_state=42).fit_resample(X, y, fp_cost=1.0, fn_cost=5.0)
    assert np.array_equal(from_init[0], from_fit[0])
    assert np.array_equal(from_init[1], from_fit[1])


def test_fit_resample_costs_override_constructor_costs(data):
    """The other half of ``Parameter.UNCHANGED``, which had no test anywhere."""
    X, y = data
    overridden = CostSensitiveSampler(random_state=42, fp_cost=1.0, fn_cost=1.0).fit_resample(
        X, y, fp_cost=1.0, fn_cost=50.0
    )
    as_if_passed_directly = CostSensitiveSampler(random_state=42).fit_resample(X, y, fp_cost=1.0, fn_cost=50.0)
    assert np.array_equal(overridden[0], as_if_passed_directly[0])

    kept_init_costs = CostSensitiveSampler(random_state=42, fp_cost=1.0, fn_cost=1.0).fit_resample(X, y)
    assert not (
        overridden[0].shape == kept_init_costs[0].shape and np.array_equal(overridden[0], kept_init_costs[0])
    ), 'the fit-time fn_cost did not override the constructor value'


def test_mixture_metric_loss_is_used_not_silently_ignored(data):
    """
    A ``MixtureMetric`` loss must actually be consulted, not silently fall back to plain costs.

    The loss-or-plain-costs branch must not check ``isinstance(loss, Metric)``, which a
    ``MixtureMetric`` never satisfies (it implements ``BaseMetric`` directly). The mixture would then
    fall through to the plain-cost branch, where the unset ``fp_cost``/``fn_cost`` constructor defaults
    (``0.0``) trigger the all-zero-costs fallback and discard the mixture entirely.
    """
    X, y = data
    cost_matrix = CostMatrix().add_fp_cost('a').add_fn_cost('b')
    mixture = MixtureMetric([MixtureComponent(1.0, Metric(cost_matrix, Cost()), {})])
    plain = Metric(cost_matrix, Cost())

    from_mixture = CostSensitiveSampler(method='rejection sampling', loss=mixture, random_state=42).fit_resample(
        X, y, a=1.0, b=5.0
    )
    from_plain = CostSensitiveSampler(method='rejection sampling', loss=plain, random_state=42).fit_resample(
        X, y, a=1.0, b=5.0
    )
    assert np.array_equal(from_mixture[0], from_plain[0])
    assert np.array_equal(from_mixture[1], from_plain[1])


class TestCostSensitiveSamplerRouting:
    """CostSensitiveSampler routes `fit_resample`, not `fit`; see `tests/models/test_metadata_routing.py`."""

    @pytest.fixture(autouse=True)
    def _enable_metadata_routing(self):
        with config_context(enable_metadata_routing=True):
            yield

    def test_two_samplers_keep_independent_routing_keys(self):
        a = CostSensitiveSampler(loss=Metric(CostMatrix().add_fp_cost('clv').add_fn_cost('other'), Cost()))
        b = CostSensitiveSampler(loss=Metric(CostMatrix().add_fp_cost('roi').add_fn_cost('another'), Cost()))

        a.set_fit_resample_request(clv=True)
        b.set_fit_resample_request(roi=True)

        assert a.get_metadata_routing().fit_resample.requests['clv'] is True
        with pytest.raises(TypeError, match='roi'):
            a.set_fit_resample_request(roi=True)

    def test_no_loss_still_routes_plain_costs(self):
        sampler = CostSensitiveSampler()
        sampler.set_fit_resample_request(fp_cost=True, fn_cost=True)
        assert sampler.get_metadata_routing().fit_resample.requests['fp_cost'] is True


@pytest.mark.parametrize('fn_cost', [np.inf, np.nan])
def test_non_finite_costs_are_rejected(data, fn_cost):
    X, y = data
    with pytest.raises(ValueError, match='fn_cost must be finite'):
        CostSensitiveSampler().fit_resample(X, y, fp_cost=1.0, fn_cost=fn_cost)


class TestMisclassificationCosts:
    """Each sample is kept in proportion to what deciding it wrongly costs over deciding it rightly."""

    @pytest.fixture
    def labels(self, seeded_rng):
        return (seeded_rng.random(2000) < 0.3).astype(int)

    @staticmethod
    def kept_fractions(sampler, y):
        kept = np.bincount(y[sampler.sample_indices_], minlength=2)
        return kept / np.bincount(y, minlength=2)

    @pytest.mark.parametrize(('fp_cost', 'fn_cost'), [(1.0, 2.5), (0.5, 1.0), (0.4, 0.6)])
    def test_fractional_costs_are_not_truncated(self, labels, fp_cost, fn_cost):
        X = np.zeros((labels.size, 1))
        sampler = CostSensitiveSampler(percentile_threshold=1.0, random_state=0)
        sampler.fit_resample(X, labels, fp_cost=fp_cost, fn_cost=fn_cost)

        np.testing.assert_allclose(self.kept_fractions(sampler, labels), [fp_cost / fn_cost, 1.0], atol=0.05)

    def test_a_benefit_of_a_correct_decision_counts_like_a_cost_of_a_wrong_one(self, labels):
        X = np.zeros((labels.size, 1))
        with_benefit = Metric(CostMatrix().add_tp_benefit('b').add_fp_cost('c'), Cost())
        with_cost = Metric(CostMatrix().add_fn_cost('b').add_fp_cost('c'), Cost())

        benefit_sampler = CostSensitiveSampler(loss=with_benefit, random_state=0)
        benefit_sampler.fit_resample(X, labels, b=4.0, c=1.0)
        cost_sampler = CostSensitiveSampler(loss=with_cost, random_state=0)
        cost_sampler.fit_resample(X, labels, b=4.0, c=1.0)

        np.testing.assert_array_equal(benefit_sampler.sample_indices_, cost_sampler.sample_indices_)
        np.testing.assert_allclose(self.kept_fractions(benefit_sampler, labels), [0.25, 1.0], atol=0.05)
