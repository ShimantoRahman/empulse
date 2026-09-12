"""
Unit tests for :class:`~empulse.samplers.CostSensitiveSampler`.

This file existed but was empty -- zero bytes, from the commit that added the sampler onward. The
sampler was therefore reached only by the generic scikit-learn/imbalanced-learn conformance checks
in ``test_samplers.py`` plus two class-count assertions in ``sampler_checks.py``; nothing exercised
what makes it cost-sensitive, namely that the costs change which rows survive.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from empulse.metrics import Cost, CostMatrix, Metric
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
