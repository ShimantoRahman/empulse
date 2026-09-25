"""
The bias-mitigating samplers, BiasRelabler and BiasResampler, and the group weights they share.

Their scikit-learn and imbalanced-learn conformance is checked in ``test_samplers.py``.
"""

import numpy as np
import pytest

from empulse._common._strategies import _independent_weights
from empulse.samplers.bias_relabler import (
    BiasRelabler,
    _get_demotion_candidates,
    _get_promotion_candidates,
    _independent_pairs,
)
from empulse.samplers.bias_resampler import BiasResampler


def test_independent_weights():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    weights = _independent_weights(y_true, protected_attr)
    unpriviledged_neg, priviledged_neg, unpriviledged_pos, priviledged_pos = weights.flatten()
    assert pytest.approx(unpriviledged_neg, abs=1e-3) == 0.666666
    assert pytest.approx(priviledged_neg, abs=1e-3) == 2
    assert pytest.approx(unpriviledged_pos, abs=1e-3) == 1.5
    assert pytest.approx(priviledged_pos, abs=1e-3) == 0.75


# --- BiasRelabler ------------------------------------------------------------------------------


def test_n_pairs_uneven():
    y_true = [1, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert _independent_pairs(y_true, protected_attr) == 1


def test_n_pairs_even():
    y_true = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert _independent_pairs(y_true, protected_attr) == 0


def test_relabler_no_protected_attr():
    y_true = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    protected_attr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    with pytest.warns(UserWarning, match='sensitive_feature only contains one class, no relabeling is performed.'):
        assert _independent_pairs(y_true, protected_attr) == 0


def test_relabler_all_protected_attr():
    y_true = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    protected_attr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    with pytest.warns(UserWarning, match='sensitive_feature only contains one class, no relabeling is performed.'):
        assert _independent_pairs(y_true, protected_attr) == 0


def test_demotion_pairs():
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_true = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    n_pairs = 2
    assert np.all(_get_demotion_candidates(y_pred, y_true, n_pairs) == np.array([1, 5]))


def test_promotion_pairs():
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_true = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    n_pairs = 2
    assert np.all(_get_promotion_candidates(y_pred, y_true, n_pairs) == np.array([3, 4]))


class _FixedProbabilities:
    """An estimator that predicts the same probabilities whatever it is fitted on."""

    def __init__(self, proba):
        self.proba = np.asarray(proba)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.column_stack((1 - self.proba, self.proba))

    def get_params(self, deep=False):
        # BiasRelabler fits a clone, which is rebuilt from these.
        return {'proba': self.proba}


PROBA = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@pytest.mark.parametrize(
    ('proba', 'expected'),
    [
        (PROBA, [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        (PROBA[::-1], [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]),
    ],
    ids=['ascending', 'descending'],
)
def test_bias_relabler(proba, expected):
    X = np.array([[0, 0] * 10]).reshape(10, 2)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    protected_attr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    relabler = BiasRelabler(_FixedProbabilities(proba))
    X, y = relabler.fit_resample(X, y, sensitive_feature=protected_attr)
    assert np.all(y == np.array(expected))


# --- BiasResampler -----------------------------------------------------------------------------


def statistical_parity(y, sensitive_feature):
    """
    Calculate the statistical parity.

    Parameters
    ----------
    y : np.ndarray
        Target variable for each sample.
    sensitive_feature : np.ndarray
        Sensitive feature for each sample.

    Returns
    -------
    statistical_parity : float
        The statistical parity value.
    """
    # Calculate the probabilities of positive outcomes for protected and non-protected groups
    prob_positive_protected = y[sensitive_feature == 1].mean()
    prob_positive_non_protected = y[sensitive_feature == 0].mean()

    # Calculate the statistical parity
    statistical_parity = prob_positive_protected - prob_positive_non_protected

    return statistical_parity


def test_bias_resampler_unbalanced():
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    X[:, 1] = np.arange(10)  # second feature is just an index
    y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=X[:, 0])
    assert statistical_parity(y_re, X_re[:, 0]) == 0.0
    # check the indices of the resampled data still match their target value in y
    for index, target in zip(X_re, y_re, strict=False):
        assert y[index[1]] == target


def test_bias_resampler_balanced():  # no resampling needed
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    X[:, 1] = np.arange(10)
    y = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=X[:, 0])
    assert np.array_equal(X_re, X)
    assert np.array_equal(y_re, y)


def test_resampler_no_protected_attr():
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    y = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    sensitive_feature = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    with pytest.warns(UserWarning, match='sensitive_feature only contains one class, no resampling is performed.'):
        X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=sensitive_feature)
    assert np.array_equal(X_re, X)
    assert np.array_equal(y_re, y)


def test_resampler_all_protected_attr():
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    y = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    sensitive_feature = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    with pytest.warns(UserWarning, match='sensitive_feature only contains one class, no resampling is performed.'):
        X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=sensitive_feature)
    assert np.array_equal(X_re, X)
    assert np.array_equal(y_re, y)
