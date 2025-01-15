import numpy as np
import pytest

from empulse.samplers.bias_resampler import BiasResampler


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
    for index, target in zip(X_re, y_re):
        assert y[index[1]] == target


def test_bias_resampler_balanced():  # no resampling needed
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    X[:, 1] = np.arange(10)
    y = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=X[:, 0])
    assert np.array_equal(X_re, X)
    assert np.array_equal(y_re, y)


def test_no_protected_attr():
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    y = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    sensitive_feature = np.array(
        [0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]
    )
    with pytest.warns(UserWarning, match="sensitive_feature only contains one class, no resampling is performed."):
        X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=sensitive_feature)
        assert np.array_equal(X_re, X)
        assert np.array_equal(y_re, y)


def test_all_protected_attr():
    X = np.array([[[1, 0]] * 5 + [[0, 0]] * 5]).reshape(10, 2)  # first feature is sensitive feature
    y = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    sensitive_feature = np.array(
        [1, 1, 1, 1, 1,
         1, 1, 1, 1, 1]
    )
    with pytest.warns(UserWarning, match="sensitive_feature only contains one class, no resampling is performed."):
        X_re, y_re = BiasResampler().fit_resample(X, y, sensitive_feature=sensitive_feature)
        assert np.array_equal(X_re, X)
        assert np.array_equal(y_re, y)
