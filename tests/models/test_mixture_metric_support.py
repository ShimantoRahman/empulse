"""Tests that MixtureMetric (and BaseMetric in general) can be used as a `loss` for model training.

Cost-sensitive models used to hard-code `Metric` as the only accepted `loss` type, both in
sklearn's `_parameter_constraints` and in scattered `isinstance(loss, Metric)` checks throughout
the codebase. This meant a `MixtureMetric` -- despite implementing every method a `Metric` does --
was rejected outright by sklearn's parameter validation before a model ever got a chance to use it.

These tests exercise the fix: models now accept any `BaseMetric`, the shared abstract interface
`Metric` and `MixtureMetric` both implement.
"""

import numpy as np
import pytest
import sympy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._param_validation import InvalidParameterError

from empulse.metrics import BaseMetric, CostMatrix, MaxProfit, Metric, MixtureComponent, MixtureMetric
from empulse.models import CSBoostClassifier, CSLogitClassifier, CSRateClassifier, CSThresholdClassifier


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=0, class_sep=1.5)
    return X, y


@pytest.fixture()
def two_point_mixture():
    """A MaxProfit MixtureMetric mixing two point masses of `gamma` -- gradient-trainable today."""
    gamma, roi = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
    return MixtureMetric([
        MixtureComponent(0.6, metric_det, {'gamma': 0.0}),
        MixtureComponent(0.4, metric_det, {'gamma': 1.0}),
    ])


def test_mixture_metric_is_a_base_metric(two_point_mixture):
    assert isinstance(two_point_mixture, BaseMetric)
    assert BaseMetric in CSLogitClassifier._parameter_constraints['loss']


def test_cslogit_rejects_non_metric_loss(dataset):
    """Parameter validation should still reject something that isn't a BaseMetric at all."""
    X, y = dataset
    model = CSLogitClassifier(loss='not a metric')  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError):
        model.fit(X, y)


def test_cslogit_accepts_mixture_metric_loss(dataset, two_point_mixture):
    from scipy.optimize import OptimizeResult

    X, y = dataset
    model = CSLogitClassifier(loss=two_point_mixture, C=1e6, l1_ratio=0.0, soft_threshold=False)
    model.fit(X, y, roi=0.2644)

    assert isinstance(model.result_, OptimizeResult)
    assert model.coef_.shape == (X.shape[1],)
    assert np.all(np.isfinite(model.coef_))

    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    assert y_proba.shape == (X.shape[0], 2)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_cslogit_mixture_metric_matches_weighted_average_gamma(dataset):
    """A two-point MixtureMetric mixing gamma=0/gamma=1 should fit like a plain Metric at the

    weighted-average gamma, since tp_benefit=gamma is linear: this is the mathematical property
    MixtureMetric relies on. The match won't be bit-exact (each fit is an independent L-BFGS run),
    but the mixture's fit should land unambiguously closer to the weighted average than to either
    extreme or to a mismatched average.
    """
    X, y = dataset
    gamma, roi = sympy.symbols('gamma roi')
    common_kwargs = {'C': 1e6, 'l1_ratio': 0.0, 'soft_threshold': False}

    def fit_plain(g):
        metric = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
        model = CSLogitClassifier(loss=metric, **common_kwargs)
        model.fit(X, y, gamma=g, roi=0.2644)
        return model.coef_

    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
    mixture = MixtureMetric([
        MixtureComponent(0.6, metric_det, {'gamma': 0.0}),
        MixtureComponent(0.4, metric_det, {'gamma': 1.0}),
    ])
    mixture_model = CSLogitClassifier(loss=mixture, **common_kwargs)
    mixture_model.fit(X, y, roi=0.2644)

    coef_weighted_average = fit_plain(0.4)  # 0.6 * 0 + 0.4 * 1
    coef_mismatched = fit_plain(0.9)

    dist_to_correct_average = np.linalg.norm(mixture_model.coef_ - coef_weighted_average)
    dist_to_mismatched_average = np.linalg.norm(mixture_model.coef_ - coef_mismatched)
    assert dist_to_correct_average < dist_to_mismatched_average


def test_cslogit_mixture_metric_weights_are_not_ignored(dataset):
    """Two mixtures with swapped weights must fit to different coefficients.

    This guards against a class of bug where a mixture's weights are silently dropped and every
    component ends up contributing equally (or only one component's `parameters` override wins).
    """
    X, y = dataset
    gamma, roi = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
    common_kwargs = {'C': 1e6, 'l1_ratio': 0.0, 'soft_threshold': False}

    mixture_a = MixtureMetric([
        MixtureComponent(0.9, metric_det, {'gamma': 0.0}),
        MixtureComponent(0.1, metric_det, {'gamma': 1.0}),
    ])
    mixture_b = MixtureMetric([
        MixtureComponent(0.1, metric_det, {'gamma': 0.0}),
        MixtureComponent(0.9, metric_det, {'gamma': 1.0}),
    ])

    model_a = CSLogitClassifier(loss=mixture_a, **common_kwargs)
    model_a.fit(X, y, roi=0.2644)
    model_b = CSLogitClassifier(loss=mixture_b, **common_kwargs)
    model_b.fit(X, y, roi=0.2644)

    assert not np.allclose(model_a.coef_, model_b.coef_, atol=1e-3)


def test_csthreshold_accepts_mixture_metric_loss(dataset, two_point_mixture):
    X, y = dataset
    model = CSThresholdClassifier(LogisticRegression(), loss=two_point_mixture)
    model.fit(X, y, roi=0.2644)

    assert model.threshold_ is not None
    assert np.isfinite(model.threshold_)
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_csrate_accepts_mixture_metric_loss(dataset, two_point_mixture):
    X, y = dataset
    model = CSRateClassifier(LogisticRegression(), loss=two_point_mixture)
    model.fit(X, y, roi=0.2644)

    assert model.rate_ is not None
    assert 0.0 <= model.rate_ <= 1.0
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_csboost_accepts_mixture_metric_loss(dataset, two_point_mixture):
    xgboost = pytest.importorskip('xgboost')

    X, y = dataset
    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=5, max_depth=1, verbosity=0),
        loss=two_point_mixture,
    )
    model.fit(X, y, roi=0.2644)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.all(np.isfinite(y_proba))
