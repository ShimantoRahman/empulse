"""
Models trained on a :class:`~empulse.metrics.BaseMetric` passed as ``loss``.

Every cost-sensitive model accepts a ``Metric`` (or a ``MixtureMetric``) whose symbols become ``fit``
-- or, for the decision-rule models, ``predict`` -- keyword arguments. These check that such a loss
is equivalent to the plain cost arguments it can express, that its parameters accept the same array
shapes and dtypes, that defaults and metadata routing work, and that a ``MixtureMetric`` is accepted
everywhere a ``Metric`` is.
"""

import pickle
from typing import ClassVar

import numpy as np
import pytest
import sympy
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils._param_validation import InvalidParameterError
from xgboost import XGBClassifier

from empulse.metrics import (
    BaseMetric,
    Cost,
    CostMatrix,
    LogCost,
    MaxProfit,
    Metric,
    MinCost,
    MixtureComponent,
    MixtureMetric,
    Profit,
)
from empulse.models import (
    CSBaggingClassifier,
    CSBoostClassifier,
    CSForestClassifier,
    CSLogitClassifier,
    CSRateClassifier,
    CSThresholdClassifier,
    CSTreeClassifier,
    ProfLogitClassifier,
    ProfTreeClassifier,
    RobustCSClassifier,
)
from empulse.optimizers import GeneticAlgorithmOptimizer, LBFGSBOptimizer

from .estimator_inventory import estimator_id

METRIC_ESTIMATORS = (
    ProfLogitClassifier(optimizer=GeneticAlgorithmOptimizer(max_iter=100, population_size=10, random_state=42)),
    ProfTreeClassifier(max_iter=2, population_size=10, random_state=42),
    CSBoostClassifier(XGBClassifier(n_estimators=5, max_depth=2)),
    CSLogitClassifier(optimizer=LBFGSBOptimizer(max_iter=10)),
    CSTreeClassifier(max_depth=2),
    CSForestClassifier(n_estimators=3, max_depth=1, random_state=10),
    CSBaggingClassifier(n_estimators=3, random_state=10),
    RobustCSClassifier(estimator=CSBoostClassifier(XGBClassifier(n_estimators=5, max_depth=2))),
    CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid', random_state=42),
    CSRateClassifier(estimator=LogisticRegression(max_iter=2)),
)

PREDICT_TIME_ESTIMATORS = (
    CSThresholdClassifier(LogisticRegression(), calibrator='sigmoid', random_state=42),
    CSRateClassifier(estimator=LogisticRegression(max_iter=2)),
)


def set_metric_loss(estimator, loss):
    """Set the metric loss for the estimator."""
    if isinstance(estimator, RobustCSClassifier):
        estimator.estimator.loss = loss
        return estimator
    elif hasattr(estimator, 'loss'):
        return estimator.set_params(loss=loss)
    elif hasattr(estimator, 'criterion'):
        return estimator.set_params(criterion=loss)
    elif hasattr(estimator, 'estimator') and hasattr(estimator.estimator, 'loss'):
        return estimator.set_params(estimator__loss=loss)
    else:
        raise ValueError(f'Estimator {estimator} does not support setting a loss function.')


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_metric_api_consistency(estimator, cost_dataset):
    """Test that the metric API is consistent with the cost matrix API."""
    X, y, _, _ = cost_dataset
    strategy_source = estimator.estimator if isinstance(estimator, RobustCSClassifier) else estimator
    kind = strategy_source._default_metric_strategy()

    loss = Metric(CostMatrix().add_fn_cost('a').add_fp_cost('b'), kind)
    model_metric = set_metric_loss(clone(estimator), loss)
    model = clone(estimator)

    if isinstance(model, CSThresholdClassifier):
        model.fit(X, y)
        model_metric.fit(X, y)

        preds_metric = model_metric.predict(X, a=1, b=1)
        preds_metric_weighted = model_metric.predict(X, a=1, b=10)
        preds = model.predict(X, fp_cost=1, fn_cost=1)
        assert np.allclose(preds_metric, preds), 'Predictions are not consistent with the metric API.'
        assert not np.allclose(preds_metric_weighted, preds), (
            'Predictions of the metric API do not change with weights.'
        )
    elif isinstance(model, CSRateClassifier):
        # CSRateClassifier computes optimal rate at fit time (rate depends on training data).
        model_metric.fit(X, y, a=1, b=1)
        model.fit(X, y, fp_cost=1, fn_cost=1)

        preds_metric = model_metric.predict(X)
        preds = model.predict(X)
        assert np.allclose(preds_metric, preds), 'Predictions are not consistent with the metric API.'

        model_metric.fit(X, y, a=1, b=10)
        preds_metric_weighted = model_metric.predict(X)
        assert not np.allclose(preds_metric_weighted, preds), (
            'Predictions of the metric API do not change with weights.'
        )
    else:
        model_metric.fit(X, y, a=1, b=1)
        model.fit(X, y, fp_cost=1, fn_cost=1)

        preds_metric = model_metric.predict_proba(X)[:, 1]
        preds = model.predict_proba(X)[:, 1]
        assert np.allclose(preds_metric, preds), 'Predictions are not consistent with the metric API.'

        # For evolutionary estimators with very few iterations, use more iterations to ensure
        # the algorithm is sensitive to cost weights (avoids platform-dependent flakiness).
        model_metric_weighted = (
            set_metric_loss(clone(estimator).set_params(max_iter=100), loss)
            if isinstance(model, ProfTreeClassifier)
            else model_metric
        )
        model_metric_weighted.fit(X, y, a=1, b=10)
        preds_metric_weighted = model_metric_weighted.predict_proba(X)[:, 1]
        assert not np.allclose(preds_metric_weighted, preds), (
            'Predictions of the metric API do not change with weights.'
        )


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_data_format(estimator, cost_dataset):
    """Test that the estimators accept data in different formats."""
    X, y, _, _ = cost_dataset
    tp_cost = 0
    tn_cost = np.expand_dims(np.zeros(y.size), axis=1)
    fn_cost = np.ones(y.size)
    fp_cost = np.expand_dims(np.ones(y.size), axis=0)

    estimator = clone(estimator)
    if isinstance(estimator, CSThresholdClassifier):
        estimator.fit(X, y)
        estimator.predict(X, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
    else:
        estimator.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost, tp_cost=tp_cost, tn_cost=tn_cost)


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_data_format_metric_loss(estimator, cost_dataset):
    """Test that the estimators accept data in different formats when using metric loss."""
    X, y, _, _ = cost_dataset
    tp_cost = 0
    tn_cost = np.expand_dims(np.zeros(y.size), axis=1)
    fn_cost = np.ones(y.size)
    fp_cost = np.expand_dims(np.ones(y.size), axis=0)

    cost_matrix = CostMatrix().add_tp_cost('tp').add_tn_cost('tn').add_fn_cost('fn').add_fp_cost('fp')
    cost_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), cost_loss)

    if isinstance(estimator, CSThresholdClassifier):
        estimator.fit(X, y)
        estimator.predict(X, tp=tp_cost, tn=tn_cost, fn=fn_cost, fp=fp_cost)
    else:
        estimator.fit(X, y, tp=tp_cost, tn=tn_cost, fn=fn_cost, fp=fp_cost)


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_data_types_metric_loss(estimator, cost_dataset):
    """Test that the estimators accept different data types when using metric loss."""
    X, y, _, _ = cost_dataset
    tp_cost = 0
    # +0.5 keeps tn_cost off integer values so fp_cost - tn_cost + fn_cost - tp_cost (the optimal
    # threshold's denominator) never lands on exactly 0 for any sample - a genuinely degenerate
    # cost matrix that Metric.optimal_threshold() now correctly rejects, which isn't what this test is about.
    tn_cost = np.arange(y.size, dtype=np.float32) + 0.5
    fn_cost = np.ones(y.size, dtype=np.int32)
    fp_cost = np.expand_dims(np.ones(y.size, dtype=np.float64), axis=0)

    tp, tn, fn, fp = sympy.symbols('tp tn fn fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_tn_cost(tn).add_fn_cost(fn).add_fp_cost(fp)
    cost_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), cost_loss)

    if isinstance(estimator, CSThresholdClassifier):
        estimator.fit(X, y)
        estimator.predict(X, tp=tp_cost, tn=tn_cost, fn=fn_cost, fp=fp_cost)
    else:
        estimator.fit(X, y, tp=tp_cost, tn=tn_cost, fn=fn_cost, fp=fp_cost)


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_metric_loss_all_default_params(estimator, cost_dataset):
    """Test that the metric loss works with all default parameters."""
    X, y, _, _ = cost_dataset

    fn, fp = sympy.symbols('fn fp')
    cost_matrix = CostMatrix().add_fn_cost(fn).add_fp_cost(fp).set_default(fp=1, fn=1)
    cost_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), cost_loss)

    if isinstance(estimator, CSThresholdClassifier):
        estimator.fit(X, y)
        estimator.predict(X)
    else:
        estimator.fit(X, y)


@pytest.mark.parametrize('estimator', PREDICT_TIME_ESTIMATORS, ids=estimator_id)
def test_data_format_metric_loss_predict_time(estimator, cost_dataset):
    """Test that predict-time estimators accept data in different formats through the Metric loss API."""
    X, y, _, _ = cost_dataset
    tp_cost = 0
    tn_cost = np.expand_dims(np.zeros(y.size), axis=1)
    fn_cost = np.ones(y.size)
    fp_cost = np.expand_dims(np.ones(y.size), axis=0)

    cost_matrix = CostMatrix().add_tp_cost('tp').add_tn_cost('tn').add_fn_cost('fn').add_fp_cost('fp')
    metric_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), metric_loss)
    estimator.fit(X, y)
    y_pred = estimator.predict(X, tp=tp_cost, tn=tn_cost, fn=fn_cost, fp=fp_cost)
    assert y_pred.shape == y.shape


@pytest.mark.parametrize('estimator', PREDICT_TIME_ESTIMATORS, ids=estimator_id)
def test_data_types_metric_loss_predict_time(estimator, cost_dataset):
    """Test that predict-time estimators accept different data types through the Metric loss API."""
    X, y, _, _ = cost_dataset
    tp_cost = 0
    # +0.5 keeps tn_cost off integer values so fp_cost - tn_cost + fn_cost - tp_cost (the optimal
    # threshold's denominator) never lands on exactly 0 for any sample - a genuinely degenerate
    # cost matrix that Metric.optimal_threshold() now correctly rejects (see
    # METRIC_CORE_REVIEW.md finding 08), which isn't what this test is about.
    tn_cost = np.arange(y.size, dtype=np.float32) + 0.5
    fn_cost = np.ones(y.size, dtype=np.int32)
    fp_cost = np.expand_dims(np.ones(y.size, dtype=np.float64), axis=0)

    cost_matrix = CostMatrix().add_tp_cost('tp').add_tn_cost('tn').add_fn_cost('fn').add_fp_cost('fp')
    metric_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), metric_loss)
    estimator.fit(X, y)
    y_pred = estimator.predict(X, tp=tp_cost, tn=tn_cost, fn=fn_cost, fp=fp_cost)
    assert y_pred.shape == y.shape


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_metric_loss_partial_defaults(estimator, cost_dataset):
    """Test that the metric loss works when only some parameters have defaults."""
    X, y, _, _ = cost_dataset

    cost_matrix = CostMatrix().add_fn_cost('fn').add_fp_cost('fp').set_default(fp=1)
    cost_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), cost_loss)
    estimator.fit(X, y, fn=1)
    y_pred = estimator.predict(X)
    assert y_pred.shape == y.shape


@pytest.mark.parametrize('estimator', PREDICT_TIME_ESTIMATORS, ids=estimator_id)
def test_metric_loss_partial_defaults_predict_time(estimator, cost_dataset):
    """Test that the metric loss works when only some parameters have defaults."""
    X, y, _, _ = cost_dataset

    cost_matrix = CostMatrix().add_fn_cost('fn').add_fp_cost('fp').set_default(fp=1)
    metric_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), metric_loss)
    estimator.fit(X, y)
    y_pred = estimator.predict(X, fn=1)
    assert y_pred.shape == y.shape


@pytest.mark.parametrize('estimator', METRIC_ESTIMATORS, ids=estimator_id)
def test_metric_loss_metadata_routing(estimator, cost_dataset):
    """Test that the metric loss metadata routing works."""
    X, y, fn_cost, fp_cost = cost_dataset

    cost_matrix = CostMatrix().add_fn_cost('fn').add_fp_cost('fp').mark_outlier_sensitive('fn')
    cost_loss = Metric(cost_matrix, Cost())

    estimator = set_metric_loss(clone(estimator), cost_loss)

    with config_context(enable_metadata_routing=True):
        if isinstance(estimator, CSThresholdClassifier | CSRateClassifier):
            estimator.set_fit_request(fp=True, fn=True)
            estimator.set_predict_request(fp=True, fn=True)
            cross_val_score(estimator, X, y, cv=2, params={'fp': fp_cost, 'fn': fn_cost})
        else:
            estimator.set_fit_request(fp=True, fn=True)
            cross_val_score(estimator, X, y, cv=2, params={'fp': fp_cost, 'fn': fn_cost})


# --- MixtureMetric as a loss ---------------------------------------------------------------------
#
# Cost-sensitive models used to hard-code `Metric` as the only accepted `loss` type, both in sklearn's
# `_parameter_constraints` and in scattered `isinstance(loss, Metric)` checks. A `MixtureMetric` --
# despite implementing every method a `Metric` does -- was rejected by parameter validation before a
# model ever got to use it. Models now accept any `BaseMetric`.


@pytest.fixture(scope='module')
def mixture_dataset():
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


def test_cslogit_rejects_non_metric_loss(mixture_dataset):
    """Parameter validation should still reject something that isn't a BaseMetric at all."""
    X, y = mixture_dataset
    model = CSLogitClassifier(loss='not a metric')  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError):
        model.fit(X, y)


def test_cslogit_accepts_mixture_metric_loss(mixture_dataset, two_point_mixture):
    from scipy.optimize import OptimizeResult

    X, y = mixture_dataset
    model = CSLogitClassifier(loss=two_point_mixture, C=1e6, l1_ratio=0.0)
    model.fit(X, y, roi=0.2644)

    assert isinstance(model.result_, OptimizeResult)
    assert model.coef_.shape == (X.shape[1],)
    assert np.all(np.isfinite(model.coef_))

    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    assert y_proba.shape == (X.shape[0], 2)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_cslogit_mixture_metric_matches_weighted_average_gamma(mixture_dataset):
    """A two-point MixtureMetric mixing gamma=0/gamma=1 should fit like a plain Metric at the

    weighted-average gamma, since tp_benefit=gamma is linear: this is the mathematical property
    MixtureMetric relies on. The match won't be bit-exact (each fit is an independent L-BFGS run),
    but the mixture's fit should land unambiguously closer to the weighted average than to either
    extreme or to a mismatched average.
    """
    X, y = mixture_dataset
    gamma, roi = sympy.symbols('gamma roi')
    common_kwargs = {'C': 1e6, 'l1_ratio': 0.0}

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


def test_cslogit_mixture_metric_weights_are_not_ignored(mixture_dataset):
    """Two mixtures with swapped weights must fit to different coefficients.

    This guards against a class of bug where a mixture's weights are silently dropped and every
    component ends up contributing equally (or only one component's `parameters` override wins).
    """
    X, y = mixture_dataset
    gamma, roi = sympy.symbols('gamma roi')
    metric_det = Metric(CostMatrix().add_tp_benefit(gamma).add_fp_cost(roi), MaxProfit())
    common_kwargs = {'C': 1e6, 'l1_ratio': 0.0}

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


def test_csthreshold_accepts_mixture_metric_loss(mixture_dataset, two_point_mixture):
    X, y = mixture_dataset
    model = CSThresholdClassifier(LogisticRegression(), loss=two_point_mixture)
    model.fit(X, y, roi=0.2644)

    assert model.threshold_ is not None
    assert np.isfinite(model.threshold_)
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_csrate_accepts_mixture_metric_loss(mixture_dataset, two_point_mixture):
    X, y = mixture_dataset
    model = CSRateClassifier(LogisticRegression(), loss=two_point_mixture)
    model.fit(X, y, roi=0.2644)

    assert model.rate_ is not None
    assert 0.0 <= model.rate_ <= 1.0
    y_pred = model.predict(X)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_csboost_accepts_mixture_metric_loss(mixture_dataset, two_point_mixture):
    xgboost = pytest.importorskip('xgboost')

    X, y = mixture_dataset
    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=5, max_depth=1, verbosity=0),
        loss=two_point_mixture,
    )
    model.fit(X, y, roi=0.2644)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.all(np.isfinite(y_proba))


# --- LogCost as a loss ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def log_cost_dataset():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0, class_sep=1.5)
    return X, y.astype(float)


def test_cslogit_accepts_log_cost_metric(log_cost_dataset):
    from scipy.optimize import OptimizeResult

    X, y = log_cost_dataset
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    model = CSLogitClassifier(loss=metric, C=1e6, l1_ratio=0.0)
    model.fit(X, y, tp=0.0, fp=1.0)

    assert isinstance(model.result_, OptimizeResult)
    assert model.coef_.shape == (X.shape[1],)
    assert np.all(np.isfinite(model.coef_))

    y_proba = model.predict_proba(X)
    assert y_proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(y_proba))


def test_cslogit_log_cost_is_picklable_after_fit(log_cost_dataset):
    """The fitted model (holding the LogCost-based Metric as `loss`) must be picklable,

    mirroring what sklearn does during cross-validation / grid search.
    """
    X, y = log_cost_dataset
    cost_matrix = CostMatrix().add_tp_benefit(1.0).add_fp_cost(1.0)
    metric = Metric(cost_matrix, LogCost())

    model = CSLogitClassifier(loss=metric, C=1e6, l1_ratio=0.0)
    model.fit(X, y)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))


def test_csboost_dispatches_log_cost_through_dynamic_gradient_boost_objective(log_cost_dataset):
    """CSBoostClassifier must route LogCost through the dynamic gradient_boost_objective path.

    LogCost's per-sample loss is non-linear in the predicted probability, so it cannot use the
    generic ``cy_boost_grad_hess`` kernel that Cost/Savings rely on (that kernel assumes a single
    precomputed constant gradient, which only holds for a loss that is linear in the score). This
    does not require xgboost/lightgbm/catboost to be installed: it inspects what callable
    ``_get_objective`` builds for the xgboost backend.
    """
    from functools import partial

    from empulse.models.boosting._backends import BoostingBackend

    _X, y = log_cost_dataset
    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    model = CSBoostClassifier(loss=metric)
    xgboost_backend = BoostingBackend(name='xgboost', classifier=None)
    objective = model._get_objective(xgboost_backend, y=y, loss=metric, tp=0.0, fp=1.0)

    assert isinstance(objective, partial)
    assert objective.func == metric._gradient_boost_objective

    # Sanity check: the returned objective actually computes finite gradients/hessians.
    y_score = np.zeros_like(y)
    gradient, hessian = objective(y, y_score)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))


def test_csboost_accepts_log_cost_metric(log_cost_dataset):
    xgboost = pytest.importorskip('xgboost')
    X, y = log_cost_dataset

    tp, fp = sympy.symbols('tp fp')
    cost_matrix = CostMatrix().add_tp_cost(tp).add_fp_cost(fp)
    metric = Metric(cost_matrix, LogCost())

    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=5, max_depth=1, verbosity=0),
        loss=metric,
    )
    model.fit(X, y, tp=0.0, fp=1.0)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.all(np.isfinite(y_proba))


# --- Sign-flipped siblings train identically ---------------------------------------------------------


class TestModelsTrainIdenticallyOnEitherPhrasing:
    """
    The point of the sign-flipped sibling strategies: which phrasing you pick cannot change the model.

    ``Profit``/``Cost`` and ``MinCost``/``MaxProfit`` hand an estimator the same loss; see
    ``tests/metrics/test_strategy_contract.py``.
    """

    COSTS: ClassVar[dict[str, float]] = {'fp_cost': 1.0, 'fn_cost': 5.0}

    @pytest.fixture(scope='class')
    def data(self, make_data):
        return make_data(n_samples=200, n_features=5, random_state=0)

    @pytest.fixture(scope='class')
    def cost_matrix(self):
        return CostMatrix().add_fp_cost('fp_cost').add_fn_cost('fn_cost')

    def test_cslogit(self, data, cost_matrix):
        X, y = data
        cost_model = CSLogitClassifier(loss=Metric(cost_matrix, Cost())).fit(X, y, **self.COSTS)
        profit_model = CSLogitClassifier(loss=Metric(cost_matrix, Profit())).fit(X, y, **self.COSTS)

        np.testing.assert_allclose(cost_model.coef_, profit_model.coef_)

    def test_csboost(self, data, cost_matrix):
        X, y = data
        cost_model = CSBoostClassifier(
            XGBClassifier(n_estimators=10, random_state=0), loss=Metric(cost_matrix, Cost())
        ).fit(X, y, **self.COSTS)
        profit_model = CSBoostClassifier(
            XGBClassifier(n_estimators=10, random_state=0), loss=Metric(cost_matrix, Profit())
        ).fit(X, y, **self.COSTS)

        np.testing.assert_allclose(cost_model.predict_proba(X), profit_model.predict_proba(X))

    def test_proftree_takes_the_same_fast_path(self, data, cost_matrix):
        """MinCost is a MaxProfit, so ProfTree uses the Cython fit_max_profit path for both."""
        X, y = data
        profit_model = ProfTreeClassifier(loss=Metric(cost_matrix, MaxProfit()), max_iter=10, random_state=0).fit(
            X, y, **self.COSTS
        )
        cost_model = ProfTreeClassifier(loss=Metric(cost_matrix, MinCost()), max_iter=10, random_state=0).fit(
            X, y, **self.COSTS
        )

        np.testing.assert_allclose(cost_model.predict_proba(X), profit_model.predict_proba(X))
