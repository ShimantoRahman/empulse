from typing import TypeVar
from unittest import mock

import numpy as np
import pytest
import sympy
from scipy.special import expit
from sklearn.datasets import make_classification

import empulse.models
from empulse.metrics import (
    CostMatrix,
    LogCost,
    MaxProfit,
    Metric,
    empc_score,
    expected_cost_loss,
    expected_log_cost_loss,
    max_profit_score,
    mpc_score,
)
from empulse.models import CSBoostClassifier

CLASSIFIERS = [('xgboost', 'XGBClassifier'), ('lightgbm', 'LGBMClassifier'), ('catboost', 'CatBoostClassifier')]


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
def test_csboost_different_classifiers(library, classifier_name, cost_dataset):
    classifier_module = pytest.importorskip(library)
    classifier_class = getattr(classifier_module, classifier_name)

    X, y, fn_cost, fp_cost = cost_dataset
    extra = {'allow_writing_files': False} if library == 'catboost' else {}
    model = CSBoostClassifier(estimator=classifier_class(n_estimators=2, verbose=0, **extra))
    model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == y.shape
    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


def test_csboost_when_xgboost_is_missing(cost_dataset):
    X, y, fn_cost, fp_cost = cost_dataset
    with mock.patch.object(empulse.models.boosting.csboost, 'XGBClassifier', TypeVar('XGBClassifier')):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.') as exc_info:
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    # The message says how to get it.
    assert 'pip install' in str(exc_info.value)


def test_csboost_with_invalid_estimator_type(cost_dataset):
    """Test that CSBoostClassifier raises ValueError when an unsupported estimator type is provided."""
    X, y, fn_cost, fp_cost = cost_dataset

    from sklearn.ensemble import RandomForestClassifier

    model = CSBoostClassifier(estimator=RandomForestClassifier())

    with pytest.raises(
        TypeError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
    ):
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_with_deterministic_max_profit_metric(cost_dataset):
    xgboost = pytest.importorskip('xgboost')
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), MaxProfit(alpha=1.0))

    X, y, _, _ = cost_dataset
    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=2, max_depth=1, verbosity=0),
        loss=metric,
    )
    model.fit(X, y, clv=5.0)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize(
    'metric',
    [
        pytest.param(empc_score, id='empc_score'),
        pytest.param(mpc_score, id='mpc_score'),
        pytest.param(max_profit_score, id='max_profit_score'),
        pytest.param(expected_log_cost_loss, id='expected_log_cost_loss'),
        pytest.param(expected_cost_loss, id='expected_cost_loss'),
    ],
)
def test_csboost_fits_with_renamed_prebuilt_metrics(cost_dataset, metric):
    """CSBoostClassifier builds the right objective for renamed prebuilt metrics.

    Every prebuilt metric renames its strategy via `Metric.__name__` (e.g.
    `empc_score.__name__ = 'empc_score'`), and that must not stop `_get_objective` from handling
    MaxProfit/LogCost-backed metrics. All prebuilt metrics used here have defaults for every
    parameter, so no `loss_params` are needed.
    """
    xgboost = pytest.importorskip('xgboost')
    X, y, _, _ = cost_dataset
    model = CSBoostClassifier(estimator=xgboost.XGBClassifier(n_estimators=2, max_depth=1, verbosity=0), loss=metric)
    model.fit(X, y)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('strategy_factory', [MaxProfit, LogCost])
def test_csboost_dispatch_ignores_strategy_name(cost_dataset, strategy_factory):
    """Renaming a custom metric must not change which boosting objective is built.

    `CSBoostClassifier._get_objective` must not compare `loss.strategy.name` against
    `{'max profit', 'log cost'}`: a renamed metric (as every prebuilt metric is) would silently take
    the constant-gradient branch instead of raising, or - for a strategy that does implement
    `prepare_boost_objective` - train on the wrong gradients.
    """
    xgboost = pytest.importorskip('xgboost')
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), strategy_factory())
    metric.__name__ = 'renamed_metric'
    assert metric.strategy.name == 'renamed_metric'

    X, y, _, _ = cost_dataset
    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=2, max_depth=1, verbosity=0),
        loss=metric,
    )
    model.fit(X, y, clv=5.0)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


def test_csboost_when_all_libraries_missing(cost_dataset):
    """Test that CSBoostClassifier fails gracefully when all boosting libraries are missing."""
    X, y, fn_cost, fp_cost = cost_dataset

    with (
        mock.patch.object(empulse.models.boosting.csboost, 'XGBClassifier', TypeVar('XGBClassifier')),
        mock.patch.object(empulse.models.boosting.csboost, 'LGBMClassifier', TypeVar('LGBMClassifier')),
        mock.patch.object(empulse.models.boosting.csboost, 'CatBoostClassifier', TypeVar('CatBoostClassifier')),
    ):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.'):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_fit_does_not_mutate_callers_fit_params_dict(cost_dataset):
    """Reusing one `fit_params` dict across fits (or GridSearchCV folds) must leave it as it was."""
    catboost = pytest.importorskip('catboost')
    X, y, fn_cost, fp_cost = cost_dataset
    shared_fit_params = {'verbose': False}

    for _ in range(2):
        model = CSBoostClassifier(
            estimator=catboost.CatBoostClassifier(n_estimators=2, depth=1, allow_writing_files=False)
        )
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost, fit_params=shared_fit_params)
        assert shared_fit_params == {'verbose': False}


@pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
@pytest.mark.parametrize('through_fit_params', [False, True])
def test_csboost_rejects_sample_weight(library, classifier_name, through_fit_params, cost_dataset):
    """The costs already weigh each sample; a sample weight on top of them has no place."""
    classifier_class = getattr(pytest.importorskip(library), classifier_name)
    kwargs = {'n_estimators': 2}
    if library == 'catboost':
        kwargs.update(verbose=False, allow_writing_files=False)
    elif library == 'lightgbm':
        kwargs.update(verbosity=-1)
    X, y, fn_cost, fp_cost = cost_dataset
    weights = np.ones(len(y))
    extra = {'fit_params': {'sample_weight': weights}} if through_fit_params else {'sample_weight': weights}

    with pytest.raises(TypeError, match='sample_weight'):
        CSBoostClassifier(classifier_class(**kwargs)).fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost, **extra)


class TestPredictProbaFromRawScores:
    """Under a custom objective, LightGBM and CatBoost predict raw scores, so ``predict_proba`` applies the link."""

    def test_lightgbm(self, cost_dataset):
        lightgbm = pytest.importorskip('lightgbm')
        X, y, fn_cost, fp_cost = cost_dataset
        model = CSBoostClassifier(estimator=lightgbm.LGBMClassifier(n_estimators=5, max_depth=2, verbosity=-1))
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

        raw_score = model.estimator_.predict_proba(X, raw_score=True)
        np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(raw_score))

    def test_catboost(self, cost_dataset):
        catboost = pytest.importorskip('catboost')
        X, y, fn_cost, fp_cost = cost_dataset
        model = CSBoostClassifier(
            estimator=catboost.CatBoostClassifier(n_estimators=5, max_depth=2, verbose=False, allow_writing_files=False)
        )
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

        raw_score = model.estimator_.predict(X, prediction_type='RawFormulaVal')
        np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(raw_score))

    @pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
    def test_every_backend_starts_at_one_half(self, library, classifier_name):
        """With a near-zero learning rate the trees add nothing, leaving the shared starting probability."""
        classifier_module = pytest.importorskip(library)
        classifier_class = getattr(classifier_module, classifier_name)
        X, y = make_classification(n_samples=100, random_state=0)

        kwargs = {'n_estimators': 3, 'max_depth': 1, 'learning_rate': 1e-6}
        if library == 'catboost':
            kwargs.update(verbose=False, allow_writing_files=False)
        elif library == 'lightgbm':
            kwargs.update(verbosity=-1)
        else:
            kwargs.update(verbosity=0)

        model = CSBoostClassifier(estimator=classifier_class(**kwargs))
        model.fit(X, y, fp_cost=1.0, fn_cost=1.0)

        assert model.predict_proba(X)[:, 1].mean() == pytest.approx(0.5, abs=0.02)


@pytest.fixture(scope='module')
def cost_data(make_data):
    return make_data(n_samples=80, n_features=5)


class TestCatBoostBackend:
    """The CatBoost backend must not pass row indices through ``sample_weight``.

    CatBoost calls its objective on chunks of rows, out of order and without any row identifier, and
    it trains on ``sample_weight`` too. The backend therefore trains on the weighted reformulation
    built by ``_catboost_training_data``, so that no row is ignored or counted more and the model does
    not depend on row order.
    """

    def test_reformulated_derivatives_match_aec_kernel(self):
        from empulse.metrics._loss import cy_boost_grad_hess
        from empulse.models.boosting._backends import CatBoostObjective, _catboost_training_data

        rng = np.random.default_rng(0)
        grad_const = rng.normal(scale=3, size=200)
        grad_const[:10] = 0.0
        y = rng.integers(0, 2, size=200)
        raw_score = rng.normal(size=200)

        target, weight = _catboost_training_data(grad_const, y)
        ders = np.array(CatBoostObjective().calc_ders_range(raw_score, target, weight))
        gradient, hessian = cy_boost_grad_hess(y.astype(np.float64), raw_score, grad_const)

        np.testing.assert_allclose(-ders[:, 0], gradient, atol=1e-12)
        np.testing.assert_allclose(-ders[:, 1], hessian, atol=1e-12)

    def test_zero_gradient_rows_keep_their_label(self):
        from empulse.models.boosting._backends import _catboost_training_data

        target, weight = _catboost_training_data(np.array([-2.0, 3.0, 0.0, 0.0]), np.array([0, 1, 1, 0]))
        np.testing.assert_array_equal(target, [1, 0, 1, 0])
        np.testing.assert_array_equal(weight, [2.0, 3.0, 0.0, 0.0])

    def test_model_does_not_depend_on_row_order(self):
        catboost = pytest.importorskip('catboost')
        X, y = make_classification(n_samples=300, random_state=0)
        fn_cost = np.random.default_rng(0).gamma(2, 5, size=y.size)

        def fit(order):
            estimator = catboost.CatBoostClassifier(
                iterations=20,
                verbose=False,
                random_seed=0,
                bootstrap_type='No',
                random_strength=0,
                thread_count=1,
                allow_writing_files=False,
            )
            model = CSBoostClassifier(estimator).fit(X[order], y[order], fn_cost=fn_cost[order], fp_cost=3.0)
            return model.predict_proba(X)

        order = np.arange(y.size)
        np.testing.assert_allclose(fit(order), fit(order[::-1]), atol=1e-10)

    def test_eval_metric_ranks_models_like_expected_cost(self):
        from empulse.models.boosting._backends import CatBoostMetric, _catboost_training_data

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        fn_cost = rng.gamma(2, 5, size=100)
        grad_const = np.where(y == 1, -fn_cost, 3.0)  # Cost: y * (tp - fn) + (1 - y) * (fp - tn)
        target, weight = _catboost_training_data(grad_const, y)
        metric = CatBoostMetric()

        def evaluate(raw_score):
            return metric.get_final_error(*metric.evaluate([raw_score], target, weight))

        raw_1, raw_2 = rng.normal(size=100), rng.normal(size=100)
        cost_1 = expected_cost_loss(y, expit(raw_1), fn_cost=fn_cost, fp_cost=3.0)
        cost_2 = expected_cost_loss(y, expit(raw_2), fn_cost=fn_cost, fp_cost=3.0)
        # The metric is the expected cost up to a positive factor and a constant.
        assert cost_1 - cost_2 == pytest.approx(weight.sum() / y.size * (evaluate(raw_1) - evaluate(raw_2)))

    @pytest.mark.parametrize('strategy_factory', [MaxProfit, LogCost])
    def test_rejects_strategies_evaluated_per_round(self, cost_dataset, strategy_factory):
        """MaxProfit's gradient depends on every row and LogCost's on per-row costs; CatBoost gives neither."""
        catboost = pytest.importorskip('catboost')
        clv = sympy.symbols('clv')
        metric = Metric(CostMatrix().add_tp_benefit(clv), strategy_factory())
        X, y, _, _ = cost_dataset
        model = CSBoostClassifier(
            catboost.CatBoostClassifier(n_estimators=2, verbose=False, allow_writing_files=False), loss=metric
        )
        with pytest.raises(ValueError, match='The CatBoost backend does not support'):
            model.fit(X, y, clv=5.0)

    def test_costs_with_nothing_to_learn_raise(self, cost_dataset):
        catboost = pytest.importorskip('catboost')
        X, y, _, _ = cost_dataset
        model = CSBoostClassifier(catboost.CatBoostClassifier(n_estimators=2, verbose=False, allow_writing_files=False))
        # Predicting positive costs more than predicting negative for positives and negatives alike.
        with pytest.raises(ValueError, match='the same prediction is the cheapest for every training sample'):
            model.fit(X, y, tp_cost=2.0, fn_cost=1.0, fp_cost=1.0)

    def test_one_sided_costs_still_fit(self, cost_dataset):
        """Only ``fp_cost`` set: positives have zero weight but keep their label, so CatBoost sees two classes."""
        catboost = pytest.importorskip('catboost')
        X, y, _, _ = cost_dataset
        model = CSBoostClassifier(catboost.CatBoostClassifier(n_estimators=5, verbose=False, allow_writing_files=False))
        y_proba = model.fit(X, y, fp_cost=1.0).predict_proba(X)
        # Only false positives cost anything, so the model must lean negative.
        assert y_proba[:, 1].mean() < 0.5
