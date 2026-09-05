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
from empulse.models.cost_sensitive.csboost import _BASE_SCORE_PROBA, _BASE_SCORE_RAW

# Define the classifiers to test
CLASSIFIERS = [('xgboost', 'XGBClassifier'), ('lightgbm', 'LGBMClassifier'), ('catboost', 'CatBoostClassifier')]


@pytest.fixture(scope='module')
def dataset():
    X, y = make_classification(n_samples=50, random_state=42)
    fn_cost = np.random.rand(y.size)
    fp_cost = 5
    return X, y, fn_cost, fp_cost


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
def test_csboost_different_classifiers(library, classifier_name, dataset):
    # Import the classifier dynamically
    classifier_module = pytest.importorskip(library)
    classifier_class = getattr(classifier_module, classifier_name)

    X, y, fn_cost, fp_cost = dataset
    model = CSBoostClassifier(estimator=classifier_class(n_estimators=2, verbose=0))
    model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    assert y_pred.shape == y.shape
    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


def test_csboost_when_xgboost_is_missing(dataset):
    X, y, fn_cost, fp_cost = dataset
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'XGBClassifier', TypeVar('XGBClassifier')):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.'):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_when_lightgbm_is_missing_with_lgbm_estimator(dataset):
    """Test that CSBoostClassifier raises ValueError when LightGBM is missing but LGBMClassifier is passed."""
    X, y, fn_cost, fp_cost = dataset

    # Mock LGBMClassifier to be TypeVar (simulating it's not installed)
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'LGBMClassifier', TypeVar('LGBMClassifier')):
        # Create a mock estimator that would fail the isinstance check
        mock_estimator = mock.Mock()
        model = CSBoostClassifier(estimator=mock_estimator)

        with pytest.raises(
            TypeError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
        ):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_when_catboost_is_missing_with_catboost_estimator(dataset):
    """Test that CSBoostClassifier raises ValueError when CatBoost is missing but CatBoostClassifier is passed."""
    X, y, fn_cost, fp_cost = dataset

    # Mock CatBoostClassifier to be TypeVar (simulating it's not installed)
    with mock.patch.object(empulse.models.cost_sensitive.csboost, 'CatBoostClassifier', TypeVar('CatBoostClassifier')):
        # Create a mock estimator that would fail the isinstance check
        mock_estimator = mock.Mock()
        model = CSBoostClassifier(estimator=mock_estimator)

        with pytest.raises(
            TypeError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
        ):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_with_invalid_estimator_type(dataset):
    """Test that CSBoostClassifier raises ValueError when an unsupported estimator type is provided."""
    X, y, fn_cost, fp_cost = dataset

    from sklearn.ensemble import RandomForestClassifier

    model = CSBoostClassifier(estimator=RandomForestClassifier())

    with pytest.raises(
        TypeError, match=r'Estimator must be an instance of XGBClassifier, LGBMClassifier, or CatBoostClassifier'
    ):
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


def test_csboost_with_deterministic_max_profit_metric(dataset):
    xgboost = pytest.importorskip('xgboost')
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), MaxProfit(alpha=1.0))

    X, y, _, _ = dataset
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
def test_csboost_fits_with_renamed_prebuilt_metrics(dataset, metric):
    """Regression test for CSBoostClassifier dispatching on `strategy.name`.

    Every prebuilt metric renames its strategy via `Metric.__name__` (e.g.
    `empc_score.__name__ = 'empc_score'`), which used to make `_get_objective` take the wrong
    branch and raise `NotImplementedError` for MaxProfit/LogCost-backed metrics. All prebuilt
    metrics used here have defaults for every parameter, so no `loss_params` are needed.
    """
    xgboost = pytest.importorskip('xgboost')
    X, y, _, _ = dataset
    model = CSBoostClassifier(estimator=xgboost.XGBClassifier(n_estimators=2, max_depth=1, verbosity=0), loss=metric)
    model.fit(X, y)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('strategy_factory', [MaxProfit, LogCost])
def test_csboost_dispatch_ignores_strategy_name(dataset, strategy_factory):
    """Renaming a custom metric must not change which boosting objective is built.

    Before the fix, `CSBoostClassifier._get_objective` compared `loss.strategy.name` against
    `{'max profit', 'log cost'}`; renaming the metric (as every prebuilt metric does) made it
    silently take the constant-gradient branch instead of raising, or - for a strategy that does
    implement `prepare_boost_objective` - would have trained on the wrong gradients silently.
    """
    xgboost = pytest.importorskip('xgboost')
    clv = sympy.symbols('clv')
    metric = Metric(CostMatrix().add_tp_benefit(clv), strategy_factory())
    metric.__name__ = 'renamed_metric'
    assert metric.strategy.name == 'renamed_metric'

    X, y, _, _ = dataset
    model = CSBoostClassifier(
        estimator=xgboost.XGBClassifier(n_estimators=2, max_depth=1, verbosity=0),
        loss=metric,
    )
    model.fit(X, y, clv=5.0)
    y_proba = model.predict_proba(X)

    assert y_proba.shape == (X.shape[0], len(np.unique(y)))


def test_csboost_when_all_libraries_missing(dataset):
    """Test that CSBoostClassifier fails gracefully when all boosting libraries are missing."""
    X, y, fn_cost, fp_cost = dataset

    with (
        mock.patch.object(empulse.models.cost_sensitive.csboost, 'XGBClassifier', TypeVar('XGBClassifier')),
        mock.patch.object(empulse.models.cost_sensitive.csboost, 'LGBMClassifier', TypeVar('LGBMClassifier')),
        mock.patch.object(empulse.models.cost_sensitive.csboost, 'CatBoostClassifier', TypeVar('CatBoostClassifier')),
    ):
        model = CSBoostClassifier()
        with pytest.raises(ImportError, match=r'XGBoost package is required to use CSBoostClassifier.'):
            model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)


@pytest.mark.parametrize(
    'missing_library,library_name',
    [
        ('XGBClassifier', 'XGBoost'),
        ('LGBMClassifier', 'LightGBM'),
        ('CatBoostClassifier', 'CatBoost'),
    ],
)
def test_csboost_import_error_message_quality(dataset, missing_library, library_name):
    """Test that import error messages are informative and include installation instructions."""
    X, y, fn_cost, fp_cost = dataset

    with mock.patch.object(empulse.models.cost_sensitive.csboost, missing_library, TypeVar(missing_library)):
        if missing_library == 'XGBClassifier':
            model = CSBoostClassifier()
            with pytest.raises(ImportError) as exc_info:
                model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

            # Check that the error message contains helpful information
            error_message = str(exc_info.value)
            assert 'required' in error_message.lower()
            assert 'install' in error_message.lower() or 'pip install' in error_message.lower()


def test_csboost_fit_does_not_mutate_callers_fit_params_dict(dataset):
    """Regression test: `_fit` used to mutate the caller's `fit_params` dict in place.

    Reusing one `fit_params` dict across two `fit()` calls (or across GridSearchCV folds) would
    silently carry `sample_weight` from the first call into the second, since `sample_weight`
    (passed as a loss-param kwarg) was popped straight into the caller-supplied `fit_params` dict.
    Uses LightGBM rather than XGBoost: XGBoost's custom-objective path does not support
    `sample_weight` at all (a separate, pre-existing limitation unrelated to this bug).
    """
    lightgbm = pytest.importorskip('lightgbm')
    X, y, fn_cost, fp_cost = dataset

    shared_fit_params: dict = {}
    model1 = CSBoostClassifier(estimator=lightgbm.LGBMClassifier(n_estimators=2, max_depth=1, verbosity=-1))
    model1.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost, fit_params=shared_fit_params, sample_weight=np.ones(len(y)))

    # The caller's dict must be untouched - it was empty going in and must still be empty.
    assert shared_fit_params == {}

    model2 = CSBoostClassifier(estimator=lightgbm.LGBMClassifier(n_estimators=2, max_depth=1, verbosity=-1))
    # Second call reuses the same (still-empty) dict and passes no sample_weight at all.
    model2.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost, fit_params=shared_fit_params)
    assert shared_fit_params == {}


class TestBaseScoreSpace:
    """Regression tests for _BASE_SCORE meaning a probability for XGBoost but a raw score elsewhere.

    XGBoost's `base_score` is documented as a probability; LightGBM's `init_score` and CatBoost's
    `baseline` are raw (log-odds) scores. Using the same literal constant for all three meant
    `expit(0.51) != 0.51` was silently ignored, and neither LightGBM nor CatBoost persist that
    offset into the saved model, so `predict_proba` must add it back manually before `expit`.
    """

    def test_base_score_constants_are_logit_pairs(self):
        """_BASE_SCORE_RAW must be the logit of _BASE_SCORE_PROBA, not the same literal value."""
        assert _BASE_SCORE_RAW != _BASE_SCORE_PROBA
        assert expit(_BASE_SCORE_RAW) == pytest.approx(_BASE_SCORE_PROBA)

    def test_lightgbm_predict_proba_reconstructs_raw_offset(self, dataset):
        """predict_proba must equal expit(raw_score + _BASE_SCORE_RAW), not expit(raw_score) alone.

        LightGBM does not persist `init_score` into the trained model, so the offset used at fit
        time must be added back manually at predict time.
        """
        lightgbm = pytest.importorskip('lightgbm')
        X, y, fn_cost, fp_cost = dataset
        model = CSBoostClassifier(estimator=lightgbm.LGBMClassifier(n_estimators=5, max_depth=2, verbosity=-1))
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

        raw_score = model.estimator_.predict_proba(X, raw_score=True)
        expected_proba = expit(raw_score + _BASE_SCORE_RAW)
        actual_proba = model.predict_proba(X)[:, 1]

        np.testing.assert_allclose(actual_proba, expected_proba)
        # Regression guard: the old (buggy) reconstruction without the offset must NOT match.
        assert not np.allclose(actual_proba, expit(raw_score))

    def test_catboost_predict_proba_reconstructs_raw_offset(self, dataset):
        """predict_proba must equal expit(raw_score + _BASE_SCORE_RAW), not expit(raw_score) alone.

        CatBoost does not persist `baseline` into the trained model either, and its predict/
        predict_proba have no way to resupply it at predict time, so it must be reconstructed
        manually from the raw formula value.
        """
        catboost = pytest.importorskip('catboost')
        X, y, fn_cost, fp_cost = dataset
        model = CSBoostClassifier(
            estimator=catboost.CatBoostClassifier(n_estimators=5, max_depth=2, verbose=False, allow_writing_files=False)
        )
        model.fit(X, y, fn_cost=fn_cost, fp_cost=fp_cost)

        raw_score = model.estimator_.predict(X, prediction_type='RawFormulaVal')
        expected_proba = expit(raw_score + _BASE_SCORE_RAW)
        actual_proba = model.predict_proba(X)[:, 1]

        np.testing.assert_allclose(actual_proba, expected_proba)
        assert not np.allclose(actual_proba, expit(raw_score))

    @pytest.mark.parametrize('library, classifier_name', CLASSIFIERS)
    def test_predict_proba_starts_near_intended_probability_with_minimal_learning(self, library, classifier_name):
        """With a near-zero learning rate, every backend's baseline probability should be ~0.51.

        This isolates the `_BASE_SCORE` initialization from each backend's own tree-growing
        algorithm (which otherwise dominates a three-way comparison and makes it a noisy test):
        with learning_rate effectively disabling any real tree contribution, predict_proba should
        reflect (approximately) the shared starting probability, ~0.51, regardless of backend.
        """
        classifier_module = pytest.importorskip(library)
        classifier_class = getattr(classifier_module, classifier_name)
        X, y = make_classification(n_samples=100, random_state=0)

        kwargs = {'n_estimators': 3, 'max_depth': 1}
        if library == 'catboost':
            kwargs.update(learning_rate=1e-6, verbose=False, allow_writing_files=False)
        elif library == 'lightgbm':
            kwargs.update(learning_rate=1e-6, verbosity=-1)
        else:
            kwargs.update(learning_rate=1e-6, verbosity=0)

        model = CSBoostClassifier(estimator=classifier_class(**kwargs))
        model.fit(X, y, fp_cost=1.0, fn_cost=1.0)
        mean_proba = model.predict_proba(X)[:, 1].mean()

        assert mean_proba == pytest.approx(_BASE_SCORE_PROBA, abs=0.02)
