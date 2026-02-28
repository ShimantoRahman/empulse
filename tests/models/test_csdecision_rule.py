import numpy as np
import pytest
from sklearn import config_context
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from empulse.metrics import CostMatrix, MaxProfit, Metric
from empulse.models import CSLogitClassifier, CSRateClassifier, CSThresholdClassifier


@pytest.fixture
def data():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    return X, y


# ---------------------------------------------------------------------------
# Dummy helpers for custom-loss tests
# ---------------------------------------------------------------------------


class _DummyCostStrategy:
    pass


class _DummyCostMatrix(CostMatrix):
    def __init__(self):
        self._defaults = {}


class _DummyMetric(Metric):
    """Metric stub whose ``_all_symbols`` contains ``'alpha'``."""

    _all_symbols = {'alpha'}  # noqa: RUF012

    def __init__(self, strategy):
        self.strategy = strategy
        self.cost_matrix = _DummyCostMatrix()

    def optimal_threshold(self, y_true, y_score, **kwargs):
        return 0.5

    def optimal_rate(self, y_true, y_score, **kwargs):
        return 0.3


def _make_threshold_model(kind, loss=None):
    """Create a CSThresholdClassifier wrapping *kind* of estimator."""
    if kind == 'logreg':
        base = LogisticRegression(max_iter=200)
    elif kind == 'cslogit':
        with config_context(enable_metadata_routing=True):
            base = CSLogitClassifier().set_fit_request(fp_cost=True, fn_cost=True, tn_cost=True, tp_cost=True)
    else:
        raise ValueError(f'Unknown kind: {kind}')
    return CSThresholdClassifier(base, calibrator=None, loss=loss)


def _make_rate_model(kind, loss=None):
    """Create a CSRateClassifier wrapping *kind* of estimator."""
    if kind == 'logreg':
        base = LogisticRegression(max_iter=200)
    elif kind == 'cslogit':
        with config_context(enable_metadata_routing=True):
            base = CSLogitClassifier().set_fit_request(fp_cost=True, fn_cost=True, tn_cost=True, tp_cost=True)
    else:
        raise ValueError(f'Unknown kind: {kind}')
    return CSRateClassifier(base, loss=loss)


# ===================================================================
# CSThresholdClassifier-specific tests
# ===================================================================


class TestCSThresholdCalibration:
    """Tests specific to the calibration functionality of CSThresholdClassifier."""

    @pytest.mark.parametrize('calibrator', ['sigmoid', 'isotonic', None])
    def test_fit_with_calibrator(self, data, calibrator):
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(max_iter=2), calibrator=calibrator)
        clf.fit(X, y)
        assert hasattr(clf, 'estimator_')

    @pytest.mark.parametrize('calibrator', ['sigmoid', 'isotonic', None])
    def test_fit_with_costs_and_calibrator(self, data, calibrator):
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(max_iter=2), calibrator=calibrator)
        clf.fit(X, y, fp_cost=1.0, fn_cost=2.0)
        assert hasattr(clf, 'decision_')


# ===================================================================
# CSRateClassifier-specific tests
# ===================================================================


class TestCSRateSpecific:
    """Tests for behaviour unique to CSRateClassifier."""

    def test_rate_bounded(self, data):
        """Fitted rate_ must be between 0 and 1."""
        X, y = data
        clf = CSRateClassifier(LogisticRegression(max_iter=200))
        clf.fit(X, y, fp_cost=1.0, fn_cost=2.0)
        assert 0.0 <= clf.decision_ <= 1.0

    def test_nan_rate_falls_back_to_estimator(self, data):
        """When rate_ is NaN, predict falls back to the estimator's default predict."""
        X, y = data
        clf = CSRateClassifier(LogisticRegression(max_iter=200))
        clf.fit(X, y, fp_cost=1.0, fn_cost=1.0)
        # Manually set rate_ to NaN to exercise the guard.
        clf.decision_ = float('nan')
        y_pred = clf.predict(X)
        expected = clf.estimator_.predict(X)
        np.testing.assert_array_equal(y_pred, expected)

    def test_different_fit_costs_change_rate(self, data):
        """Different costs at fit time should produce different rates."""
        X, y = data
        clf_a = CSRateClassifier(LogisticRegression(max_iter=200))
        clf_a.fit(X, y, fn_cost=10.0, fp_cost=1.0)

        clf_b = CSRateClassifier(LogisticRegression(max_iter=200))
        clf_b.fit(X, y, fn_cost=1.0, fp_cost=10.0)

        assert clf_a.decision_ != clf_b.decision_


# ===================================================================
# Shared behaviour tests (both classifiers)
# ===================================================================

CLASSIFIERS = ['threshold', 'rate']


def _make_model(classifier_type, kind='logreg', loss=None):
    if classifier_type == 'threshold':
        return _make_threshold_model(kind, loss=loss)
    else:
        return _make_rate_model(kind, loss=loss)


class TestFitWithCosts:
    """Both classifiers learn a decision attribute when costs are provided at fit time."""

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize('kind', ['logreg', 'cslogit'])
    def test_fit_standard_costs(self, classifier_type, kind, data):
        X, y = data
        with config_context(enable_metadata_routing=True):
            clf = _make_model(classifier_type, kind)
            clf.set_fit_request(fp_cost=True, fn_cost=True, tn_cost=True, tp_cost=True)
            clf.fit(X, y, fp_cost=1.0, fn_cost=2.0)
        assert hasattr(clf, 'decision_')

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_fit_costs_via_init(self, classifier_type, data):
        """Init costs are used when no fit-time costs are passed."""
        X, y = data
        clf = _make_model(classifier_type)
        clf.set_params(fp_cost=1.0, fn_cost=2.0)
        clf.fit(X, y)
        assert hasattr(clf, 'decision_')

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize('kind', ['logreg', 'cslogit'])
    def test_custom_loss_fit(self, classifier_type, kind, data):
        X, y = data
        loss = _DummyMetric(_DummyCostStrategy())
        clf = _make_model(classifier_type, kind, loss=loss)
        clf.fit(X, y, alpha=0.1)
        assert hasattr(clf, 'decision_')

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize('kind', ['logreg', 'cslogit'])
    def test_maxprofit_fit(self, classifier_type, kind, data):
        """MaxProfit strategy can be used at fit time (requires training data)."""
        X, y = data
        loss = _DummyMetric(MaxProfit())
        clf = _make_model(classifier_type, kind, loss=loss)
        clf.fit(X, y, alpha=0.1)
        assert hasattr(clf, 'decision_')


class TestSkipCostSensitiveFit:
    """When no costs are provided, cost-sensitive fitting is skipped."""

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_fit_no_costs_skips_decision(self, classifier_type, data):
        """Without any costs the classifier fits but does not compute a decision."""
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y)
        assert hasattr(clf, 'estimator_')
        assert not hasattr(clf, 'decision_')

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_predict_without_decision_raises(self, classifier_type, data):
        """Predicting without a fitted decision and without predict-time costs raises."""
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y)
        with pytest.raises(ValueError, match='has not been set during fit'):
            clf.predict(X)

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_custom_loss_no_params_skips(self, classifier_type, data):
        """Custom loss without params and without defaults skips cost-sensitive fit."""
        X, y = data
        loss = _DummyMetric(_DummyCostStrategy())
        clf = _make_model(classifier_type, loss=loss)
        clf.fit(X, y)
        assert hasattr(clf, 'estimator_')
        assert not hasattr(clf, 'decision_')


class TestPredictTimeCosts:
    """Both classifiers can recompute the decision at predict time."""

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize(
        'tp_cost, tn_cost, fn_cost, fp_cost',
        [
            (1.0, 0.0, 5.0, 1.0),
            (0.0, 1.0, 1.0, 5.0),
            (2.0, 2.0, 2.0, 2.0),
            (1.0, 1.0, 1.0, 1.0),
        ],
    )
    def test_predict_with_costs(self, classifier_type, data, tp_cost, tn_cost, fn_cost, fp_cost):
        """Standard costs can be provided at predict time."""
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y)
        y_pred = clf.predict(X, tp_cost=tp_cost, tn_cost=tn_cost, fn_cost=fn_cost, fp_cost=fp_cost)
        assert y_pred.shape == y.shape

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_predict_costs_change_predictions(self, classifier_type, data):
        """Different costs at predict time should yield different predictions."""
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y)
        y_pred_a = clf.predict(X, fn_cost=10.0, fp_cost=1.0)
        y_pred_b = clf.predict(X, fn_cost=1.0, fp_cost=10.0)
        assert not np.array_equal(y_pred_a, y_pred_b)

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize('kind', ['logreg', 'cslogit'])
    def test_predict_standard_costs(self, classifier_type, kind, data):
        X, y = data
        clf = _make_model(classifier_type, kind)
        clf.fit(X, y)
        y_pred = clf.predict(X, fp_cost=1.0, fn_cost=2.0)
        assert y_pred.shape == y.shape

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize('kind', ['logreg', 'cslogit'])
    def test_custom_loss_predict_time_params(self, classifier_type, kind, data):
        """Custom loss params can be passed at predict time."""
        X, y = data
        loss = _DummyMetric(_DummyCostStrategy())
        clf = _make_model(classifier_type, kind, loss=loss)
        clf.fit(X, y)
        y_pred = clf.predict(X, alpha=0.2)
        assert y_pred.shape == y.shape

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    @pytest.mark.parametrize('kind', ['logreg', 'cslogit'])
    def test_maxprofit_predict_raises(self, classifier_type, kind, data):
        """MaxProfit strategy cannot be used at predict time for either classifier."""
        X, y = data
        loss = _DummyMetric(MaxProfit())
        clf = _make_model(classifier_type, kind, loss=loss)
        clf.fit(X, y, alpha=0.1)
        with pytest.raises(ValueError, match='MaxProfit'):
            clf.predict(X, alpha=0.2)


class TestPredictUsesFittedDecision:
    """When costs were provided at fit time, predict uses the fitted decision."""

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_predict_after_fit_with_costs(self, classifier_type, data):
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y, fp_cost=1.0, fn_cost=2.0)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_predict_after_custom_loss_fit(self, classifier_type, data):
        X, y = data
        loss = _DummyMetric(_DummyCostStrategy())
        clf = _make_model(classifier_type, loss=loss)
        clf.fit(X, y, alpha=0.1)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


class TestFeaturePropagation:
    """After fitting, n_features_in_ and feature_names_in_ are propagated."""

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_n_features_in(self, classifier_type, data):
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y)
        assert clf.n_features_in_ == X.shape[1]

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_n_features_in_with_costs(self, classifier_type, data):
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y, fp_cost=1.0, fn_cost=2.0)
        assert clf.n_features_in_ == X.shape[1]


class TestMetadataRouting:
    """Metadata routing: estimator fit params are forwarded correctly."""

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_routing_with_cost_sensitive_estimator(self, classifier_type, data):
        """When the inner estimator requests costs, they are routed through."""
        X, y = data
        with config_context(enable_metadata_routing=True):
            inner = CSLogitClassifier().set_fit_request(fp_cost=True, fn_cost=True, tn_cost=True, tp_cost=True)
            if classifier_type == 'threshold':
                clf = CSThresholdClassifier(inner, calibrator=None)
            else:
                clf = CSRateClassifier(inner)
            clf.set_fit_request(fp_cost=True, fn_cost=True, tn_cost=True, tp_cost=True)
            clf.fit(X, y, fp_cost=1.0, fn_cost=2.0)
        assert hasattr(clf, 'decision_')
