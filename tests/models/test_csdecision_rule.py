import warnings

import numpy as np
import pytest
import sympy
from sklearn import config_context
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from empulse.metrics import (
    Capability,
    Cost,
    CostMatrix,
    EmpiricalMaxProfit,
    MaxProfit,
    Metric,
    MixtureComponent,
    MixtureMetric,
)
from empulse.metrics.metric.cost_matrix import _symbol_names
from empulse.models import CSLogitClassifier, CSRateClassifier, CSThresholdClassifier


@pytest.fixture(scope='module')
def data(make_data):
    return make_data(n_samples=1000, n_features=20)


# ---------------------------------------------------------------------------
# Dummy helpers for custom-loss tests
# ---------------------------------------------------------------------------


class _DummyCostStrategy:
    # Metric._prepare_parameters reads this to decide which keyword arguments are unknown.
    # MetricStrategy supplies an empty set; this stub mirrors that default.
    _extra_kwargs: frozenset[str] = frozenset()

    # csdecision_rule.py's predict-time gate reads this to decide whether a decision can be
    # recomputed from cost parameters alone, like Cost's can -- which is exactly what this stub
    # stands in for (a Cost-like, non-ranking-based strategy).
    capabilities: frozenset[Capability] = frozenset({Capability.COST_ONLY_DECISION})


class _DummyMetric(Metric):
    """Metric stub whose ``_all_symbols`` contains ``'alpha'``."""

    _all_symbols = {'alpha'}  # ruff: ignore[mutable-class-default]

    def __init__(self, strategy):
        self._strategy = strategy
        self._cost_matrix = CostMatrix().add_fp_cost('alpha').add_fn_cost('alpha')
        self._symbols = _symbol_names(self._cost_matrix.fp_cost, self._cost_matrix.fn_cost)

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

    def test_custom_calibrator_instance_not_mutated_by_fit(self, data):
        """A user-supplied calibrator object must not have its `estimator` set in place.

        `_get_calibrator` must not call `self.calibrator.set_params(estimator=estimator)` directly on the
        constructor argument: fitting (or refitting with a clone, e.g. inside cross-validation) would bind
        a fitted estimator into the object the caller still holds a reference to.
        """
        from sklearn.calibration import CalibratedClassifierCV

        custom_calibrator = CalibratedClassifierCV(cv=3)
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(max_iter=2), calibrator=custom_calibrator)
        clf.fit(X, y)
        assert custom_calibrator.estimator is None


# ===================================================================
# CSRateClassifier-specific tests
# ===================================================================


def test_cs_threshold_multiclass_target_is_rejected(data):
    X, y = data
    model = CSThresholdClassifier(estimator=LogisticRegression(max_iter=10), fp_cost=1, fn_cost=1, random_state=42)
    with pytest.raises(ValueError, match='Only binary classification is supported'):
        model.fit(X, np.arange(len(y)) % 3)


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
WRAPPERS = {
    'threshold': lambda estimator: CSThresholdClassifier(estimator, calibrator=None),
    'rate': CSRateClassifier,
}


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


class TestDecisionDirection:
    """The threshold follows the cost-optimal decision, whatever the signs of the costs."""

    def test_targeting_nobody_excludes_probabilities_of_exactly_one(self, data):
        """With fp_cost=1 and tp_cost=3 targeting never pays, so a tree's probabilities of 1.0 stay negative."""
        X, y = data
        clf = CSThresholdClassifier(DecisionTreeClassifier(random_state=0), calibrator=None, tp_cost=3.0, fp_cost=1.0)
        with pytest.warns(UserWarning, match='every sample or none'):
            clf.fit(X, y)

        assert clf.threshold_ == np.inf
        assert (clf.estimator_.predict_proba(X)[:, 1] == 1.0).any()
        assert not clf.predict(X).any()

    def test_costs_favouring_the_least_likely_positives_raise(self, data):
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None, fp_cost=-1.0, fn_cost=-1.0)

        with pytest.raises(ValueError, match='less likely it is to be positive'):
            clf.fit(X, y)


class TestInstanceDependentCostsReduceToTheirAverage:
    """The single fitted threshold uses each cost averaged over the samples, not the averaged parameters."""

    def test_cost_of_two_instance_dependent_parameters(self, data):
        X, y = data
        a, b = sympy.symbols('a b')
        loss = Metric(CostMatrix().add_fp_cost(a * b).add_fn_cost(30), Cost())
        # a * b is 9 for every sample, while mean(a) * mean(b) is 25.
        a_values = np.where(np.arange(len(y)) % 2 == 0, 1.0, 9.0)
        b_values = 10.0 - a_values

        clf = CSThresholdClassifier(LogisticRegression(), loss=loss, calibrator=None).fit(X, y, a=a_values, b=b_values)

        expected = loss.optimal_threshold(np.array([]), np.array([]), a=9.0, b=1.0)
        assert clf.threshold_ == pytest.approx(expected)

    def test_the_same_holds_for_a_mixture(self, data):
        """A MixtureMetric of such a component thresholds exactly like the component on its own."""
        X, y = data
        a, b = sympy.symbols('a b')
        component = Metric(CostMatrix().add_fp_cost(a * b).add_fn_cost(30), Cost())
        mixture = MixtureMetric([MixtureComponent(1.0, component, {})])
        a_values = np.where(np.arange(len(y)) % 2 == 0, 1.0, 9.0)
        b_values = 10.0 - a_values

        on_metric = CSThresholdClassifier(LogisticRegression(), loss=component, calibrator=None)
        on_mixture = CSThresholdClassifier(LogisticRegression(), loss=mixture, calibrator=None)

        assert on_mixture.fit(X, y, a=a_values, b=b_values).threshold_ == pytest.approx(
            on_metric.fit(X, y, a=a_values, b=b_values).threshold_
        )


def test_threshold_of_a_cost_mixture_matches_its_combined_metric(data):
    """A mixture of Cost components thresholds like one Cost metric with the weighted costs."""
    X, y = data
    a, b = sympy.symbols('a b')
    metric = Metric(CostMatrix().add_fp_cost(a).add_fn_cost(b), Cost())
    mixture = MixtureMetric([
        MixtureComponent('w', metric, {'a': 1.0, 'b': 9.0}),
        MixtureComponent('1 - w', metric, {'a': 9.0, 'b': 1.0}),
    ])

    mixed = CSThresholdClassifier(LogisticRegression(), loss=mixture, calibrator=None).fit(X, y, w=0.5)
    combined = CSThresholdClassifier(LogisticRegression(), loss=metric, calibrator=None).fit(X, y, a=5.0, b=5.0)

    assert mixed.threshold_ == pytest.approx(combined.threshold_)
    np.testing.assert_array_equal(mixed.predict(X), combined.predict(X))


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
            (2.0, 2.0, 2.0, 3.0),
            (1.0, 0.5, 1.0, 1.0),
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
        with pytest.raises(ValueError, match="'max profit' at predict time"):
            clf.predict(X, alpha=0.2)

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_empirical_max_profit_predict_raises(self, classifier_type, data):
        """
        A ranking-based strategy other than MaxProfit also raises a clear error at predict time.

        Before Capability.COST_ONLY_DECISION existed, only ``isinstance(loss.strategy, MaxProfit)``
        was checked here, so an EmpiricalMaxProfit (or AUEPC) loss fell through to
        ``optimal_threshold(np.array([]), np.array([]))`` -- calling a ranking-based computation on
        empty arrays -- instead of raising.
        """
        X, y = data
        loss = Metric(CostMatrix().add_tp_benefit('a').add_fp_cost('b'), EmpiricalMaxProfit())
        clf = _make_model(classifier_type, 'logreg', loss=loss)
        clf.fit(X, y, a=5.0, b=1.0)
        with pytest.raises(ValueError, match="'empirical max profit' at predict time"):
            clf.predict(X, a=5.0, b=1.0)

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_predict_scores_estimator_only_once(self, classifier_type, data):
        """
        `predict` must not call the base estimator's `predict_proba` twice on the same X.

        Scoring `X` a second time in `_apply_decision` would double the cost of an expensive base
        estimator.
        """
        X, y = data
        clf = _make_model(classifier_type)
        clf.fit(X, y)
        call_count = 0
        original_predict_proba = clf.estimator_.predict_proba

        def counting_predict_proba(X_inner):
            nonlocal call_count
            call_count += 1
            return original_predict_proba(X_inner)

        clf.estimator_.predict_proba = counting_predict_proba
        clf.predict(X, fp_cost=1.0, fn_cost=2.0)
        assert call_count == 1


class _RecordingMetric(Metric):
    """Metric stub whose ``_all_symbols`` contains ``'alpha'`` and that records which object
    served each ``optimal_threshold``/``optimal_rate`` call, so a test can tell whether the call
    was served by the per-fit deep copy (``self._loss``) or by the shared original instance.
    """

    _all_symbols = {'alpha'}  # ruff: ignore[mutable-class-default]

    def __init__(self, strategy):
        self._strategy = strategy
        self._cost_matrix = CostMatrix().add_fp_cost('alpha').add_fn_cost('alpha')
        self._symbols = _symbol_names(self._cost_matrix.fp_cost, self._cost_matrix.fn_cost)
        self.call_log = []

    def optimal_threshold(self, y_true, y_score, **kwargs):
        self.call_log.append(self)
        return 0.5

    def optimal_rate(self, y_true, y_score, **kwargs):
        self.call_log.append(self)
        return 0.3


class TestPredictLossFitCopyIsolation:
    """`predict()` must consult the per-fit loss copy (`self._loss`), not the shared
    `self.loss` object, when recomputing the decision from predict-time loss params.

    Two estimators constructed with the same shared metric instance (as happens with a
    module-level prebuilt metric such as `empc_score`) must not read or write each other's
    metric state through predict-time recomputation of the decision.
    """

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_predict_does_not_touch_shared_loss_instance(self, classifier_type, data):
        X, y = data
        shared_loss = _RecordingMetric(_DummyCostStrategy())
        clf_a = _make_model(classifier_type, loss=shared_loss)
        clf_b = _make_model(classifier_type, loss=shared_loss)
        clf_a.fit(X, y, alpha=0.1)
        clf_b.fit(X, y, alpha=0.1)

        clf_a.predict(X, alpha=0.2)
        clf_b.predict(X, alpha=0.3)

        assert shared_loss.call_log == [], (
            'predict() consulted the shared self.loss object instead of the per-fit copy self._loss'
        )
        assert len(clf_a._loss.call_log) == 2  # one from fit, one from predict
        assert all(obj is clf_a._loss for obj in clf_a._loss.call_log)
        assert len(clf_b._loss.call_log) == 2
        assert all(obj is clf_b._loss for obj in clf_b._loss.call_log)


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


class TestAliasedMetric:
    """
    Cost-sensitive fitting with an aliased Metric.

    `_should_skip_cost_sensitive_fit` must not compare `loss._all_parameters` (which lists both a
    symbol's raw name and its alias) for equality against the caller's supplied keys. A caller only
    supplies one spelling per parameter, so that equality can never hold for an aliased metric: the
    model would skip learning a threshold/rate and forward the loss's kwargs straight to the base
    estimator's `fit`, which doesn't recognize them.
    """

    @staticmethod
    def _make_aliased_metric():
        clv, d = sympy.symbols('clv d')
        return Metric(CostMatrix().add_fn_cost(clv).add_fp_cost(d).alias({'incentive_cost': 'd'}), Cost())

    @staticmethod
    def _make_unaliased_metric():
        clv, d = sympy.symbols('clv d')
        return Metric(CostMatrix().add_fn_cost(clv).add_fp_cost(d), Cost())

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_aliased_metric_fits_cost_sensitively(self, classifier_type, data):
        """Fitting with every parameter supplied via its alias must not skip the decision."""
        X, y = data
        clf = _make_model(classifier_type, loss=self._make_aliased_metric())
        clf.fit(X, y, clv=100.0, incentive_cost=10.0)
        assert hasattr(clf, 'decision_')
        assert clf.decision_ is not None
        assert not (isinstance(clf.decision_, float) and np.isnan(clf.decision_))

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_aliased_and_unaliased_metric_agree(self, classifier_type, data):
        """The alias is purely cosmetic: it must not change the fitted decision."""
        X, y = data
        clf_aliased = _make_model(classifier_type, loss=self._make_aliased_metric())
        clf_aliased.fit(X, y, clv=100.0, incentive_cost=10.0)

        clf_unaliased = _make_model(classifier_type, loss=self._make_unaliased_metric())
        clf_unaliased.fit(X, y, clv=100.0, d=10.0)

        assert clf_aliased.decision_ == pytest.approx(clf_unaliased.decision_)

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_extra_kwarg_does_not_disable_cost_sensitive_fit(self, classifier_type, data):
        """An unrelated extra kwarg (e.g. routed sample_weight) must not disable the decision."""
        X, y = data
        with config_context(enable_metadata_routing=True):
            base = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True)
            if classifier_type == 'threshold':
                clf = CSThresholdClassifier(base, calibrator=None, loss=self._make_unaliased_metric())
            else:
                clf = CSRateClassifier(base, loss=self._make_unaliased_metric())
            clf.fit(X, y, clv=100.0, d=10.0, sample_weight=np.ones(len(y)))
        assert hasattr(clf, 'decision_')
        assert clf.decision_ is not None

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_partial_aliased_params_do_not_leak_to_estimator(self, classifier_type, data):
        """Supplying only some of an aliased metric's params must not forward them to fit()."""
        X, y = data
        clf = _make_model(classifier_type, loss=self._make_aliased_metric())
        # Only `clv` is supplied; `incentive_cost`/`d` is missing, so cost-sensitive fitting is
        # correctly skipped - but `clv` must not leak into LogisticRegression.fit() as a kwarg.
        clf.fit(X, y, clv=100.0)
        assert hasattr(clf, 'estimator_')
        assert not hasattr(clf, 'decision_')


# ---------------------------------------------------------------------------
# pos_label
# ---------------------------------------------------------------------------

DECISION_RULE_FACTORIES = [
    pytest.param(lambda **kw: CSThresholdClassifier(LogisticRegression(), calibrator=None, **kw), id='threshold'),
    pytest.param(lambda **kw: CSRateClassifier(LogisticRegression(), **kw), id='rate'),
]


class TestPosLabel:
    """``pos_label`` makes its class the positive one.

    The decision is computed from the ``pos_label`` class's probabilities, so ``pos_label=classes_[0]``
    must not invert predictions, for ``CSThresholdClassifier`` and ``CSRateClassifier`` alike.
    """

    @pytest.mark.parametrize('make_model', DECISION_RULE_FACTORIES)
    def test_pos_label_matches_flipping_the_labels(self, data, make_model):
        X, y = data
        with_pos_label = make_model(pos_label=0).fit(X, y, fn_cost=5.0, fp_cost=1.0)
        flipped = make_model().fit(X, 1 - y, fn_cost=5.0, fp_cost=1.0)

        np.testing.assert_array_equal(with_pos_label.predict(X), 1 - flipped.predict(X))

    @pytest.mark.parametrize('make_model', DECISION_RULE_FACTORIES)
    def test_pos_label_matches_flipping_the_labels_with_predict_time_costs(self, data, make_model):
        X, y = data
        with_pos_label = make_model(pos_label=0).fit(X, y)
        flipped = make_model().fit(X, 1 - y)

        np.testing.assert_array_equal(
            with_pos_label.predict(X, fn_cost=5.0, fp_cost=1.0), 1 - flipped.predict(X, fn_cost=5.0, fp_cost=1.0)
        )

    @pytest.mark.parametrize('make_model', DECISION_RULE_FACTORIES)
    def test_pos_label_with_string_labels(self, data, make_model):
        X, y = data
        labels = np.where(y == 1, 'yes', 'no')
        with_pos_label = make_model(pos_label='no').fit(X, labels, fn_cost=5.0, fp_cost=1.0)
        flipped = make_model(pos_label='yes').fit(X, np.where(y == 1, 'no', 'yes'), fn_cost=5.0, fp_cost=1.0)

        np.testing.assert_array_equal(with_pos_label.predict(X), np.where(flipped.predict(X) == 'yes', 'no', 'yes'))

    def test_unknown_pos_label_raises(self, data):
        X, y = data
        model = CSThresholdClassifier(LogisticRegression(), calibrator=None, pos_label=2)
        with pytest.raises(ValueError, match='pos_label=2 is not one of the classes'):
            model.fit(X, y, fn_cost=5.0, fp_cost=1.0)


class TestWrappedEstimatorGetsXAsGiven:
    """X is the wrapped estimator's to validate, so a pipeline can encode strings and keep feature names."""

    @pytest.fixture
    def frame(self, data):
        pd = pytest.importorskip('pandas')
        X, y = data
        frame = pd.DataFrame(X[:, :4], columns=list('abcd'))
        frame['colour'] = np.where(X[:, 4] > 0, 'red', 'blue')
        return frame, y

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_pipeline_encoding_a_string_column(self, classifier_type, frame):
        X, y = frame
        pipeline = make_pipeline(
            ColumnTransformer([('encode', OneHotEncoder(), ['colour'])], remainder='passthrough'),
            LogisticRegression(),
        )
        clf = WRAPPERS[classifier_type](pipeline).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        assert clf.predict(X).shape == y.shape
        assert list(clf.feature_names_in_) == list(X.columns)

    def test_feature_names_reach_the_wrapped_estimator(self, frame):
        X, y = frame
        X = X.drop(columns='colour')
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            clf = CSThresholdClassifier(LogisticRegression(), calibrator=None).fit(X, y, fp_cost=1.0, fn_cost=5.0)
            clf.predict(X)
        assert list(clf.estimator_.feature_names_in_) == list(X.columns)


class TestFitParamsReachTheWrappedEstimator:
    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_pipeline_with_costs(self, classifier_type, data):
        X, y = data
        clf = WRAPPERS[classifier_type](make_pipeline(LogisticRegression()))
        clf.fit(X, y, fp_cost=1.0, fn_cost=5.0)
        assert hasattr(clf, 'decision_')

    @pytest.mark.parametrize('classifier_type', CLASSIFIERS)
    def test_sample_weight_without_routing(self, classifier_type, data):
        X, y = data
        weights = np.linspace(0.5, 1.5, len(y))
        clf = WRAPPERS[classifier_type](LogisticRegression()).fit(X, y, fp_cost=1.0, fn_cost=5.0, sample_weight=weights)
        expected = LogisticRegression().fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(clf.estimator_.coef_, expected.coef_)

    def test_sample_weight_routed_through_a_pipeline(self, data):
        X, y = data
        weights = np.linspace(0.5, 1.5, len(y))
        with config_context(enable_metadata_routing=True):
            pipeline = make_pipeline(LogisticRegression().set_fit_request(sample_weight=True))
            clf = CSThresholdClassifier(pipeline, calibrator=None)
            clf.set_fit_request(fp_cost=True, fn_cost=True)
            clf.fit(X, y, fp_cost=1.0, fn_cost=5.0, sample_weight=weights)
        expected = LogisticRegression().fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(clf.estimator_[-1].coef_, expected.coef_)

    def test_costs_reach_a_cost_sensitive_estimator_without_routing(self, data):
        X, y = data
        clf = CSThresholdClassifier(CSLogitClassifier(), calibrator=None).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        expected = CSLogitClassifier().fit(X, y, fp_cost=1.0, fn_cost=5.0)
        np.testing.assert_allclose(clf.estimator_.coef_, expected.coef_)

    def test_costs_reach_a_cost_sensitive_estimator_inside_the_default_calibrator(self, data):
        """The calibrator forwards its keyword arguments, so the costs must not stop at it."""
        X, y = data
        clf = CSThresholdClassifier(CSLogitClassifier(), random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)
        expected = CSLogitClassifier().fit(X, y, fp_cost=1.0, fn_cost=5.0)
        inner = clf.estimator_.calibrated_classifiers_[0].estimator
        np.testing.assert_allclose(inner.coef_, expected.coef_)

    def test_costs_are_swapped_for_an_estimator_whose_positive_class_is_the_other(self, data):
        """With ``pos_label=0``, a missed class 0 is a false negative for the wrapper but a false positive inside."""
        X, y = data
        clf = CSThresholdClassifier(CSLogitClassifier(), calibrator=None, pos_label=0)
        clf.fit(X, y, fp_cost=1.0, fn_cost=5.0, tp_cost=0.5)
        expected = CSLogitClassifier().fit(X, y, fp_cost=5.0, fn_cost=1.0, tn_cost=0.5)
        np.testing.assert_allclose(clf.estimator_.coef_, expected.coef_)

    @pytest.mark.parametrize(
        ('outer', 'inner', 'swapped'), [(0, 0, True), (0, 1, True), (1, 0, False), (None, 0, False)]
    )
    def test_costs_reach_a_nested_wrapper_stated_for_its_own_positive_class(self, data, outer, inner, swapped):
        """A wrapped estimator with a ``pos_label`` of its own swaps the costs itself, so they reach it unswapped."""
        X, y = data
        nested = CSThresholdClassifier(CSLogitClassifier(), calibrator=None, pos_label=inner)
        clf = CSThresholdClassifier(nested, calibrator=None, pos_label=outer).fit(X, y, fp_cost=1.0, fn_cost=5.0)

        if swapped:
            expected = CSLogitClassifier().fit(X, y, fp_cost=5.0, fn_cost=1.0)
        else:
            expected = CSLogitClassifier().fit(X, y, fp_cost=1.0, fn_cost=5.0)
        np.testing.assert_allclose(clf.estimator_.estimator_.coef_, expected.coef_)

    def test_sample_weight_routed_through_the_default_calibrator(self, data):
        """The calibrator is built internally, so its weight request follows the estimator's."""
        X, y = data
        weights = np.where(y == 1, 3.0, 1.0)
        with config_context(enable_metadata_routing=True):
            estimator = LogisticRegression().set_fit_request(sample_weight=True)
            clf = CSThresholdClassifier(estimator, random_state=0)
            clf.set_fit_request(fp_cost=True, fn_cost=True)
            clf.fit(X, y, fp_cost=1.0, fn_cost=5.0, sample_weight=weights)
        expected = CSThresholdClassifier(LogisticRegression(), random_state=0)
        expected.fit(X, y, fp_cost=1.0, fn_cost=5.0, sample_weight=weights)
        np.testing.assert_allclose(clf.predict_proba(X), expected.predict_proba(X))

    def test_a_non_cost_array_parameter_of_another_length_reaches_the_estimator(self, data):
        X, y = data
        feature_weights = np.ones(X.shape[1])

        class _FeatureWeighted(LogisticRegression):
            def fit(self, X, y, feature_weights=None):
                self.feature_weights_ = feature_weights
                return super().fit(X, y)

        clf = CSThresholdClassifier(_FeatureWeighted(), calibrator=None)
        clf.fit(X, y, fp_cost=1.0, fn_cost=5.0, feature_weights=feature_weights)
        np.testing.assert_array_equal(clf.estimator_.feature_weights_, feature_weights)


class TestDecisionOfAScoreDependentLoss:
    """A threshold that depends on the scores is learned from the costs the parameters make up."""

    def test_instance_dependent_parameters_are_not_averaged_before_they_are_combined(self, data):
        """``tp_benefit = a * b`` with ``b = 1 / a`` is 1 everywhere, which averaging ``a`` and ``b`` first breaks."""
        X, y = data
        a, b = sympy.symbols('a b')
        loss = Metric(CostMatrix().add_tp_benefit(a * b).add_fp_cost(1), MaxProfit())
        a_values = np.random.default_rng(0).uniform(0.5, 4.0, size=len(y))

        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None, loss=loss)
        clf.fit(X, y, a=a_values, b=1 / a_values)

        y_score = clf.predict_proba(X)[:, 1]
        assert clf.threshold_ == pytest.approx(loss.optimal_threshold(y, y_score, a=1.0, b=1.0))


class TestRefitAndPredictTimeCosts:
    """Predict-time costs and refits start from what the latest fit learned."""

    def test_a_cost_not_passed_to_predict_keeps_its_fit_value(self, data):
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None).fit(X, y, fp_cost=5.0, fn_cost=10.0)

        np.testing.assert_array_equal(clf.predict(X, fn_cost=10.0), clf.predict(X))
        np.testing.assert_array_equal(clf.predict(X, fp_cost=10.0), clf.predict(X, fp_cost=10.0, fn_cost=10.0))

    def test_an_instance_dependent_fit_cost_falls_back_to_its_average(self, data):
        X, y = data
        fp_cost = np.linspace(1.0, 9.0, len(y))
        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None).fit(X, y, fp_cost=fp_cost, fn_cost=10.0)

        np.testing.assert_array_equal(clf.predict(X, fn_cost=10.0), clf.predict(X, fp_cost=5.0, fn_cost=10.0))

    def test_a_refit_without_costs_forgets_the_previous_decision(self, data):
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None).fit(X, y, fp_cost=1.0, fn_cost=9.0)
        clf.fit(X[:, :5], y)

        assert clf.threshold_ is None
        with pytest.raises(ValueError, match='decision has not been set during fit'):
            clf.predict(X[:, :5])

    def test_a_fit_without_costs_fits_the_calibrated_estimator(self, data):
        """Fitting without costs, to pass them to predict, still calibrates the probabilities they are applied to."""
        X, y = data
        without_costs = CSThresholdClassifier(LogisticRegression(), random_state=0).fit(X, y)
        with_costs = CSThresholdClassifier(LogisticRegression(), random_state=0).fit(X, y, fp_cost=1.0, fn_cost=5.0)

        np.testing.assert_allclose(without_costs.predict_proba(X), with_costs.predict_proba(X))

    def test_a_predict_time_cost_of_the_wrong_length_is_rejected(self, data):
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None).fit(X, y, fp_cost=1.0, fn_cost=5.0)

        with pytest.raises(ValueError, match="'fn_cost' has length 7, but expected length 10"):
            clf.predict(X[:10], fn_cost=np.full(7, 5.0))


class TestOutOfFoldDecision:
    """With ``cv``, a decision that depends on the scores is learned from out-of-fold scores."""

    def test_threshold_of_a_score_dependent_loss_uses_out_of_fold_scores(self, data):
        from sklearn.model_selection import KFold, cross_val_predict

        X, y = data
        loss = Metric(CostMatrix().add_tp_benefit(10).add_fp_cost(3), MaxProfit())
        cv = KFold(n_splits=4, shuffle=True, random_state=0)

        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None, loss=loss, cv=cv).fit(X, y)

        oof_score = cross_val_predict(LogisticRegression(), X, y, cv=cv, method='predict_proba')[:, 1]
        assert clf.threshold_ == pytest.approx(loss.optimal_threshold(y, oof_score))
        np.testing.assert_allclose(clf.estimator_.coef_, LogisticRegression().fit(X, y).coef_)

    def test_rate_uses_out_of_fold_scores(self, data):
        from sklearn.model_selection import KFold, cross_val_predict

        X, y = data
        loss = Metric(CostMatrix().add_tp_benefit(10).add_fp_cost(3), MaxProfit())
        cv = KFold(n_splits=4, shuffle=True, random_state=0)

        clf = CSRateClassifier(LogisticRegression(), loss=loss, cv=cv).fit(X, y)

        oof_score = cross_val_predict(LogisticRegression(), X, y, cv=cv, method='predict_proba')[:, 1]
        assert clf.rate_ == pytest.approx(loss.optimal_rate(y, oof_score))

    def test_a_cost_only_threshold_ignores_cv(self, data, monkeypatch):
        from empulse.models.threshold import csdecision_rule

        def _fail(*args, **kwargs):
            raise AssertionError('a threshold that depends only on the costs needs no out-of-fold scores')

        monkeypatch.setattr(csdecision_rule, 'cross_val_predict', _fail)
        X, y = data
        clf = CSThresholdClassifier(LogisticRegression(), calibrator=None, cv=5).fit(X, y, fp_cost=1.0, fn_cost=4.0)

        assert clf.threshold_ == pytest.approx(0.2)


class TestRateTargetsExactlyItsShare:
    """CSRate predicts exactly ``ceil(rate * n)`` positives, however many scores tie."""

    def test_a_rate_of_k_over_n_targets_k_of_n_samples(self, data):
        """0.07 * 100 is 7.000000000000001 in floating point, which must not round up to 8."""
        X, y = data
        X, y = X[:100], y[:100]
        clf = CSRateClassifier(LogisticRegression()).fit(X, y, fp_cost=1.0, fn_cost=1.0)

        for k in range(101):
            clf.decision_ = k / 100
            assert clf.predict(X).sum() == k

    @pytest.mark.parametrize('rate', [0.1, 0.25, 0.4, 0.0, 1.0])
    def test_ties_at_the_cut_off_do_not_exceed_the_rate(self, data, rate):
        X, y = data
        clf = CSRateClassifier(DecisionTreeClassifier(max_depth=2, random_state=0)).fit(X, y, fp_cost=1.0, fn_cost=2.0)
        clf.decision_ = rate
        y_score = clf.predict_proba(X)[:, 1]
        assert np.unique(y_score).size < 5

        y_pred = clf.predict(X)

        assert y_pred.sum() == int(np.ceil(rate * len(y)))
        assert y_score[y_pred == 1].min(initial=np.inf) >= y_score[y_pred == 0].max(initial=-np.inf)
